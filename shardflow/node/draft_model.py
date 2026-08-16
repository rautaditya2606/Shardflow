"""
Speculative Decoding Draft Sampler and KV Cache Rewind Utilities.
Enables generating K candidate tokens locally on Node 0 in a single network roundtrip.
"""

import logging
import queue
import threading
import time
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache, StaticCache, Cache
from shardflow.orchestrator.sampler import sample_next_token

logger = logging.getLogger(__name__)


def rewind_dynamic_cache(cache: DynamicCache, target_seq_len: int) -> None:
    """Rewind DynamicCache to target_seq_len by slicing key/value tensors in place."""
    # ponytail: native cache.crop() truncates dynamic cache in a single standard call
    if hasattr(cache, "crop"):
        cache.crop(target_seq_len)
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for layer_idx in range(len(cache.key_cache)):
            if cache.key_cache[layer_idx] is not None:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :target_seq_len, :]
            if cache.value_cache[layer_idx] is not None:
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :target_seq_len, :]


def rewind_static_cache(cache: StaticCache, target_seq_len: int) -> None:
    """Rewind StaticCache to target_seq_len by zeroing out rejected key/value slots."""
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = target_seq_len
    # ponytail: safely zero out all rejected slots under inference_mode
    with torch.inference_mode():
        if hasattr(cache, "layers"):
            for layer in cache.layers:
                if hasattr(layer, "keys") and layer.keys is not None:
                    try:
                        layer.keys[:, :, target_seq_len:, :].zero_()
                    except Exception:
                        pass
                if hasattr(layer, "values") and layer.values is not None:
                    try:
                        layer.values[:, :, target_seq_len:, :].zero_()
                    except Exception:
                        pass


def rewind_kv_cache(cache: Cache, target_seq_len: int) -> None:
    """Universal KV cache rewind for either DynamicCache or StaticCache."""
    if isinstance(cache, StaticCache):
        rewind_static_cache(cache, target_seq_len)
    elif isinstance(cache, DynamicCache):
        rewind_dynamic_cache(cache, target_seq_len)


# Global acceleration flags
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class DraftSampler:
    """
    Lightweight draft model sampler running on Node 0.
    Generates K candidate tokens locally at high throughput (~80+ TPS).
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        spec_k: int = 4,
    ):
        self.model_path = model_path
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.spec_k = spec_k
        self.cache = DynamicCache()
        self._seq_len: int = 0

        logger.info("Loading draft model %s on %s (dtype=%s, K=%d)...", model_path, device, dtype, spec_k)
        import os
        target_cache = "/kaggle/working/hf_home" if os.path.exists("/kaggle") else None
        actual_path = model_path
        if not os.path.exists(actual_path):
            if "0.5B" in actual_path:
                actual_path = "Qwen/Qwen2.5-0.5B-Instruct"
            elif "1.5B" in actual_path:
                actual_path = "Qwen/Qwen2.5-1.5B-Instruct"
            elif "7B" in actual_path:
                actual_path = "Qwen/Qwen2.5-7B-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            actual_path,
            torch_dtype=dtype,
            cache_dir=target_cache,
        ).to(self.device)
        self.model.eval()

        self.transformer = getattr(self.model, "model", self.model)
        self.lm_head = getattr(self.model, "lm_head", None)

        # Pre-allocated single-token buffer and pre-computed position_ids
        self._cur_tensor = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._pos_ids = torch.arange(8192, dtype=torch.long, device=self.device).unsqueeze(0)

        # Optional compilation on raw decoder stack with cudagraphs backend
        try:
            self.transformer = torch.compile(
                self.transformer,
                backend="cudagraphs",
            )
            logger.info("DraftSampler: torch.compile(backend=cudagraphs) enabled on %s", self.device)
        except Exception as e:
            logger.warning("DraftSampler: torch.compile skipped (%s), running in direct eager mode", e)

    @property
    def seq_len(self) -> int:
        """Current absolute sequence length of the draft model's KV cache."""
        return self._seq_len

    def reset(self) -> None:
        """Reset draft model KV cache for a new session."""
        self.cache = DynamicCache()
        self._seq_len = 0

    @torch.inference_mode()
    def prefill(self, prompt_tokens: List[int]) -> None:
        """
        Prefill draft model with prompt tokens so candidate tokens are generated
        in the correct conversational context (prevents gibberish / repetition loops).
        """
        self.reset()
        if not prompt_tokens:
            return
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        pos = self._pos_ids[:, :len(prompt_tokens)]
        self.transformer(
            input_ids=input_ids,
            past_key_values=self.cache,
            position_ids=pos,
            use_cache=True,
            return_dict=False,
        )
        self._seq_len = len(prompt_tokens)

    def rewind(self, target_seq_len: int) -> None:
        """Rewind draft model KV cache after speculative rejection."""
        rewind_dynamic_cache(self.cache, target_seq_len)
        self._seq_len = target_seq_len

    @torch.inference_mode()
    def generate_drafts(
        self,
        current_token: int,
        k: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> List[int]:
        """
        Generate K candidate draft tokens locally.
        ponytail: GPU-resident argmax loop with zero per-token CPU-GPU synchronization.
        """
        k = k or self.spec_k
        if k <= 0:
            return []

        # Greedy path: 100% GPU resident, direct forward on compiled transformer
        if temperature <= 0 or temperature < 1e-8:
            self._cur_tensor.fill_(current_token)
            gpu_drafts: List[torch.Tensor] = []

            for _ in range(k):
                pos = self._pos_ids[:, self._seq_len : self._seq_len + 1]
                hidden_states = self.transformer(
                    input_ids=self._cur_tensor,
                    past_key_values=self.cache,
                    position_ids=pos,
                    use_cache=True,
                    return_dict=False,
                )[0]
                if self.lm_head is not None:
                    next_tok = self.lm_head(hidden_states[:, -1:, :]).argmax(dim=-1)
                else:
                    next_tok = hidden_states[:, -1:, :].argmax(dim=-1)
                gpu_drafts.append(next_tok)
                self._cur_tensor.copy_(next_tok)
                self._seq_len += 1

            if not gpu_drafts:
                return []
            return torch.cat(gpu_drafts, dim=-1).squeeze(0).tolist()

        # Sampling path (when temperature > 0)
        draft_tokens: List[int] = []
        next_tok = current_token

        for _ in range(k):
            input_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
            outputs = self.model(
                input_ids=input_tensor,
                past_key_values=self.cache,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            sampled_id = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            draft_tokens.append(sampled_id)
            next_tok = sampled_id
            self._seq_len += 1

        return draft_tokens


class AsyncDraftSampler:
    """
    Dedicated background worker thread for DraftSampler running on secondary GPU (e.g. cuda:1).
    Decouples draft generation from main inference thread so drafting on cuda:1 overlaps
    100% with Node 0 forward pass and TCP network verification round-trip.
    """

    def __init__(self, sampler: DraftSampler):
        self.sampler = sampler
        self._jobs: queue.Queue = queue.Queue(maxsize=4)
        self._results: queue.Queue = queue.Queue(maxsize=4)
        self._stopped = threading.Event()
        self._active_job_id = 0
        self._lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AsyncDraftSamplerWorker",
        )
        self._thread.start()

    def _worker_loop(self) -> None:
        if self.sampler.device.type == "cuda":
            try:
                torch.cuda.set_device(self.sampler.device)
            except Exception:
                pass

        while not self._stopped.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue

            if job is None:
                break

            job_id, current_token, k, temperature, top_k, top_p = job
            t0 = time.perf_counter()
            try:
                drafts = self.sampler.generate_drafts(
                    current_token=current_token,
                    k=k,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            except Exception as e:
                logger.error("AsyncDraftSampler error in worker: %s", e)
                drafts = []
            t1 = time.perf_counter()
            draft_time_ms = (t1 - t0) * 1000.0

            self._results.put((job_id, drafts, draft_time_ms))
            self._jobs.task_done()

    def submit(
        self,
        current_token: int,
        k: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> int:
        """Submit an asynchronous draft generation job to cuda:1."""
        with self._lock:
            self._active_job_id += 1
            jid = self._active_job_id
            # Clear old unprocessed results if any
            while not self._results.empty():
                try:
                    self._results.get_nowait()
                except queue.Empty:
                    break
            self._jobs.put((jid, current_token, k, temperature, top_k, top_p))
            return jid

    def get(self, job_id: int, timeout: Optional[float] = None) -> Tuple[List[int], float, float]:
        """
        Wait for draft result.
        Returns: (draft_tokens, draft_gen_ms, wait_at_recon_ms)
        """
        t_wait_0 = time.perf_counter()
        while True:
            res_jid, drafts, draft_time_ms = self._results.get(timeout=timeout)
            if res_jid == job_id:
                t_wait_1 = time.perf_counter()
                wait_at_recon_ms = (t_wait_1 - t_wait_0) * 1000.0
                return drafts, draft_time_ms, wait_at_recon_ms

    def prefill(self, prompt_tokens: List[int]) -> None:
        """Synchronously prefill draft model cache before starting decode."""
        self.sampler.prefill(prompt_tokens)

    def rewind(self, target_seq_len: int) -> None:
        """Rewind draft model cache to target sequence length."""
        self.sampler.rewind(target_seq_len)

    @property
    def seq_len(self) -> int:
        return self.sampler.seq_len

    @property
    def spec_k(self) -> int:
        return self.sampler.spec_k

    @spec_k.setter
    def spec_k(self, val: int) -> None:
        self.sampler.spec_k = val

    def stop(self) -> None:
        self._stopped.set()
        try:
            self._jobs.put_nowait(None)
        except Exception:
            pass
