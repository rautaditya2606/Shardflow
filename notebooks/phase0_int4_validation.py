"""
Phase 0 — int4 Partial Layer Loading Validation Script

Run this script to test and validate 4-bit (NF4) partial layer loading on GPU.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shardflow.node.int4_loader import load_int4_layer_slice


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Phase 0] Validating 4-bit partial layer loading for '{model_id}' on device: {device}...")

    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. bitsandbytes 4-bit quantization requires a GPU.")
        print("To validate on CPU / local env, standard fp16/bf16 loading in `layer_loader.py` is used.")
        return

    try:
        slice_info = load_int4_layer_slice(
            model_path=model_id,
            layer_start=0,
            layer_end=11,
            include_norm=False,
            include_lm_head=False,
            load_in_4bit=True,
            device=device,
        )
        print(f"✅ Successfully loaded layers 0-11 in 4-bit (NF4)!")
        print(f"Layers extracted: {len(slice_info['layers'])}")
        allocated_vram = torch.cuda.memory_allocated() / 1e6
        print(f"VRAM usage: {allocated_vram:.2f} MB")
    except Exception as e:
        print(f"❌ int4 loading test failed: {e}")


if __name__ == "__main__":
    main()
