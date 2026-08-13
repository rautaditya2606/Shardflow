"""
Benchmark Tokens Per Second (TPS), TTFT, and Latency for ShardFlow.
Surfaces transport path (wireguard / socks5 / loopback) so you always
know exactly what you're measuring.
"""

import json
import time
import requests
import statistics


def get_transport_path(base_url: str = "http://127.0.0.1:8000") -> str:
    """Query the gateway for active node0 transport path."""
    try:
        r = requests.get(f"{base_url}/debug/transport", timeout=3.0)
        if r.ok:
            return r.json().get("node0_transport_path", "unknown")
    except Exception:
        pass
    return "unknown"


def benchmark(url="http://127.0.0.1:8000/v1/chat/completions", max_tokens=50, num_runs=3):
    base_url = url.rsplit("/v1", 1)[0]
    transport = get_transport_path(base_url)

    transport_label = {
        "wireguard": "✅ WireGuard (direct UDP — ~5ms RTT)",
        "socks5":    "⚠️  SOCKS5 (userspace proxy — ~200ms RTT) — RESULTS ARE DEGRADED",
        "loopback":  "🔁 Loopback (same-host)",
        "unknown":   "❓ Unknown (gateway may not expose /debug/transport yet)",
    }.get(transport, f"❓ {transport}")

    print("=" * 65)
    print("⚡ SHARDFLOW DISTRIBUTED INFERENCE BENCHMARK")
    print(f"Target:     {url}")
    print(f"Transport:  {transport_label}")
    print(f"Max Tokens: {max_tokens} | Runs: {num_runs}")
    print("=" * 65)

    if transport == "socks5":
        print()
        print("  ⚠️  WARNING: Running over SOCKS5 (Tailscale userspace mode).")
        print("  ⚠️  TPS will be ~1-2 tok/s due to TCP-over-TCP latency, NOT GPU speed.")
        print("  ⚠️  Fix: ensure kernel TUN mode starts (check /tmp/tailscaled.log on Colab).")
        print()

    prompt = "Explain why pipeline parallelism is important for large language models."
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    tps_results = []
    ttft_results = []
    total_time_results = []

    for run_idx in range(1, num_runs + 1):
        print(f"\n--- Benchmark Run {run_idx}/{num_runs} ---")
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        text_chunks = []

        response = requests.post(url, json=payload, stream=True, timeout=120.0)
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            continue

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                            text_chunks.append(content)
                            print(content, end="", flush=True)
                except Exception:
                    pass

        end_time = time.perf_counter()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time
        decode_time = end_time - first_token_time if first_token_time else total_time
        tps = (token_count - 1) / decode_time if decode_time > 0 and token_count > 1 else (token_count / decode_time if decode_time > 0 else 0)

        print()
        print(f"  ➜ Tokens: {token_count} | TTFT: {ttft:.3f}s | Decode Time: {decode_time:.3f}s | TPS: {tps:.2f} tok/s")

        tps_results.append(tps)
        ttft_results.append(ttft)
        total_time_results.append(total_time)

    # Refresh transport path after warm requests
    final_transport = get_transport_path(base_url)
    if final_transport != "unknown":
        transport = final_transport
        transport_label = {
            "wireguard": "✅ WireGuard (direct UDP — ~5ms RTT)",
            "socks5":    "⚠️  SOCKS5 (userspace proxy — ~200ms RTT) — RESULTS ARE DEGRADED",
            "loopback":  "🔁 Loopback (same-host)",
            "unknown":   "❓ Unknown (gateway may not expose /debug/transport yet)",
        }.get(transport, f"❓ {transport}")

    print("\n" + "=" * 65)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  Transport:          {transport_label}")
    if tps_results:
        print(f"  Avg Decode TPS:     {statistics.mean(tps_results):.2f} tok/s (Max: {max(tps_results):.2f} tok/s)")
        print(f"  Avg TTFT (Prefill): {statistics.mean(ttft_results):.3f}s (Min: {min(ttft_results):.3f}s)")
        print(f"  Avg Total Time:     {statistics.mean(total_time_results):.3f}s")
    if transport == "socks5":
        print()
        print("  ⚠️  These results reflect SOCKS5 latency, not GPU throughput.")
        print("  ⚠️  Re-run after fixing kernel TUN or running on Kaggle 2x T4 to get real numbers.")
    print("=" * 65)


if __name__ == "__main__":
    benchmark()
