# Optimizing & Bypassing WAN Proxy Latency in ShardFlow

This document provides a comprehensive engineering guide on reducing, bypassing, and amortizing the **WAN network round-trip time (RTT)** caused by `bore.pub` and public reverse tunnels in ShardFlow.

---

## Executive Summary: The WAN Latency Problem

In distributed LLM pipeline parallelism, generating each token requires sequential communication between worker nodes:
$$\text{Token Latency} = T_{\text{Node 0 GPU}} + T_{\text{Forward Hop (Node 0} \to \text{Node 1)}} + T_{\text{Node 1 GPU}} + T_{\text{Return Hop (Node 1} \to \text{Node 0)}}$$

When using the default public `bore.pub` proxy server:
- `bore.pub` is an unmanaged, shared proxy located in a fixed geographic region (typically US-East or Europe).
- Even if two Google Colab VMs are hosted in the **same Google Cloud region** (e.g., `us-central1`), packets leave the Google Cloud backbone, travel over the public internet to `bore.pub`, and travel back into Google Cloud.
- This **hairpin routing** adds **120 ms – 220 ms of pure network latency per token**, capping throughput at $\le 4.5\text{ TPS}$ regardless of GPU speed.

```
CURRENT (HAIRPIN ROUTING VIA PUBLIC BORE.PUB):
┌─────────────────────────┐                                 ┌─────────────────────────┐
│ Colab Node 0            │                                 │ Colab Node 1            │
│ (Google Cloud us-west1) │                                 │ (Google Cloud us-west1) │
└───────────┬─────────────┘                                 └─────────────▲───────────┘
            │                                                             │
            │  100ms WAN Hop                               100ms WAN Hop  │
            ▼                                                             │
     ┌────────────────────────────────────────────────────────────────────┴─────┐
     │                       Shared bore.pub Server                            │
     │                      (Located in US-East / EU)                          │
     └──────────────────────────────────────────────────────────────────────────┘
Total Round-Trip Time per token: ~200 ms
```

---

## 5 Strategies to Eliminate or Amortize WAN Latency

| Strategy | Complexity | Estimated RTT | TPS Potential | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **1. Tailscale P2P Mesh VPN** | Low (5 lines in Colab) | **1 ms – 20 ms** | **25 – 45+ TPS** | Colab $\leftrightarrow$ Colab / Runpod $\leftrightarrow$ Colab |
| **2. Self-Hosted Private `bore` Server** | Low ($5/mo Cloud VM) | **15 ms – 40 ms** | **15 – 25 TPS** | Centralized proxy without public queueing |
| **3. Speculative Multi-Token Verification** | Software (Codebase) | **Amortized $\div 4$** | **20 – 35 TPS** | Works even over slow high-latency WAN |
| **4. Ngrok with Regional Binding** | Low (Auth Token) | **30 ms – 60 ms** | **12 – 20 TPS** | Quick drop-in replacement for bore |
| **5. OS & TCP Socket Kernel Tuning** | Zero (Script flags) | **-15% to -25%** | **+2 – 4 TPS** | Combine with any transport |

---

## Strategy 1: Direct P2P Mesh Networking via Tailscale (Recommended)

### Why it works:
Tailscale uses **WireGuard** and NAT traversal (STUN/ICE hole-punching). After the initial handshake, Node 0 and Node 1 connect **directly peer-to-peer over UDP**.

If both Google Colab notebooks are allocated in Google Cloud:
- Traffic **never leaves the Google Cloud internal network**.
- RTT drops from **~200 ms** to **< 5 ms** (intra-cloud) or **~25 ms** (cross-cloud).

```
DIRECT P2P (TAILSCALE MESH):
┌─────────────────────────┐                                 ┌─────────────────────────┐
│ Colab Node 0            │ ═══════════════════════════════>│ Colab Node 1            │
│ (Google Cloud us-west1) │     Direct P2P UDP WireGuard    │ (Google Cloud us-west1) │
│ IP: 100.x.y.1           │          Latency: 2 - 8 ms      │ IP: 100.x.y.2           │
└─────────────────────────┘ <═══════════════════════════════└─────────────────────────┘
```

### 1. Tailscale ACL Configuration (Mandatory)
By default, Tailscale enforces a default-deny policy between tagged devices. You **must** add the following rule in your [Tailscale Access Controls](https://login.tailscale.com/admin/acls/file):

```json
{
  "tagOwners": {
    "tag:shardflow-node": ["autogroup:admin"]
  },
  "acls": [
    {
      "action": "accept",
      "src": ["tag:shardflow-node"],
      "dst": ["tag:shardflow-node:*"]
    }
  ]
}
```

### 2. Ephemeral Auth Key
Generate a **Reusable + Ephemeral** auth key with `tag:shardflow-node` assigned in the [Tailscale Keys Console](https://login.tailscale.com/admin/settings/keys). When notebooks disconnect, they are automatically purged from your network.

### 3. Google Colab vs. Kaggle Installation

#### In Google Colab (Automated 1-Line CLI or Manual Setup):

**Option A: Automated via Runner Flag (Recommended):**
```python
!python scripts/colab_runner.py \
    --registry-url https://shardflow.onrender.com \
    --model Qwen/Qwen2.5-7B-Instruct \
    --node-id colab-node-1 \
    --tailscale-authkey "tskey-auth-kXXXXX-XXXXXXXX"
```

**Option B: Manual Tailscale Setup & MagicDNS Hostnames:**
> **Tip on MagicDNS:** Tailscale MagicDNS provides stable hostnames (`colab-node-1.your-tailnet.ts.net`) that persist across Colab session restarts, so peer routing targets remain constant even when new ephemeral IPs are assigned.

```python
# 1. Install Tailscale
!curl -fsSL https://tailscale.com/install.sh | sh

# 2. Create tun device & start daemon
!mkdir -p /dev/net && mknod /dev/net/tun c 10 200 && chmod 600 /dev/net/tun
!tailscaled --tun=userspace-networking &

# 3. Authenticate
!tailscale up --authkey="tskey-auth-kXXXXX-XXXXXXXX" --hostname="colab-node-1" --accept-routes

# 4. Get Tailscale IP and MagicDNS Hostname
import json, subprocess
status = json.loads(subprocess.check_output(["tailscale", "status", "--json"]).decode())
tailscale_ip = status["Self"]["TailscaleIPs"][0]
magicdns_name = status["Self"].get("DNSName", "").rstrip(".")
print(f"Node Tailscale IP: {tailscale_ip} | Hostname: {magicdns_name}")
```

#### In Kaggle Notebooks (Unprivileged Container - No `mknod`):
On Kaggle, running `mknod` will fail with `Operation not permitted`. You **must** use userspace networking mode:
```python
# 1. Download static tailscaled binaries
!curl -sL https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz | tar -xz -C /tmp

# 2. Start daemon in userspace mode
import subprocess
subprocess.Popen(["/tmp/tailscale_1.80.0_amd64/tailscaled", "--tun=userspace-networking", "--socks5-server=localhost:1055"])

# 3. Authenticate with ephemeral key
!/tmp/tailscale_1.80.0_amd64/tailscale up --authkey="tskey-auth-kXXXXX-XXXXXXXX" --hostname="kaggle-node-1"

# 4. Extract assigned 100.x.y.z IP
status = json.loads(subprocess.check_output(["/tmp/tailscale_1.80.0_amd64/tailscale", "status", "--json"]).decode())
tailscale_ip = status["Self"]["TailscaleIPs"][0]
print(f"Kaggle Tailscale IP: {tailscale_ip}")
```

### 4. Automatic Same-Host Loopback Routing (0.1 ms Latency)
In multi-GPU Kaggle notebooks, ShardFlow's `NodeClient` automatically inspects the target address. If the target host matches any local interface or the notebook's own Tailscale IP, it bypasses the network stack and connects directly via `127.0.0.1` (< 0.2 ms latency), ensuring GPU 0 $\leftrightarrow$ GPU 1 communication never routes through external proxies.


---

## Strategy 2: Self-Hosted Private `bore` Server in the Same Cloud Region

If direct P2P mesh VPN cannot be used, run your own private `bore` relay server on a lightweight cloud VM located in the same geographic region as your worker nodes (e.g. AWS `us-east-1`, GCP `us-central1`, or Hetzner).

### Benefits over public `bore.pub`:
- **Zero noisy neighbors**: No socket buffer contention from thousands of external users.
- **Lower ping**: Placing the server geographically close to your Colab region cuts RTT in half.
- **Dedicated bandwidth**: 1 Gbps+ uncontended uplink.

### Deployment (1 Minute on any Linux VM):

1. **On your Cloud VM (e.g., Ubuntu 22.04 on GCP / AWS / DigitalOcean / Hetzner):**
```bash
# Download bore binary
curl -sL https://github.com/ekzhang/bore/releases/download/v0.5.0/bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz | tar -xz -C /usr/local/bin

# Start bore server with a secret key and allowed port range
bore server --secret "your-cluster-secret" --min-port 30000
```

2. **In ShardFlow Colab Runner:**
Update `start_bore_tunnel` in [shardflow/transport/tunnel.py](file:///home/adityaraut/Documents/Shardflow/shardflow/transport/tunnel.py) or pass `--tunnel-server`:

```python
# Pass private server address to colab runner
python scripts/colab_runner.py \
  --registry-url https://shardflow.onrender.com \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tunnel-server "your-vm-ip.com" \
  --tunnel-secret "your-cluster-secret"
```

---

## Strategy 3: Speculative Decoding (Algorithmic Latency Amortization)

The laws of physics dictate that intercontinental network packets cannot travel faster than the speed of light in fiber (~5 ms per 1,000 km). If Node 0 is in California and Node 1 is in Tokyo, the theoretical minimum physical RTT is ~100 ms.

**Speculative Decoding beats the network barrier mathematically**:
Instead of transmitting activations across the network 1 token at a time, we transmit **$K$ candidate tokens simultaneously in a single network round-trip**.

```
STANDARD DECODING (1 Token per Network Roundtrip):
Step 1: [Embed Token 1] ──(RTT 100ms)──> [Node 1 Head] ──(RTT 100ms)──> Next Token (Total: 200ms)
Step 2: [Embed Token 2] ──(RTT 100ms)──> [Node 1 Head] ──(RTT 100ms)──> Next Token (Total: 200ms)
Result for 4 tokens: 4 × 200ms = 800 ms (5 TPS)

SPECULATIVE PIPELINE (K=4 Tokens in 1 Network Roundtrip):
Step 1: Node 0 runs tiny draft model (e.g. Qwen-0.5B at 80 TPS) -> generates 4 draft tokens locally in 15ms.
Step 2: Node 0 computes hidden states for all 4 tokens in batch [1, 4, D].
Step 3: Node 0 sends [1, 4, D] to Node 1 in ONE network trip (100ms).
Step 4: Node 1 verifies all 4 tokens in parallel via replay_verify() in 20ms.
Step 5: Node 1 returns accepted tokens in ONE return trip (100ms).
Result for 4 tokens: 15ms + 100ms + 20ms + 100ms = 235 ms (17.0 TPS -> 3.4x faster!)
```

### Architecture in ShardFlow:
- [shardflow/node/cuda_graph.py](file:///home/adityaraut/Documents/Shardflow/shardflow/node/cuda_graph.py) already contains the static buffers and `replay_verify()` for `spec_k=4`.
- Integrating a small local draft model on Node 0 amortizes the WAN RTT by **$3\times - 5\times$**.

---

## Strategy 4: Ngrok with Region-Specific TCP Endpoints

Ngrok provides global Anycast and regional edge routing with strict TCP throughput guarantees.

### Setup in Colab:
```python
!pip install pyngrok
from pyngrok import conf, ngrok

conf.get_default().auth_token = "YOUR_NGROK_AUTHTOKEN"
conf.get_default().region = "us" # or "eu", "ap", "in", "jp"

tunnel = ngrok.connect(9500, "tcp")
public_url = tunnel.public_url # e.g. "tcp://4.tcp.ngrok.io:12345"
host, port = public_url.replace("tcp://", "").split(":")
print(f"Ngrok TCP endpoint: {host}:{port}")
```

---

## Strategy 5: TCP Socket & Linux Kernel Level Tuning

Apply these socket options in Python and Linux sysctl to minimize latency jitter:

### 1. Enable TCP BBR Congestion Control on Colab
BBR minimizes packet queuing delay over lossy WAN links:
```bash
sudo sysctl -w net.core.default_qdisc=fq
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0
sudo sysctl -w net.ipv4.tcp_notsent_lowat=16384
```

### 2. Socket Flags in [shardflow/transport/connection.py](file:///home/adityaraut/Documents/Shardflow/shardflow/transport/connection.py)
Ensure the following are set on all reader/writer sockets:
- `TCP_NODELAY = 1`: Disables Nagle's algorithm (sends small tensor packets immediately without buffering).
- `TCP_QUICKACK = 1`: Disables delayed ACKs (forces the OS to acknowledge received packets immediately).
- `SO_SNDBUF` / `SO_RCVBUF` set to `262144` (256 KB) to prevent socket buffer stalls.

---

## Benchmark Progression Log

### Run 1: v1 Control-Plane (Gateway-Driven WAN Loop)
- **Architecture**: v1 Gateway-driven token loop (Laptop in India $\leftrightarrow$ Colab US nodes)
- **Model**: `Qwen/Qwen2.5-7B-Instruct` (FP16, 2 nodes)
- **Avg Decode TPS**: **1.89 tok/s**
- **Avg TTFT**: **6.21s**
- **Avg Total Time**: **32.11s** (50 tokens)
- **Bottleneck**: Every token required a full cross-continent WAN round-trip (~500ms RTT) between local laptop and cloud nodes.

### Run 2: v2 Peer-to-Peer Data-Plane (Direct P2P Token Loop) ⚡
- **Architecture**: v2 Data-Plane (`START_SESSION` — token loop runs directly between GPU nodes; streaming tokens back asynchronously)
- **Model**: `Qwen/Qwen2.5-7B-Instruct` (FP16, 2 nodes)
- **Avg Decode TPS**: **4.59 tok/s** (Max: **4.61 tok/s**) — **2.44x Speedup** over v1!
- **Avg TTFT**: **7.40s** (Min: **3.859s**)
- **Avg Total Time**: **18.08s** (50 tokens)
- **Raw Run Output**:
```text
--- Benchmark Run 3/3 ---
Pipeline processing, often discussed in the V-riberglass composite form- -working through processes, operations, and network contexts-fashion or, more specifically, in the-depth, is crucial for the seamless and efficient processing, especially in the- ldings;
  ➜ Tokens: 50 | TTFT: 3.859s | Decode Time: 10.625s | TPS: 4.61 tok/s
=================================================================
📊 BENCHMARK SUMMARY
=================================================================
  Transport:          v2 P2P Data-Plane (Gateway Local)
  Avg Decode TPS:     4.59 tok/s (Max: 4.61 tok/s)
  Avg TTFT (Prefill): 7.404s (Min: 3.859s)
  Avg Total Time:     18.082s
=================================================================
```
- **Key Takeaway**: Eliminating per-token gateway round-trips produced a 2.44x throughput jump. The remaining ceiling is inter-node transport latency (SOCKS5 proxy vs direct kernel WireGuard / local PCIe loopback).

---

## Action Plan & Benchmark Summary

| Action | Latency Reduction | Implementation Effort |
| :--- | :--- | :--- |
| **Switch to Tailscale P2P / Direct Kernel TUN** | **~200 ms $\to$ 10 ms** | 10 minutes |
| **Disable bitsandbytes (Run pure FP16)** | **~180 ms $\to$ 20 ms** | 1 minute (remove `--load-in-4bit`) |
| **Enable Speculative Verification ($K=4$)** | **Amortize RTT by $4\times$** | Architectural upgrade |
| **Combined Target Performance** | **$\mathbf{500\text{ ms/token} \to 30\text{ ms/token}}$** | **$\mathbf{\approx 30 - 35\text{ TPS}}$** |

