"""
Tailscale P2P Mesh VPN helper for Google Colab and Kaggle.
Enables low-latency (<5ms) direct WireGuard communication between GPU nodes.
"""

import json
import logging
import os
import shutil
import subprocess
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_tailscale_status() -> Optional[dict]:
    """Get JSON status from tailscale CLI if running."""
    tailscale_bin = shutil.which("tailscale") or "/tmp/tailscale_bin/tailscale"
    if not os.path.exists(tailscale_bin) and not shutil.which("tailscale"):
        return None
    for sock in ["/var/run/tailscale/tailscaled.sock", "/tmp/tailscaled.sock"]:
        cmd = [tailscale_bin]
        if os.path.exists(sock):
            cmd.append(f"--socket={sock}")
        cmd.extend(["status", "--json"])
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
            if res.returncode == 0 and res.stdout.strip():
                return json.loads(res.stdout)
        except Exception as e:
            logger.debug("Tailscale status inspect error: %s", e)
    return None


def get_tailscale_ip() -> Optional[str]:
    """Return the assigned 100.x.y.z IPv4 address."""
    status = get_tailscale_status()
    if status and "Self" in status and "TailscaleIPs" in status["Self"]:
        ips = status["Self"]["TailscaleIPs"]
        for ip in ips:
            if "." in ip:  # IPv4
                return ip
    return None


def get_tailscale_hostname() -> Optional[str]:
    """Return the MagicDNS FQDN hostname (stable across restarts)."""
    status = get_tailscale_status()
    if status and "Self" in status:
        # MagicDNS DNSName e.g. "colab-node-1.tail1234.ts.net."
        dns_name = status["Self"].get("DNSName", "").rstrip(".")
        if dns_name:
            return dns_name
        return status["Self"].get("HostName")
    return None


def setup_tailscale_colab(authkey: str, hostname: str = "shardflow-colab") -> Tuple[str, str]:
    """
    Configure and authenticate Tailscale in Google Colab environment.
    Aggressively attempts kernel TUN mode (direct WireGuard UDP) with userspace SOCKS5 fallback.
    Returns: (tailscale_ip, magicdns_hostname)
    """
    logger.info("Setting up Tailscale P2P Mesh for Colab (%s)...", hostname)

    # 1. Install Tailscale if not present
    if not shutil.which("tailscale"):
        logger.info("Installing Tailscale...")
        subprocess.run(
            "curl -fsSL https://tailscale.com/install.sh | sh",
            shell=True, check=True, timeout=60.0,
        )

    tailscale_bin = shutil.which("tailscale") or "/usr/bin/tailscale"
    tailscaled_bin = shutil.which("tailscaled") or "/usr/sbin/tailscaled"
    socket_path = "/var/run/tailscale/tailscaled.sock"
    os.makedirs("/var/run/tailscale", exist_ok=True)
    os.makedirs("/var/lib/tailscale", exist_ok=True)

    # 2. Start tailscaled if not already running
    already_running = subprocess.run(["pgrep", "-x", "tailscaled"], capture_output=True).returncode == 0
    if not already_running:
        started_kernel = False

        # ponytail: skip systemctl — Colab doesn't register tailscaled as a proper systemd service.
        # Force-create /dev/net/tun (Colab VMs run as root and support this) then launch directly.
        subprocess.run(
            "mkdir -p /dev/net && mknod /dev/net/tun c 10 200 2>/dev/null; chmod 600 /dev/net/tun 2>/dev/null || true",
            shell=True, timeout=5.0,
        )

        log_f = open("/tmp/tailscaled.log", "a")
        proc = subprocess.Popen(
            [tailscaled_bin,
             "--state=/var/lib/tailscale/tailscaled.state",
             f"--socket={socket_path}"],
            stdout=log_f, stderr=subprocess.STDOUT,
        )

        # Wait up to 10s for socket to appear and process to stay alive
        for _ in range(20):
            time.sleep(0.5)
            if os.path.exists(socket_path) and proc.poll() is None:
                started_kernel = True
                break

        if started_kernel:
            logger.info("tailscaled started in kernel TUN mode (direct WireGuard UDP, ~5ms RTT)")
        else:
            # Log why kernel TUN failed
            try:
                with open("/tmp/tailscaled.log") as f:
                    last_lines = f.read()[-800:]
                logger.warning("Kernel TUN failed. Last tailscaled log:\n%s", last_lines)
            except Exception:
                pass

            # Fallback: userspace SOCKS5
            logger.warning(
                "Colab kernel TUN unavailable — falling back to SOCKS5 userspace mode (~200ms RTT). "
                "To get ~5ms RTT, run Colab as a Docker container with --cap-add=NET_ADMIN."
            )
            log_f2 = open("/tmp/tailscaled.log", "a")
            proc = subprocess.Popen(
                [tailscaled_bin,
                 "--tun=userspace-networking",
                 "--socks5-server=localhost:1055",
                 "--state=/var/lib/tailscale/tailscaled.state",
                 f"--socket={socket_path}"],
                stdout=log_f2, stderr=subprocess.STDOUT,
            )
            for _ in range(20):
                if os.path.exists(socket_path):
                    break
                time.sleep(0.5)
            logger.info("tailscaled started in userspace SOCKS5 mode")

    # 3. Authenticate
    logger.info("Authenticating with Tailscale network...")
    subprocess.run(
        [tailscale_bin,
         f"--socket={socket_path}",
         "up",
         f"--authkey={authkey}",
         f"--hostname={hostname}",
         "--accept-routes",
         "--accept-dns=false"],
        check=True, timeout=30.0,
    )

    # 4. Read assigned IP
    for _ in range(10):
        ip = get_tailscale_ip()
        hname = get_tailscale_hostname() or ip
        if ip:
            logger.info("Tailscale connected! IP: %s | Hostname: %s", ip, hname)
            return ip, hname
        time.sleep(1.0)

    raise RuntimeError("Tailscale connected but failed to retrieve assigned IP")


def setup_tailscale_kaggle(authkey: str, hostname: str = "shardflow-kaggle") -> Tuple[str, str]:
    """
    Configure and authenticate Tailscale in Kaggle unprivileged container (userspace networking).
    Returns: (tailscale_ip, magicdns_hostname)
    """
    logger.info("Setting up Tailscale userspace mode for Kaggle (%s)...", hostname)
    bin_dir = "/tmp/tailscale_bin"
    os.makedirs(bin_dir, exist_ok=True)
    tailscaled_path = os.path.join(bin_dir, "tailscaled")
    tailscale_path = os.path.join(bin_dir, "tailscale")

    # 1. Download and extract static binaries cleanly
    if not os.path.exists(tailscale_path) or not os.path.exists(tailscaled_path):
        logger.info("Downloading static Tailscale userspace binaries...")
        tar_url = "https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz"
        subprocess.run(
            f"curl -sL {tar_url} | tar -xz -C /tmp && cp -f /tmp/tailscale_*_amd64/tailscale* {bin_dir}/ 2>/dev/null || cp -f /tmp/tailscale_*_amd64/tailscale /tmp/tailscale_*_amd64/tailscaled {bin_dir}/",
            shell=True,
            check=True,
            timeout=60.0,
        )
        os.chmod(tailscale_path, 0o755)
        os.chmod(tailscaled_path, 0o755)

    # 2. Start tailscaled in userspace mode if not running
    socket_path = "/tmp/tailscaled.sock"
    p_check = subprocess.run(["pgrep", "tailscaled"], capture_output=True)
    if p_check.returncode != 0:
        log_f = open("/tmp/tailscaled.log", "a")
        subprocess.Popen(
            [
                tailscaled_path,
                "--tun=userspace-networking",
                "--socks5-server=localhost:1055",
                "--state=/tmp/tailscaled.state",
                f"--socket={socket_path}",
            ],
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        for _ in range(20):
            if os.path.exists(socket_path):
                break
            time.sleep(0.5)

    # 3. Authenticate if not already connected
    curr_ip = get_tailscale_ip()
    if not curr_ip:
        logger.info("Authenticating Kaggle node with Tailscale...")
        subprocess.run(
            [
                tailscale_path,
                f"--socket={socket_path}",
                "up",
                f"--authkey={authkey}",
                f"--hostname={hostname}",
            ],
            check=True,
            timeout=30.0,
        )

    for _ in range(15):
        ip = get_tailscale_ip()
        hname = get_tailscale_hostname() or ip
        if ip:
            logger.info("Kaggle Tailscale connected! IP: %s | Hostname: %s", ip, hname)
            return ip, hname
        time.sleep(1.0)

    raise RuntimeError("Tailscale userspace mode failed to acquire IP on Kaggle")
