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
    try:
        res = subprocess.run(
            [tailscale_bin, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if res.returncode == 0:
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
    Uses kernel TUN mode (via systemctl) so 100.x.x.x addresses are natively
    routable by the Linux kernel over WireGuard UDP — no SOCKS5 proxy.
    Returns: (tailscale_ip, magicdns_hostname)
    """
    logger.info("Setting up Tailscale P2P Mesh for Colab (%s)...", hostname)

    # 1. Install Tailscale if not present (registers tailscaled.service via systemd)
    if not shutil.which("tailscale"):
        logger.info("Installing Tailscale...")
        subprocess.run(
            "curl -fsSL https://tailscale.com/install.sh | sh",
            shell=True,
            check=True,
            timeout=60.0,
        )

    tailscale_bin = shutil.which("tailscale") or "/usr/bin/tailscale"

    # 2. Start tailscaled via systemd (kernel TUN mode — creates real tailscale0 interface)
    #    Colab VMs have full systemd; the install.sh registers tailscaled.service.
    #    This makes 100.x.x.x routable directly via kernel routing table.
    p_check = subprocess.run(["pgrep", "tailscaled"], capture_output=True)
    if p_check.returncode != 0:
        started = False
        # Try systemctl first (preferred — full kernel TUN, direct WireGuard UDP)
        if shutil.which("systemctl"):
            r = subprocess.run(["systemctl", "start", "tailscaled"], capture_output=True)
            if r.returncode == 0:
                time.sleep(2.0)
                started = subprocess.run(["pgrep", "tailscaled"], capture_output=True).returncode == 0
                if started:
                    logger.info("tailscaled started via systemctl (kernel TUN mode)")

        # Fallback: direct Popen with kernel TUN (no userspace-networking)
        if not started:
            tailscaled_bin = shutil.which("tailscaled") or "/usr/sbin/tailscaled"
            os.makedirs("/var/run/tailscale", exist_ok=True)
            os.makedirs("/var/lib/tailscale", exist_ok=True)
            os.makedirs("/dev/net", exist_ok=True)
            if not os.path.exists("/dev/net/tun"):
                subprocess.run("mknod /dev/net/tun c 10 200 && chmod 600 /dev/net/tun", shell=True, timeout=5.0)
            log_f = open("/tmp/tailscaled.log", "a")
            subprocess.Popen(
                [tailscaled_bin,
                 "--state=/var/lib/tailscale/tailscaled.state",
                 "--socket=/var/run/tailscale/tailscaled.sock"],
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            # Wait up to 10s for socket
            for _ in range(20):
                if os.path.exists("/var/run/tailscale/tailscaled.sock"):
                    break
                time.sleep(0.5)
            logger.info("tailscaled started via Popen (kernel TUN mode)")

    # 3. Authenticate
    logger.info("Authenticating with Tailscale network...")
    subprocess.run(
        [tailscale_bin, "up",
         f"--authkey={authkey}",
         f"--hostname={hostname}",
         "--accept-routes"],
        check=True,
        timeout=30.0,
    )

    # 4. Read assigned IP
    for _ in range(10):
        ip = get_tailscale_ip()
        hname = get_tailscale_hostname() or ip
        if ip:
            logger.info("Tailscale connected! IP: %s | Hostname: %s (kernel TUN, direct WireGuard)", ip, hname)
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

    if not os.path.exists(tailscale_path):
        logger.info("Downloading static Tailscale userspace binaries...")
        tar_url = "https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz"
        subprocess.run(
            f"curl -sL {tar_url} | tar -xz -C /tmp && cp /tmp/tailscale_*_amd64/* {bin_dir}/",
            shell=True,
            check=True,
            timeout=60.0,
        )

    # Start tailscaled in userspace mode
    p_check = subprocess.run(["pgrep", "tailscaled"], capture_output=True)
    if p_check.returncode != 0:
        subprocess.Popen(
            [tailscaled_path, "--tun=userspace-networking", "--socks5-server=localhost:1055"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.0)

    # Authenticate
    logger.info("Authenticating Kaggle node with Tailscale...")
    subprocess.run(
        [
            tailscale_path, "up",
            f"--authkey={authkey}",
            f"--hostname={hostname}",
        ],
        check=True,
        timeout=30.0,
    )

    for _ in range(10):
        status = get_tailscale_status()
        if status and "Self" in status and "TailscaleIPs" in status["Self"]:
            ip = status["Self"]["TailscaleIPs"][0]
            hname = status["Self"].get("DNSName", "").rstrip(".") or ip
            logger.info("Kaggle Tailscale connected! IP: %s | Hostname: %s", ip, hname)
            return ip, hname
        time.sleep(1.0)

    raise RuntimeError("Tailscale userspace mode failed to acquire IP on Kaggle")
