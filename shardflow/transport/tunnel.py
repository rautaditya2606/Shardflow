"""
Cloudflare Tunnel wrapper for Colab nodes.

Downloads cloudflared if needed and runs:
    cloudflared tunnel --url tcp://localhost:{port}
Extracts the assigned public hostname / TCP address for registering with the Topology Registry.
"""

import subprocess
import re
import time
import os
import shutil
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def install_cloudflared() -> str:
    """Ensure cloudflared binary is installed and executable."""
    bin_path = shutil.which("cloudflared")
    if bin_path:
        return bin_path

    logger.info("Downloading cloudflared binary...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    target = "/tmp/cloudflared"
    subprocess.run(["curl", "-L", "-o", target, url], check=True)
    subprocess.run(["chmod", "+x", target], check=True)
    return target


def start_cloudflare_tcp_tunnel(local_port: int) -> Tuple[subprocess.Popen, str, int]:
    """
    Start cloudflared tunnel for TCP on local_port.

    Returns:
        (process, public_host, public_port)
    """
    bin_path = install_cloudflared()
    cmd = [bin_path, "tunnel", "--url", f"tcp://localhost:{local_port}"]
    
    logger.info("Starting Cloudflare TCP tunnel for localhost:%d ...", local_port)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Parse stdout/stderr log output for tunnel address (.trycloudflare.com or tcp://...)
    public_url = None
    start_time = time.time()
    
    while time.time() - start_time < 30:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.5)
            continue
        
        # Look for trycloudflare.com URL pattern
        match = re.search(r"https://([a-zA-Z0-9-]+\.trycloudflare\.com)", line)
        if match:
            host = match.group(1)
            # Cloudflare TCP tunnels expose standard TLS/TCP ports or hostname
            public_url = host
            logger.info("Cloudflare Tunnel established: %s", public_url)
            break
            
    if not public_url:
        proc.terminate()
        raise RuntimeError("Failed to obtain Cloudflare tunnel URL within timeout")

    # Cloudflare tunnel hostname defaults to port 7844 or 443 TCP depending on routing
    return proc, public_url, 443
