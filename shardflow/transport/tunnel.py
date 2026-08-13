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
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def install_cloudflared() -> str:
    """Ensure cloudflared binary is installed and executable (race-condition safe)."""
    bin_path = shutil.which("cloudflared")
    if bin_path:
        return bin_path

    target = "/tmp/cloudflared"
    if os.path.exists(target) and os.access(target, os.X_OK) and os.path.getsize(target) > 0:
        return target

    import fcntl
    lock_path = "/tmp/cloudflared_install.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(target) and os.access(target, os.X_OK) and os.path.getsize(target) > 0:
                return target

            logger.info("Downloading cloudflared binary...")
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            tmp_target = f"/tmp/cloudflared_{os.getpid()}"
            subprocess.run(["curl", "-sL", "-o", tmp_target, url], check=True)
            subprocess.run(["chmod", "+x", tmp_target], check=True)
            os.replace(tmp_target, target)
            return target
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


import threading


def _drain_stdout(proc: subprocess.Popen, name: str) -> None:
    """Continuously drain subprocess stdout in background to prevent pipe deadlock."""
    try:
        if proc.stdout is None:
            return
        for line in iter(proc.stdout.readline, ""):
            if line:
                logger.debug("[%s] %s", name, line.rstrip())
    except Exception as e:
        logger.debug("[%s] stdout drain exception: %s", name, e)


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

    # Start background drain thread to prevent pipe buffer from filling and freezing cloudflared
    drain_t = threading.Thread(target=_drain_stdout, args=(proc, "cloudflared"), daemon=True)
    drain_t.start()

    return proc, public_url, 443


def install_bore() -> str:
    """Ensure bore binary is installed and executable (race-condition safe for multi-GPU runners)."""
    bin_path = shutil.which("bore")
    if bin_path:
        return bin_path

    target = "/tmp/bore"
    if os.path.exists(target) and os.access(target, os.X_OK) and os.path.getsize(target) > 0:
        return target

    import fcntl
    lock_path = "/tmp/bore_install.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(target) and os.access(target, os.X_OK) and os.path.getsize(target) > 0:
                return target

            logger.info("Downloading bore binary...")
            url = "https://github.com/ekzhang/bore/releases/download/v0.5.0/bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz"
            tar_path = f"/tmp/bore_{os.getpid()}.tar.gz"
            tmp_dir = f"/tmp/bore_extract_{os.getpid()}"
            os.makedirs(tmp_dir, exist_ok=True)
            subprocess.run(["curl", "-sL", "-o", tar_path, url], check=True)
            subprocess.run(["tar", "-xzf", tar_path, "-C", tmp_dir], check=True)
            extracted_bin = os.path.join(tmp_dir, "bore")
            subprocess.run(["chmod", "+x", extracted_bin], check=True)
            os.replace(extracted_bin, target)
            subprocess.run(["chmod", "+x", target], check=True)
            try:
                os.remove(tar_path)
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            return target
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def start_bore_tunnel(local_port: int, server: str = "bore.pub", remote_port: Optional[int] = None) -> Tuple[subprocess.Popen, str, int]:
    """
    Start bore tunnel on local_port.

    Returns:
        (process, public_host, public_port)
    """
    bin_path = install_bore()
    cmd = [bin_path, "local", str(local_port), "--to", server]
    if remote_port:
        cmd.extend(["--port", str(remote_port)])

    logger.info("Starting bore tunnel for localhost:%d to %s...", local_port, server)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    pub_port = None
    start_time = time.time()

    while time.time() - start_time < 30:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.2)
            continue

        match = (
            re.search(r"bore\.pub:(\d+)", line)
            or re.search(r"bound_port=(\d+)", line)
            or re.search(r"listening at [^:]+:(\d+)", line)
            or re.search(r"port (\d+)", line)
        )
        if match:
            pub_port = int(match.group(1))
            logger.info("bore.pub Tunnel established at %s:%d", server, pub_port)
            break

    if not pub_port:
        proc.terminate()
        raise RuntimeError("Failed to obtain bore tunnel port within timeout")

    # Start background drain thread to prevent pipe buffer from filling and freezing bore
    drain_t = threading.Thread(target=_drain_stdout, args=(proc, "bore"), daemon=True)
    drain_t.start()

    return proc, server, pub_port

