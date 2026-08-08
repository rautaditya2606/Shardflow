"""
Global PyTest fixtures for ShardFlow test suite.
"""

import os
import sys
import socket
import pytest
import torch

# Ensure repository and tests directory are in Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shardflow.registry.app import _reset_registry_state


@pytest.fixture(autouse=True)
def clean_cuda_memory():
    """Ensure CUDA memory is cleared before and after each test function."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the FastAPI registry memory state before each test."""
    _reset_registry_state()
    yield
    _reset_registry_state()


@pytest.fixture
def free_port() -> int:
    """Get an ephemeral free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
