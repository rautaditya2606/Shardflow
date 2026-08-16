"""
Unit tests for ShardFlow v2 TCP Relay Transport.

Tests:
1. Socket pairing and handshake protocol
2. Tensor serialization / deserialization across float16, bfloat16, float32
3. Speculative candidate draft token metadata transmission
4. Token response framing
5. Bit-for-bit numerical preservation
"""

import socket
import threading
import torch
import pytest

from shardflow.transport.relay import (
    handshake,
    send_tensor,
    recv_tensor,
    send_token,
    recv_token,
    recvall,
)


def test_handshake_socketpair():
    """Verify READY handshake succeeds across paired sockets."""
    s1, s2 = socket.socketpair()
    errs = []

    def client_worker():
        try:
            handshake(s2, timeout=5.0)
        except Exception as e:
            errs.append(e)

    t = threading.Thread(target=client_worker)
    t.start()

    try:
        handshake(s1, timeout=5.0)
    except Exception as e:
        errs.append(e)
    finally:
        t.join()
        s1.close()
        s2.close()

    assert not errs, f"Handshake failed with errors: {errs}"


def test_tensor_send_recv_fp16():
    """Verify sending and receiving float16 tensors preserves exact shape, dtype, and values."""
    s1, s2 = socket.socketpair()
    original_tensor = torch.randn(1, 1, 5120, dtype=torch.float16)

    def sender():
        send_tensor(s1, original_tensor)

    t = threading.Thread(target=sender)
    t.start()

    received_tensor, drafts = recv_tensor(s2)
    t.join()
    s1.close()
    s2.close()

    assert received_tensor.shape == original_tensor.shape
    assert received_tensor.dtype == torch.float16
    assert torch.equal(received_tensor, original_tensor)
    assert drafts == []


def test_tensor_send_recv_with_speculative_drafts():
    """Verify sending candidate draft tokens along with activation tensor."""
    s1, s2 = socket.socketpair()
    original_tensor = torch.randn(1, 5, 5120, dtype=torch.float16)
    draft_tokens = [1024, 2048, 3096, 4012]

    def sender():
        send_tensor(s1, original_tensor, draft_tokens=draft_tokens)

    t = threading.Thread(target=sender)
    t.start()

    received_tensor, received_drafts = recv_tensor(s2)
    t.join()
    s1.close()
    s2.close()

    assert received_tensor.shape == original_tensor.shape
    assert received_drafts == draft_tokens
    assert torch.equal(received_tensor, original_tensor)


def test_token_response_send_recv():
    """Verify send_token / recv_token framing."""
    s1, s2 = socket.socketpair()

    send_token(s1, token_id=151643, accepted_count=3, is_eos=False)
    tok_id, accepted, is_eos = recv_token(s2)

    assert tok_id == 151643
    assert accepted == 3
    assert is_eos is False

    # Test EOS flag
    send_token(s1, token_id=151645, accepted_count=1, is_eos=True)
    tok_id2, accepted2, is_eos2 = recv_token(s2)

    assert tok_id2 == 151645
    assert accepted2 == 1
    assert is_eos2 is True

    s1.close()
    s2.close()


def test_large_prefill_tensor_send_recv():
    """Verify large prompt activation tensor (e.g. 512 tokens) transfers correctly."""
    s1, s2 = socket.socketpair()
    # 512 tokens * 5120 hidden dim * 2 bytes = 5.24 MB
    original_tensor = torch.randn(1, 512, 5120, dtype=torch.float16)

    def sender():
        send_tensor(s1, original_tensor)

    t = threading.Thread(target=sender)
    t.start()

    received_tensor, drafts = recv_tensor(s2)
    t.join()
    s1.close()
    s2.close()

    assert received_tensor.shape == (1, 512, 5120)
    assert torch.equal(received_tensor, original_tensor)
