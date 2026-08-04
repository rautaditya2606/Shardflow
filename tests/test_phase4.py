"""
Phase 4 unit test suite — test Auto-Partition Engine, Request Scheduler, and Session lifecycle.
"""

import pytest
import asyncio
from shardflow.partition.engine import AutoPartitionEngine, NodeVRAMInfo
from shardflow.scheduler.session import Session, SessionState
from shardflow.scheduler.scheduler import RequestScheduler


def test_auto_partition_engine_vram_allocation():
    # Model parameters for LLaMA 8B (32 layers, hidden_size=4096, vocab=128000)
    engine = AutoPartitionEngine(
        total_layers=32,
        hidden_size=4096,
        vocab_size=128000,
        bytes_per_param=2,  # fp16
    )

    nodes = [
        NodeVRAMInfo(node_id="node-0", vram_available_mb=16000.0, vram_total_mb=16000.0),
        NodeVRAMInfo(node_id="node-1", vram_available_mb=16000.0, vram_total_mb=16000.0),
        NodeVRAMInfo(node_id="node-2", vram_available_mb=24000.0, vram_total_mb=24000.0),
    ]

    assignments = engine.compute_partition(nodes)

    assert len(assignments) == 3
    assert assignments[0].is_first is True
    assert assignments[2].is_last is True

    # Check total layers cover 32
    assert assignments[0].layer_start == 0
    assert assignments[2].layer_end == 32
    assert assignments[0].layer_end == assignments[1].layer_start
    assert assignments[1].layer_end == assignments[2].layer_start

    # Node 2 has 24GB VRAM so it should get a proportional share even after LM head budget
    assert assignments[2].layer_end - assignments[2].layer_start >= assignments[0].layer_end - assignments[0].layer_start


@pytest.mark.asyncio
async def test_request_scheduler_lifecycle():
    scheduler = RequestScheduler(max_concurrent_sessions=2)
    session = Session(prompt="Hello world", max_tokens=10)

    # 1. Submit request
    session_id = await scheduler.submit_request(session)
    assert session_id == session.session_id
    assert scheduler.pending_queue.qsize() == 1

    # 2. Dequeue session
    dequeued_session = await scheduler.pending_queue.get()
    scheduler.active_sessions[dequeued_session.session_id] = dequeued_session

    dequeued_session.mark_prefill()
    assert dequeued_session.state == SessionState.PREFILLING

    dequeued_session.mark_decode()
    assert dequeued_session.state == SessionState.DECODING

    dequeued_session.mark_completed()
    assert dequeued_session.state == SessionState.COMPLETED

    # 3. Cleanup
    scheduler.remove_session(session_id)
    assert len(scheduler.active_sessions) == 0
