"""
Unit tests for RequestScheduler and Session state machine.
"""

import pytest
from shardflow.scheduler.session import Session, SessionState
from shardflow.scheduler.scheduler import RequestScheduler


@pytest.mark.asyncio
async def test_session_lifecycle_state_machine():
    """Verify session state transitions from PENDING -> PREFILLING -> DECODING -> COMPLETED."""
    scheduler = RequestScheduler(max_concurrent_sessions=2)
    session = Session(prompt="Test prompt", max_tokens=10)

    session_id = await scheduler.submit_request(session)
    assert session_id == session.session_id
    assert scheduler.pending_queue.qsize() == 1

    dequeued = await scheduler.pending_queue.get()
    scheduler.active_sessions[dequeued.session_id] = dequeued

    dequeued.mark_prefill()
    assert dequeued.state == SessionState.PREFILLING

    dequeued.mark_decode()
    assert dequeued.state == SessionState.DECODING

    dequeued.mark_completed()
    assert dequeued.state == SessionState.COMPLETED

    scheduler.remove_session(session_id)
    assert len(scheduler.active_sessions) == 0


@pytest.mark.asyncio
async def test_session_cancellation_in_queue():
    """Verify cancelling a queued request marks it cancelled and prevents execution."""
    scheduler = RequestScheduler(max_concurrent_sessions=2)
    s1 = Session(prompt="Session 1", max_tokens=5)
    s2 = Session(prompt="Session 2", max_tokens=5)

    await scheduler.submit_request(s1)
    await scheduler.submit_request(s2)

    assert scheduler.cancel_session(s2.session_id) is True
    assert s2.state == SessionState.CANCELLED

    deq1 = await scheduler.pending_queue.get()
    assert deq1.session_id == s1.session_id

    deq2 = await scheduler.pending_queue.get()
    assert deq2.session_id == s2.session_id
    assert deq2.state == SessionState.CANCELLED
