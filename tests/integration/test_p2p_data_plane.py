"""
Integration tests for v2 peer-to-peer data plane cluster with direct streaming back to Gateway.
"""

import asyncio
import pytest
import torch

from shardflow.node.node import PipelineNode
from shardflow.transport.connection import NodeClient, StreamReceiverServer
from shardflow.transport.protocol import MessageType, TensorMessage
from tests.fixtures.mock_models import create_dummy_model_slice


@pytest.mark.asyncio
async def test_p2p_data_plane_cluster_generation(free_port: int):
    """
    Test 2-node peer-to-peer cluster:
    - Node 0: embedding + layer 0
    - Node 1: layer 1 + LM head + direct stream-back to Gateway
    """
    # 1. Gateway Stream Receiver
    stream_receiver = StreamReceiverServer(host="127.0.0.1", port=0)
    gw_stream_port = await stream_receiver.start()

    # 2. Node 1 (Terminal node with LM head)
    node1_slice = create_dummy_model_slice(layer_start=1, layer_end=2, is_first=False, is_last=True)
    node1 = PipelineNode(
        model_slice=node1_slice,
        is_first_node=False,
        is_last_node=True,
        listen_host="127.0.0.1",
        listen_port=0,
    )
    await node1.start()
    node1_port = node1._server._server.sockets[0].getsockname()[1]

    # 3. Node 0 (Data Plane Controller with Embedding)
    node0_slice = create_dummy_model_slice(layer_start=0, layer_end=1, is_first=True, is_last=False)
    node0 = PipelineNode(
        model_slice=node0_slice,
        is_first_node=True,
        is_last_node=False,
        next_node_host="127.0.0.1",
        next_node_port=node1_port,
        listen_host="127.0.0.1",
        listen_port=0,
    )
    await node0.start()
    node0_port = node0._server._server.sockets[0].getsockname()[1]

    session_id = "p2p-cluster-test-sess"
    stream_q = stream_receiver.register_session(session_id)

    # 4. Gateway initiates session on Node 0
    gw_to_node0 = NodeClient("127.0.0.1", node0_port)
    await gw_to_node0.connect()

    start_msg = TensorMessage(
        msg_type=MessageType.START_SESSION,
        session_id=session_id,
        prompt_tokens=[1, 2, 3],
        max_tokens=5,
        temperature=0.0,
        stream_back_host="127.0.0.1",
        stream_back_port=gw_stream_port,
    )
    await gw_to_node0.send(start_msg)

    # 5. Receive streamed tokens directly from terminal node
    received_tokens = []
    while True:
        token_msg = await asyncio.wait_for(stream_q.get(), timeout=5.0)
        if token_msg.is_eos or (token_msg.finish_reason is not None and token_msg.finish_reason != ""):
            break
        received_tokens.append(token_msg.token_id)
        if len(received_tokens) >= 5:
            break

    assert len(received_tokens) > 0
    assert all(isinstance(t, int) for t in received_tokens)

    # Cleanup
    stream_receiver.unregister_session(session_id)
    await gw_to_node0.close()
    await node0.stop()
    await node1.stop()
    await stream_receiver.stop()
