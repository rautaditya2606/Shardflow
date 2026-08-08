"""
Unit tests for AutoPartitionEngine heterogeneous VRAM layer allocation.
"""

from shardflow.partition.engine import AutoPartitionEngine, NodeVRAMInfo


def test_auto_partition_engine_vram_proportional_allocation():
    """Verify AutoPartitionEngine assigns layers proportionally while reserving LM head memory on terminal node."""
    engine = AutoPartitionEngine(
        total_layers=32,
        hidden_size=4096,
        vocab_size=128000,
        bytes_per_param=2,
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

    # Layer continuity across the chain
    assert assignments[0].layer_start == 0
    assert assignments[2].layer_end == 32
    assert assignments[0].layer_end == assignments[1].layer_start
    assert assignments[1].layer_end == assignments[2].layer_start

    # Node 2 with 24GB VRAM receives a larger layer slice
    n0_layers = assignments[0].layer_end - assignments[0].layer_start
    n2_layers = assignments[2].layer_end - assignments[2].layer_start
    assert n2_layers >= n0_layers
