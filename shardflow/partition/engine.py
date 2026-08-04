"""
Auto-Partition Engine — assigns layer ranges to nodes based on reported VRAM.

Rules:
1. VRAM-proportional: a node with 24GB gets ~2x the layers of a node with 12GB.
2. LM head budget: for final node, subtract LM head VRAM from available VRAM before allocating layers.
3. Threshold check: node must have at least `layer_size_mb * 1.5` VRAM, else reject.
4. Total layers must cover all `num_hidden_layers` of the model.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NodeVRAMInfo:
    node_id: str
    vram_available_mb: float
    vram_total_mb: float


@dataclass
class PartitionAssignment:
    node_id: str
    layer_start: int
    layer_end: int  # exclusive
    is_first: bool
    is_last: bool


class AutoPartitionEngine:
    """Calculates optimal layer distribution across N nodes."""

    def __init__(
        self,
        total_layers: int,
        hidden_size: int,
        vocab_size: int,
        bytes_per_param: int = 2,  # fp16/bf16 = 2 bytes, int4 = 0.5 bytes
        min_vram_multiplier: float = 1.5,
    ):
        self.total_layers = total_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.bytes_per_param = bytes_per_param
        self.min_vram_multiplier = min_vram_multiplier

        # Calculate estimated size per layer in MB
        # LLaMA transformer layer ~ 4 * hidden_size^2 + 3 * intermediate_size * hidden_size
        # Simplified estimate: (model_params / num_layers) * bytes_per_param / 1e6
        # For LLaMA 8B (32 layers) ~ 250MB per layer in fp16
        self.estimated_layer_size_mb = self._estimate_layer_size_mb()
        self.lm_head_size_mb = (self.vocab_size * self.hidden_size * self.bytes_per_param) / 1e6

    def _estimate_layer_size_mb(self) -> float:
        # Standard LLaMA layer parameter estimation
        # Attention: Q, K, V, O projections = 4 * hidden_size^2
        # MLP: Gate, Up, Down projections ~ 3 * (4 * hidden_size) * hidden_size = 12 * hidden_size^2
        layer_params = 16 * (self.hidden_size ** 2)
        return (layer_params * self.bytes_per_param) / 1e6

    def compute_partition(
        self, nodes: List[NodeVRAMInfo]
    ) -> List[PartitionAssignment]:
        """
        Compute layer assignments for a list of nodes sorted by node registration order or performance.
        The last node in the list is assigned the LM head.
        """
        if not nodes:
            raise ValueError("Cannot partition with empty node list")

        # 1. Filter out nodes below minimum VRAM threshold
        min_vram_required = self.estimated_layer_size_mb * self.min_vram_multiplier
        valid_nodes = [
            n for n in nodes if n.vram_available_mb >= min_vram_required
        ]

        if not valid_nodes:
            raise ValueError(
                f"No nodes satisfy minimum VRAM threshold ({min_vram_required:.1f} MB)"
            )

        num_nodes = len(valid_nodes)
        
        # 2. Adjust VRAM budgets (subtract LM head size for final node)
        effective_vram = []
        for i, n in enumerate(valid_nodes):
            vram = n.vram_available_mb
            if i == num_nodes - 1:  # Last node gets LM head
                vram = max(1.0, vram - self.lm_head_size_mb)
            effective_vram.append(vram)

        total_effective_vram = sum(effective_vram)

        # 3. Allocate layers proportionally
        assignments = []
        current_layer = 0

        for i, n in enumerate(valid_nodes):
            is_first = (i == 0)
            is_last = (i == num_nodes - 1)

            if is_last:
                # Last node gets remaining layers
                layer_count = self.total_layers - current_layer
            else:
                ratio = effective_vram[i] / total_effective_vram
                layer_count = round(self.total_layers * ratio)
                # Ensure each intermediate node gets at least 1 layer
                layer_count = max(1, min(layer_count, self.total_layers - current_layer - (num_nodes - 1 - i)))

            layer_end = current_layer + layer_count
            assignments.append(
                PartitionAssignment(
                    node_id=n.node_id,
                    layer_start=current_layer,
                    layer_end=layer_end,
                    is_first=is_first,
                    is_last=is_last,
                )
            )
            current_layer = layer_end

        return assignments
