""" DARTS-style MixedOperation modules for QNAS.

MixedOp runs all candidate operations in parallel and combines their outputs
via softmax(alpha) weights. Alpha values are nn.Parameters updated by gradient
descent alongside the network weights, enabling DARTS-style architecture search.

Channel alignment: each op's output is projected to canonical_out_channels via
a learnable 1x1 conv. Spatial size is aligned to the input spatial size via
adaptive average pooling so the weighted sum is well-defined across ops with
different strides.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Dict, List, Any

from cnn.model import (
    functions_dict, FullyConnected,
    NoOp, MaxPooling, AvgPooling, StochasticPooling,
)

# Operations that do not have an output channel dimension (pass-through channels)
_POOLING_CLASSES = (MaxPooling, AvgPooling, StochasticPooling)


def _get_op_out_channels(fn_name: str, fn_cfg: Dict, in_channels: int) -> int:
    """Return the number of output channels for a given operation config."""
    func_name = fn_cfg["function"]
    if func_name == "NoOp":
        return in_channels
    params = fn_cfg.get("params", {})
    if "filters" in params:
        return params["filters"]
    # Pooling ops keep in_channels
    return in_channels


def _build_op(fn_name: str, fn_cfg: Dict, in_channels: int) -> nn.Module:
    """Instantiate a single operation from its config dict."""
    func_name = fn_cfg["function"]
    if func_name == "NoOp":
        return NoOp()
    cls = functions_dict[func_name]
    params = dict(fn_cfg.get("params", {}))
    # Set in_channels for ops that need it (conv/residual blocks)
    primary_blocks = {
        "ConvBlock", "DWConvBlock", "SEConvBlock", "DefConvBlock",
        "CBAMConvBlock", "MBConv", "MBConvV2", "MBConv_EPPGA",
        "ResidualV1", "ResidualV1Pr", "ResidualV1CBAM",
    }
    if func_name in primary_blocks:
        params["in_channels"] = in_channels
    return cls(**params)


class MixedOp(nn.Module):
    """Runs all candidate ops in parallel and combines via softmax(alpha) weighted sum.

    Args:
        fn_dict: dict mapping op names to their config (function + params).
        fn_list: ordered list of op names (defines alpha index order).
        in_channels: number of input channels flowing into this node.
        alpha_init: np.array of shape [num_ops], initial alpha logit values.
    """

    def __init__(self, fn_dict: Dict, fn_list: List[str],
                 in_channels: int, alpha_init: np.ndarray):
        super().__init__()

        self.fn_list = fn_list
        self.num_ops = len(fn_list)

        # Determine canonical output channels (max across all ops)
        out_channels_per_op = [
            _get_op_out_channels(name, fn_dict[name], in_channels)
            for name in fn_list
        ]
        self.canonical_out_channels = max(out_channels_per_op)

        # Build ops and channel projectors
        ops = []
        projectors = []
        for name, op_out_ch in zip(fn_list, out_channels_per_op):
            op = _build_op(name, fn_dict[name], in_channels)
            ops.append(op)
            if op_out_ch == self.canonical_out_channels:
                proj = nn.Identity()
            else:
                proj = nn.Conv2d(op_out_ch, self.canonical_out_channels,
                                 kernel_size=1, bias=False)
                init.kaiming_normal_(proj.weight, nonlinearity="relu")
            projectors.append(proj)

        self.ops = nn.ModuleList(ops)
        self.projectors = nn.ModuleList(projectors)

        # Alpha logits as learnable parameter
        alpha_tensor = torch.tensor(alpha_init, dtype=torch.float32)
        self.alpha = nn.Parameter(alpha_tensor)

    @property
    def out_channels(self) -> int:
        return self.canonical_out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.alpha, dim=0)
        target_h, target_w = x.shape[2], x.shape[3]

        result = None
        for i, (op, proj, w) in enumerate(zip(self.ops, self.projectors, weights)):
            if isinstance(op, NoOp):
                # NoOp: use input directly; project if channels differ
                out = proj(x)
            else:
                out = op(x)
                # Align spatial size back to input size
                if out.shape[2] != target_h or out.shape[3] != target_w:
                    out = F.adaptive_avg_pool2d(out, (target_h, target_w))
                out = proj(out)

            if result is None:
                result = w * out
            else:
                result = result + w * out

        return result


class MixedNetworkGraph(nn.Module):
    """Sequential network of MixedOp nodes, mirroring NetworkGraph.create_functions().

    Each node position runs ALL candidate operations in parallel (softmax-weighted sum).
    Alpha logits are nn.Parameters updated during training.
    """

    def __init__(self, num_classes: int, in_channels: int = 3,
                 network_gap: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.use_gap = network_gap
        self.mixed_ops: nn.ModuleList = nn.ModuleList()
        self.fc: nn.Module = None

    def create_mixed_ops(self, fn_dict: Dict[str, Any], fn_list: List[str],
                         alpha_matrix: np.ndarray):
        """Build MixedOp at each node position.

        Args:
            fn_dict: full operation config dict (may include ops not in fn_list).
            fn_list: ordered list of op names matching alpha_matrix columns.
            alpha_matrix: np.array of shape [num_nodes, num_ops], initial alpha logits.
        """
        num_nodes = alpha_matrix.shape[0]
        # Filter fn_dict to fn_list entries only
        node_fn_dict = {name: fn_dict[name] for name in fn_list if name in fn_dict}

        cur_channels = self.in_channels
        ops = []
        for node_idx in range(num_nodes):
            alpha_init = alpha_matrix[node_idx]  # shape [num_ops]
            mixed_op = MixedOp(
                fn_dict=node_fn_dict,
                fn_list=fn_list,
                in_channels=cur_channels,
                alpha_init=alpha_init,
            )
            ops.append(mixed_op)
            cur_channels = mixed_op.canonical_out_channels

        self.mixed_ops = nn.ModuleList(ops)

    def get_trained_alpha(self) -> np.ndarray:
        """Extract trained alpha values from all MixedOp nodes.

        Returns:
            np.ndarray of shape [num_nodes, num_ops].
        """
        alphas = []
        for op in self.mixed_ops:
            alphas.append(op.alpha.detach().cpu().numpy())
        return np.stack(alphas, axis=0)

    def get_softmax_weights(self) -> np.ndarray:
        """Return softmax-normalized weights per node, shape [num_nodes, num_ops]."""
        weights = []
        for op in self.mixed_ops:
            w = F.softmax(op.alpha, dim=0).detach().cpu().numpy()
            weights.append(w)
        return np.stack(weights, axis=0)

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        for mixed_op in self.mixed_ops:
            x = mixed_op(x)
            if debug:
                print(f"MixedOp output shape: {x.shape}")

        x = torch.flatten(x, 1)
        if self.fc is None:
            self.fc = FullyConnected(input_features=x.size(1),
                                     units=self.num_classes)
            self.fc = self.fc.to(x.device)
        return self.fc(x)
