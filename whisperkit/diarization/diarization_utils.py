#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.
#

from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    _TORCH_OPS_REGISTRY, register_torch_op
)
from coremltools.converters.mil.mil import Builder as mb


def register_torch_ops_for_speaker_segmenter():
    # Note: SpeakerSegmenter requires one hot encoding for `self.postproc(powerset_probs, soft=False)``
    # which is not implemented in coremltools as of 8.0b2
    if "one_hot" not in _TORCH_OPS_REGISTRY:
        @register_torch_op
        def one_hot(context, node):
            indices, one_hot_vector_size = _get_inputs(context, node, expected=2)
            out = mb.one_hot(indices=indices, one_hot_vector_size=one_hot_vector_size, name=node.name)
            context.add(out)

    if "unfold" not in _TORCH_OPS_REGISTRY:
        @register_torch_op
        def unfold(context, node):
            x, axis, window_size, window_stride = _get_inputs(context, node, expected=4)
            y = mb.sliding_windows(x=x, axis=axis, size=window_size.val, stride=window_stride.val)
            context.add(y, torch_name="waveform_sliding_window")
