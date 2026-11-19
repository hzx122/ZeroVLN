"""Preserve uneven FSDP shards during CPU offload.

FSDPModule._apply invokes reset_sharded_param immediately after Tensor.to.
For an uneven shard that method copies the new CPU tensor into padded CPU
storage. An asynchronous D2H transfer has not necessarily completed then;
synchronizing after Module.to is too late to protect that internal CPU read.
"""
import torch


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache=True):
    model.to("cpu", non_blocking=False)
    if empty_cache:
        from verl.utils.device import get_torch_device
        get_torch_device().empty_cache()


def install_blocking_fsdp2_offload():
    """Use the original synchronous transfer in this engine's process."""
    from verl.utils import fsdp_utils
    fsdp_utils.offload_fsdp2_model_to_cpu = offload_fsdp2_model_to_cpu
