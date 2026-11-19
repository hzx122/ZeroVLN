"""One NCCL device per process, including a relay that owns two TP GPUs.

Ray's initial barrier uses every visible GPU before any tensor collective has
initialized its device set. That creates different communicator keys/world
sizes for a one-GPU learner and a two-GPU relay. Initialize explicitly on the
relay's first device; TP fanout remains the separate CUDA IPC operation.
"""
import ray.util.collective as collective
import torch

from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine


class SingleDeviceNCCLCheckpointEngine(NCCLCheckpointEngine):
    def init_process_group(self, rank, world_size, master_metadata):
        if rank < 0:
            self.rank, self.world_size = rank, world_size
            return
        if self.rebuild_group or not collective.is_group_initialized(self.group_name):
            collective.init_collective_group(world_size, rank, "nccl", self.group_name)
            self.rank, self.world_size = rank, world_size
        elif self.rank != rank or self.world_size != world_size:
            raise ValueError("direct NCCL rank/world changed within a persistent group")
        if rank > 0:
            self._connect_zmq_client(master_metadata)
        # Both sides' transport buffers must be on logical device zero. Do not
        # shrink CUDA visibility: the relay still needs both devices for IPC.
        device = self.send_buf.device
        index = device.index if isinstance(device, torch.device) else device.id
        if index != 0:
            raise ValueError("direct NCCL transport buffer must use logical GPU0")
        collective.allreduce(torch.ones(1, dtype=torch.int64, device="cuda:0"),
                             group_name=self.group_name)

