"""Restore the pinned VERL v2 text checkpoint when its optimizer was frozen."""
from pathlib import Path


def _frozen_scheduler_state(text_checkpoint):
    import torch.distributed.checkpoint as dcp

    # The v2 TorchDist backend stores non-tensor state in this ShardedObject.
    # Read only that small object, without materializing optimizer tensors or
    # introducing an extra distributed collective among model-parallel ranks.
    state = {'common_state/shard_0_1': None}
    dcp.load_state_dict(state, storage_reader=dcp.FileSystemReader(
        Path(text_checkpoint) / 'optimizer' / 'dist_ckpt'), no_dist=True)
    common = state['common_state/shard_0_1']
    if isinstance(common, list):
        common = common[0]
    return common['lr_scheduler']


def load_text_checkpoint(engine, text_checkpoint, *, trainable=True):
    """Keep model/RNG/scheduler restoration, with no invented frozen Adam state.

    A never-updated distributed Adam checkpoint has no exp_avg/exp_avg_sq.
    MCore's loading template allocates those moments, so its ordinary optimizer
    loader cannot read this valid frozen checkpoint. The trainable FM branch
    is restored independently by the joint worker and is unaffected here.
    """
    path = str(text_checkpoint)
    if trainable:
        engine.load_checkpoint(path, del_local_after_load=False)
        return dict(optimizer_loaded=True, frozen_optimizer=False)

    manager = engine.checkpoint_mananager
    scheduler = _frozen_scheduler_state(path)
    contents = manager.checkpoint_load_contents
    manager.checkpoint_load_contents = [x for x in contents if x != 'optimizer']
    try:
        # Restore the saved scheduler before the native model/RNG load so the
        # latter remains the last operation that sets the training RNG state.
        engine.lr_scheduler.load_state_dict(scheduler)
        engine.load_checkpoint(path, del_local_after_load=False)
    finally:
        manager.checkpoint_load_contents = contents
    return dict(optimizer_loaded=False, frozen_optimizer=True,
                frozen_optimizer_reason='no updates; Adam moments are absent',
                scheduler_loaded=True)
