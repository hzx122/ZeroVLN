"""Isolated actual-model worker; imported only after an explicit execute gate."""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
import time

import torch
from torch.distributed.tensor import DTensor

from action_head_rl.checkpoint import training_state_digest
from action_head_rl.joint_audit import forbid_updates, pack_rank_pair, tensor_difference
from action_head_rl.worker import ActionHeadGRPOWorker, _cpu_model_positions
from verl.single_controller.base.decorator import Dispatch, register


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def model_state(engine):
    scheduler = engine.lr_scheduler.state_dict()
    # LRScheduler serializes its instance __dict__. Our temporary method
    # tripwire is not scheduler state; all genuine counters remain included.
    if callable(scheduler.get("step")):
        scheduler.pop("step")
    return {
        "parameters": training_state_digest(dict(engine.module.named_parameters())),
        "optimizer": training_state_digest(engine.optimizer.state_dict()),
        "scheduler": training_state_digest(scheduler),
        "optimizer_state_count": len(engine.optimizer.state),
        "optimizer_step_values": sorted({float(row["step"]) for row in engine.optimizer.state.values() if "step" in row}),
        "scheduler_last_epoch": engine.lr_scheduler.last_epoch,
    }


@contextmanager
def observe_outputs(module):
    """Detached audit-only hooks. Removed before every throughput measurement."""
    state = {"identity": None, "values": {}}
    def save(name, tensor):
        key = state["identity"] + "/" + name
        if key in state["values"]:
            raise RuntimeError(f"unexpected duplicate root/LM-head audit output: {key}")
        state["values"][key] = tensor.detach().float().cpu()
    def root_hook(_module, _args, result):
        for name in ("query_hidden", "new_logprob", "text_logprobs"):
            if name in result:
                save(name, result[name])
    def logits_hook(_module, _args, result):
        save("selected_response_logits", result)
    hooks = [module.register_forward_hook(root_hook),
             module.qwen.get_output_embeddings().register_forward_hook(logits_hook)]
    try:
        yield state
    finally:
        for hook in hooks:
            hook.remove()


class JointForwardAuditWorker(ActionHeadGRPOWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def prepare_audit(self, joint_rows, separate_rows):
        if len(self.engine.optimizer.state):
            raise ValueError("zero-update audit must start with a fresh empty optimizer")
        if any(parameter.dtype != torch.float32 for parameter in self.engine.module.parameters()
               if parameter.is_floating_point()):
            raise ValueError("zero-update audit requires FP32 parameter storage")
        from action_head_rl.text_policy import is_text_record
        if any(len(row["query_routed_experts"]) != self.engine.module.query_encoder.QUERY_COUNT
               for row in joint_rows if not is_text_record(row)):
            raise ValueError("audit fixture query count differs from initialized model")
        positions = lambda payload: _cpu_model_positions(self.engine.module.query_forward.model.model, payload)
        self._audit_packed = pack_rank_pair(joint_rows, separate_rows, self.model_config.processor, positions)
        shapes = {mode: [{"identity": identity, "family": family,
                         "shapes": {key: list(data[key].shape) for key in data.keys()}}
                        for identity, family, data in rows]
                  for mode, rows in self._audit_packed.items()}
        return {"rank": self.rank, "shapes": shapes}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def zero_update_audit(self, plan):
        """No generic worker train/finish/publication method is called here."""
        thresholds = plan["thresholds"]
        engine = self.engine
        rank = torch.distributed.get_rank()
        before = model_state(engine)
        rng = engine.checkpoint_manager.get_rng_state()
        rng_digest = training_state_digest(rng)
        parameters = list(engine.module.named_parameters())
        report = {"rank": rank, "before": before, "cold_audit_seconds": {}, "timings": [],
                  "optimizer_updates": 0, "thresholds": thresholds}
        # Keep the baseline sharded gradient on CPU, not a second 35B GPU
        # gradient allocation. Never call full_tensor/all-gather for comparison.
        baseline_gradients = None
        baseline_outputs = None
        baseline_losses = None
        with forbid_updates(engine) as attempts, engine.train_mode(disable_auto_offload=False):
            packed = {mode: [(identity, family, data.to(self.device_name)) for identity, family, data in rows]
                      for mode, rows in self._audit_packed.items()}

            def run(mode, observer=None):
                os.environ["ACTION_HEAD_JOINT_TEXT_FM_FORWARD"] = "1" if mode == "joint" else "0"
                engine.optimizer_zero_grad()
                engine.checkpoint_manager.load_rng_state(rng)
                # Full policy syncs each backward. Deferring sync retains a
                # complete unsharded 35B gradient and is not this audit's path.
                engine.module.set_requires_gradient_sync(True, recurse=True)
                if hasattr(engine.module, "set_is_last_backward"):
                    engine.module.set_is_last_backward(True)
                torch.cuda.synchronize()
                torch.distributed.barrier()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                losses = {}
                for identity, family, data in packed[mode]:
                    if observer is not None:
                        observer["identity"] = identity
                    result = engine.forward_backward_batch(data, loss_function=lambda **_: None, forward_only=False)
                    losses[identity] = losses.get(identity, 0.0) + float(result["loss"])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated() / 1024**3
                return losses, seconds, peak

            for mode in ("separate", "joint"):
                with observe_outputs(engine.module) as observer:
                    losses, seconds, peak = run(mode, observer)
                report["cold_audit_seconds"][mode] = seconds
                if mode == "separate":
                    baseline_losses, baseline_outputs = losses, observer["values"]
                    baseline_gradients = {name: None if parameter.grad is None else
                        local_tensor(parameter.grad).detach().cpu().clone() for name, parameter in parameters}
                else:
                    current = observer["values"]
                    if set(current) != set(baseline_outputs) or set(losses) != set(baseline_losses):
                        raise ValueError("separate/joint audit output identities differ")
                    output_checks = {name: tensor_difference(baseline_outputs[name], value,
                        atol=thresholds["output_atol"], rtol=thresholds["output_rtol"])
                        for name, value in current.items()}
                    loss_checks = {name: {"baseline": baseline_losses[name], "joint": value,
                        "passed": math.isclose(baseline_losses[name], value,
                            abs_tol=thresholds["loss_atol"], rel_tol=thresholds["loss_rtol"])}
                        for name, value in losses.items()}
                    gradient_checks = {}
                    for name, parameter in parameters:
                        current_grad = None if parameter.grad is None else local_tensor(parameter.grad)
                        gradient_checks[name] = tensor_difference(baseline_gradients[name], current_grad,
                            atol=thresholds["gradient_atol"], rtol=thresholds["gradient_rtol"])
                    norm_sq = sum(row.get("baseline_squared", 0.0) for row in gradient_checks.values())
                    error_sq = sum(row.get("error_squared", 0.0) for row in gradient_checks.values())
                    report.update(output_checks=output_checks, loss_checks=loss_checks,
                        gradient_checks=gradient_checks, local_gradient_squared=norm_sq,
                        local_gradient_error_squared=error_sq,
                        numeric_passed=all(row["passed"] for table in (output_checks, loss_checks, gradient_checks)
                                           for row in table.values()))
            del baseline_gradients, baseline_outputs
            # Warm runs and AB/BA measurements exclude hooks, CPU packing,
            # transfers, gradient copies and initialization/first compilation.
            for _ in range(plan["warmup"]):
                for mode in ("separate", "joint"):
                    run(mode)
            for repeat in range(plan["repeats"]):
                modes = ("separate", "joint") if repeat % 2 == 0 else ("joint", "separate")
                for mode in modes:
                    _, seconds, peak = run(mode)
                    report["timings"].append({"repeat": repeat, "mode": mode, "seconds": seconds,
                                              "peak_memory_allocated_gib": peak})
            # Compare before leaving: EngineTrainModeCtx.__exit__ clears grads.
            engine.optimizer_zero_grad()
            engine.checkpoint_manager.load_rng_state(rng)
            report["after"] = model_state(engine)
            report["rng_restored"] = training_state_digest(engine.checkpoint_manager.get_rng_state()) == rng_digest
            report["forbidden_call_attempts"] = list(attempts)
            report["state_unchanged"] = before == report["after"]
            if not report["state_unchanged"] or not report["rng_restored"] or attempts:
                raise RuntimeError("zero-update audit changed protected state")
        self._audit_packed = None
        return report
