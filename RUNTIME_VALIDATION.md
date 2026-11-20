# Minimal runtime validation

These commands exercise real training and navigation with the configured model
weights and datasets. They require the environment in `README.md` and
`DEPENDENCIES.md`; no synthetic model, mocked simulator, or fabricated trajectory
is substituted. Use fresh output directories for every attempt.

## Bounded action-expert training

The explicit smoke option uses one node with four devices and one anchor per
device, retaining SAGE:R2R:RxR = 1:1:2. It invokes the same v8 exec4 trainer as the
default open-loop workflow, with a frozen brain and trainable query/action-head
parameters. It does not exercise joint reinforcement learning. Omitting this
option preserves the published four-node training recipe.

```bash
./scripts/train/run_openloop.sh prepare --smoke-steps 1 \
  --output-dir "$RL_ARTIFACT_ROOT/review-training"
./scripts/train/run_openloop.sh initialize \
  --output-dir "$RL_ARTIFACT_ROOT/review-training"
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train/run_openloop.sh run \
  --output-dir "$RL_ARTIFACT_ROOT/review-training" --node-rank 0 \
  --master-addr 127.0.0.1 --master-port 29671
```

Check `launch/run-node0.json` for the real exit code and the final checkpoint's
`report.json` for parameter deltas and the frozen-brain audit. Cold accelerator
kernel compilation may take substantially longer than subsequent updates.
Initialization intentionally follows the configured D16 initialization procedure;
it does not silently warm-start a trained action expert.

## One joint RL update on 16 devices

Start with the configured model, initial action-expert checkpoint, three-source
training manifest, and navigation assets. Set `SAGE_SOURCE_OBJECT_MAP_SHA256`,
`SAGE_NAVMESH_ROOT`, and `MP3D_HOUSE_ROOT` before starting the gyms. Use the
joint-RL training environment specified in `DEPENDENCIES.md`. Each TP2
inference replica shares two learner devices, so eight replicas cover this
single 16-device node. The command keeps the default joint training scope,
Trajectory-Level RLOO, reward settings, and native STOP.

```bash
export EXPR_NAME=review-rl-16 RAY_ADDRESS=local
export TRAIN_GYMS=8 TRAIN_PORT_BASE=24850 RENDER_PORT=24840 CPU_RENDER_PORT=24841
export SAGE_CPU_RENDER_PARALLELISM=2 CPU_RENDER_PARALLELISM=2
export BENCH_URLS=$(printf 'http://127.0.0.1:%s,' {24850..24857})
export BENCH_URLS=${BENCH_URLS%,}
opts=(recipe.nnodes=1 'recipe.learner_gpus_layout="16"'
      recipe.megatron_tp=2 recipe.megatron_pp=2 recipe.megatron_cp=1 recipe.megatron_ep=4
      recipe.rollout_actors=8 recipe.group_size=2 recipe.num_groups_per_step=4
      recipe.case_source_groups.sage=2 recipe.case_source_groups.r2r=1 recipe.case_source_groups.rxr=1
      recipe.rollout_groups_in_flight=4 recipe.stream_groups_per_update=4
      recipe.fast_policy_workers=1 recipe.colocated_vllm_replicas=8 recipe.v8_cuda_graphs=false
      recipe.vllm_max_num_seqs=8 recipe.vllm_max_num_batched_tokens=16384
      recipe.max_tool_calls=3 recipe.opt_steps=1 recipe.checkpoint_interval=1)
(
  set -e
  ./scripts/train/rloo/run_rl_services.sh run.mode=start "${opts[@]}"
  trap './scripts/train/rloo/run_rl_services.sh run.mode=stop' EXIT
  ./scripts/train/rloo/run_rloo.sh "${opts[@]}" "run.output_dir=$RL_ARTIFACT_ROOT/$EXPR_NAME"
)
```

The trainer records selected episodes in `train/rollouts.jsonl`. Check
`train/training_updates.jsonl` for both optimizer counts and gradient norms
on every rank. Check `train/metrics.jsonl` and the step checkpoint separately
for successful publication and persistence.

## One navigation case per benchmark

First set `SAGE_ROOT` to the dataset directory containing `SAGE-3D_VLN_Data`
and `InteriorGS`; `SAGE_SCENES_ROOT` separately points to the converted Gaussian
scenes. Set `RL_ARTIFACT_ROOT` to the writable artifact directory. Source the
completed environment file before launching.

Create a local smoke selection from your completed evaluation configuration
(the template paths must be replaced with actual weights and datasets):

```bash
python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('configs/gpu/evaluation.json').read_text())
config.update(nodes=1, evaluation_repeats=1, max_tool_calls=2)
Path('evaluation.smoke.local.json').write_text(json.dumps(config, indent=2)+'\n')
PY
./scripts/eval/run_evaluation.sh smoke --config evaluation.smoke.local.json \
  --output-dir "$RL_ARTIFACT_ROOT/review-evaluation" --smoke-gpus 0,1
```

This mode samples one case from each selected benchmark, starts one TP2 brain
service plus one Gym per benchmark, and shares the first selected device with
the action expert. Serving concurrency is capped at four sequences and 4,096
batched tokens, while retaining the configured context window. It reuses the unified evaluator's camera, prompts, context,
motion execution, and native scoring. The example explicitly limits tool calls
to two; this is a functionality check and its scores must not be compared with
the paper's full-budget results. The default evaluation mode is unchanged.

Success requires a zero launcher exit code and `smoke/smoke-summary.json` with
`complete: true`, `ok: true`, and a valid completed result for every selected
benchmark. A navigation failure can be a valid evaluated episode; infrastructure
errors and incomplete episodes cannot. Logs, model events, and per-attempt exit
receipts remain under the output directory.

## Relocatable inputs

The bundled R2R/RxR episode list retains all original episode identities,
instructions, and split membership. Its source paths are relative to
`RYNNBRAIN_NAV_DIR`, and both the bundled manifest hashes and original dataset
hashes are checked. The configured navmesh directory may reside on any explicitly
selected absolute path; its manifest, mesh identity, and navmesh content checks
remain enabled.

## Observed training check

On an existing PPU-ZW810E environment (Python 3.12.3, PyTorch 2.9.0,
Transformers 5.5.4, and vLLM 0.18.0+ppu2.0.0), one actual training update
completed on four devices with global batch size four and loss 0.888292.
The final checkpoint reported `ok: true`; the query and action-head parameter
change norms were 0.00000172444 and 0.00170336 respectively. The frozen brain
had zero trainable parameters, zero optimizer overlap, and zero nonempty
gradients.

## Observed joint RL check

On one 16-device PPU-ZW810E node, the joint recipe selected eight real
trajectories in four pairs (four SAGE, two R2R, two RxR), with 23 tool calls
and 167 action steps. Every learner rank completed one optimizer step for both
the brain and the action expert. The brain and action-expert gradient norms
were 0.422144 and 45.242859. Compared with initialization, the query and
action-head weight changes had L2 norms of 0.0001809604 and 0.005878368.
In a 64-by-256 sample of the exported brain's first-layer QKV input
projection, 110 of 16,384 values changed.

The launcher exited with code zero and the tracker recorded step 1.
The package's checkpoint resolver accepted the joint checkpoint with
`allow_weights_only=False`, including the optimizer, scheduler, and RNG
state. The check ended at this committed step. Multi-step training, a rollout
with the updated policy, and training resumed from this checkpoint are
outside its measured scope.

## Observed navigation and regression checks

Using existing trained weights, one real episode completed on each of SAGE,
R2R, and RxR with two tool calls per episode. All three returned exit code zero
and `evaluation_valid: true`; the overall smoke summary was complete and valid.
The recorded action-step counts were 14, 33, and 17, respectively. All three
terminated at the deliberately small tool budget without navigation success.
These are execution checks, not benchmark accuracy measurements. Evaluation used
the existing trained checkpoint, not the newly initialized one-step SFT checkpoint.

Before the review package was trimmed, the validation environment passed 77 related regression tests, including launcher
configuration, dataset hashes, context archival confinement, native navigation
execution/scoring, and multiprocess case scheduling. The real-SAGE archive test
accepts `SAGE_RUNTIME_TEST_MANIFEST` pointing to a prepared suite's
`manifests/sage-3d.json`; without that external input, that one test is skipped.
It was supplied and passed in this validation.

See [runtime_validation.json](runtime_validation.json) for the machine-readable
summary. The full multi-node RL schedule, full evaluation budgets, NVIDIA
execution and installation from an empty environment remain untested.
