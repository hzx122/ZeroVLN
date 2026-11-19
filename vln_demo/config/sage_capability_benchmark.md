# SAGE-3D capability benchmark v2

The benchmark is non-canonical and never writes `/data/zerovln/SAGE-3D/SAGE-3D_VLN_Data/demo/Trajectory_demo`.

## v3-dev construction contract

`sage-capability-v3-dev` is an opt-in manifest schema; frozen v2 manifests remain immutable. Every v3-dev case declares `evaluation_contract.primary_success=semantic_terminal_region_v1`, and every family reports the same success rate: the agent must explicitly stop inside the correct frozen semantic terminal region. Legacy SPL and capability-specific trajectory observables are diagnostics only. A model failure remains in the denominator; only infrastructure/runtime errors, unavailable semantic geometry, ambiguous overlapping terminal regions, or an invalid Recovery intervention can make evaluation invalid.

Instruction generation is per case family rather than per source trajectory. All five families consume the existing adaptive route frames, but their claim contracts differ: Perception requires visible target properties without route landmarks; Grounding requires a decision-relevant visual/spatial contrast claim; Planning and Recovery require all audited landmarks and turns in order; Execution requires the short local turn sequence and visually supported approach claims. Every instruction must contain the target label and an explicit `stop` command. Use `--require-vlm-instructions` for construction/evaluation freezing so a rejected Qwen instruction cannot silently fall back to the source template. Rejected coverage rows retain the complete generation and verification attempts.

For visualization, pass the same `--console-output` directory to every family build and use `--console-append` after the first build. This materializes one GVLN dataset with dimension-tagged trajectory ids and instruction types. Load it only through `SAGE_SPLIT=capability` plus `SAGE_CAPABILITY_VLN_ROOT=<absolute Trajectory_demo path>`; do not start separate Perception/Grounding/Planning/Recovery/Execution consoles. The current unified visualization artifact is `/workspace/ZeroVLN/tmp/capability_v3_unified/Trajectory_demo/`.

## Construction/evaluation scene holdout

`/workspace/ZeroVLN/vln_demo/src/sage/capability_scene_split.py` hashes every physically reachable train scene with the fixed salt `sage-capability-scene-holdout-v1`: 80% construction, 20% evaluation. The partitions are deterministic and disjoint; never choose scene IDs manually.

```bash
cd /workspace/ZeroVLN
CAPABILITY_PARTITION=construction CAPABILITY_SCENE_LIMIT=8 CAPABILITY_EPISODES_PER_SCENE=1 \
  bash /workspace/ZeroVLN/vln_demo/build_capability_profiles.sh perception
CAPABILITY_PARTITION=evaluation CAPABILITY_SCENE_LIMIT=8 GROUNDING_EPISODES_PER_SCENE=6 \
  bash /workspace/ZeroVLN/vln_demo/build_capability_profiles.sh grounding
```

Sparse profiles set `runtime.require_all_scenes=false`: an individual scene is committed only when it produces the complete requested count. No threshold is relaxed when a scene fails.

## Manifest and scoring

```bash
cd /workspace/ZeroVLN/vln_demo/src
PYTHONPATH=. /opt/navigation-runtime/venv/bin/python \
  -m sage.build_capability_cases \
  --source /data/zerovln/capability_benchmark/sources/evaluation/grounding/Trajectory_demo \
  --output /data/zerovln/capability_benchmark/evaluation_grounding_v2
```

Before online evaluation, start render `7038`, Qwen3.6 `28899`, and bench `8907` with the matching `SAGE_SPLIT=cap_eval_<dimension>`. Then run `/workspace/ZeroVLN/vln_demo/run_capability_benchmark.sh` with `CAPABILITY_MANIFEST` pointing at the manifest.

Historical rollout artifacts can be rescored without contacting bench/model:

```bash
PYTHONPATH=/workspace/ZeroVLN/vln_demo/src python3 \
  -m verl_access.capability_eval --rescore-run /absolute/run/directory
```

This writes `summary.rescored.json` and never overwrites the historical summary. Runtime errors, invalid/blocked Recovery interventions, and Execution traces with no macro-correct action stay outside the valid denominator.

## Model-visible observation contract

The production default is `agent_v1`. It reuses `/workspace/ZeroVLN/vln_demo/src/agent_tools.py::move_observation/turn_observation`: the model sees requested/actual relative motion, blocked state, accumulated relative mileage/yaw, step count, and the new RGB. `distance_to_goal`, absolute `gps/world_xy`, and absolute `yaw_deg/world_yaw_deg` remain in evaluator-only rollout info and must never be interpolated into a model message.

`/workspace/ZeroVLN/vln_demo/src/vln_gym/env.py` returns the existing `agent_tools.drive_*` execution dict as `info.tool_result`; no second motion implementation is allowed. The pre-fix contract is retained only for an explicit diagnostic A/B:

```bash
bash /workspace/ZeroVLN/vln_demo/run_capability_benchmark.sh \
  --observation-contract agent_v1
# Historical leakage baseline only; never use as the production default:
bash /workspace/ZeroVLN/vln_demo/run_capability_benchmark.sh \
  --observation-contract legacy_oracle_v0
```

Every rollout and summary records `observation_contract`. Regression tests are in `/workspace/ZeroVLN/vln_demo/src/verl_access/test_observation_contract.py`.

## General evidence contracts

- Perception: 2–5m, at most one turn/door, target verified early, no same-family distractor within 6m panoramic 2-D LOS.
- Grounding: exactly shared start, two distinct target terms, balanced path length/ratio/turn/door difficulty, separated goal points, both targets independently pass ordinary two-view RGB/depth verification.
- Planning: 8–15m, at least three turn/door decisions, at least one ordered landmark passing topology + camera-frustum + metric-depth evidence.
- Recovery: 8–15m multi-decision route; from the actual trigger pose, the perturbation must be collision-free, create a measurable corridor departure, and retain a short A* return path. Landmark evidence is intentionally not required.
- Execution: 2–6m, one semantic corner, low clearance, verified target; a rollout is valid only if at least one macro-correct motion was observed.

No contract may branch on scene ID, trajectory ID, object instance ID, or one-off failure cases.
