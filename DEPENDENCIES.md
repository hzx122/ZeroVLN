# Runtime dependencies

The default GPU entry points were traced through their launchers, service
scripts, and native evaluation/collection paths. Bounded hardware checks are
documented separately in [RUNTIME_VALIDATION.md](RUNTIME_VALIDATION.md); their
scope does not include reproducing the reported navigation experiments.

## Collection and navigation dependencies

The SAGE collection, RL service and evaluation entry points load the ABot
renderer under `third_party/abot_renderer/`. Its CPU path compiles the bundled
C++ compositor locally. The GPU service uses compatible native extensions from
the configured environment or extension seed. R2R and RxR use the external
Habitat navigation environment. The Fine-Grained-R2R notice credits the clause
segmentation adapted by the collection planner in `vln_demo/src/rule_plan_chunks.py`.

The unified evaluation configuration selects ZeroVLN action-expert checkpoints.
Separate NaVILA and DualVLN adapters require the upstream NaVILA, InternNav or
scaling_on_scales sources appropriate to those baselines. The legacy SocialNav
FM driver requires its own model environment. For the standalone CAST augmentation
scripts, provide the four prompt/schema files from upstream CAST revision
`ec7a214e76167e0f844800ea91d6664863a3d9b1` under `third_party/CAST/cast/data/`.
These optional resources are external to this package.

## Required environment and external inputs

- `run_rloo.sh` invokes the included `action_head_rl.trainer`. It requires the
  configured VERL source (`VERL_SOURCE_ROOT`), Megatron Core and Bridge
  (`MEGATRON_ENV`), compatible PyTorch/accelerator packages, Ray, checkpoints,
  manifests, and reachable Gym services. See `REQUIREMENTS.md` and
  `configs/gpu/environment.example.sh`. The reference joint-RL run used VERL
  revision `b213f118f8df230bc97f68cdc56043623c6babdb`. The Megatron path
  imports `accepts_packed_thd_vlm_inputs` and `preprocess_thd_engine` from
  `verl.models.mcore.util`; an installation used for open-loop SFT alone can
  lack these APIs. Provide the published SHA256 for the SAGE source-object map
  and the training scene navmesh directory to both the trainer and Gym services.
  Set `MP3D_HOUSE_ROOT` for the Gym processes to the directory containing
  `<scene>/<scene>.house`. The R2R/RxR information reward reads object
  annotations from these files at episode reset.
- `run_rl_services.sh`, `run_evaluation.sh`, and `run_navigation.sh` use the
  included CPU renderer in `third_party/abot_renderer` and an installed
  navigation environment/data tree (`RYNNBRAIN_NAV_DIR`). Habitat and the
  licensed scene, mesh, navmesh, semantic, and benchmark inputs remain external.
- `run_openloop.sh` requires the training environment, base model/visual weights,
  prepared trajectory data, and initialization inputs specified by its config.
- `run_collection.sh` requires a prepared JSON plan and the plan's model,
  renderer, navigation, and service inputs. It does not synthesize missing data.
Set paths for the target machine; `/workspace`, `/data/zerovln`, and
`/opt/external` are example locations. Legacy platform job submission and
`workspace_config` utilities are not the portable launch path. The root GPU
launchers bypass that platform bootstrap through `GPU_DIRECT_RUN=1` where
applicable. Evaluation and interactive navigation record the revision of the current Git checkout.

## Validation scope

The bounded hardware checks cover one open-loop
training update, one 16-device joint RL update, and navigation on three
benchmarks. See `RUNTIME_VALIDATION.md` and `runtime_validation.json` for
the configurations, measured results and scope of each check.

## Optional data and vendored documentation

The optional ObjectNav catalog and historical derived subset are omitted; supply
an external catalog and scene assets when explicitly selecting that workflow.
The default SAGE/R2R/RxR launchers do not consume those inputs.
