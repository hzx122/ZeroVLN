# Copy to a local file and fill absolute paths on every GPU host, then source it.
# No platform login, cloud image, job ID, shared user venv or Git token is used.
export RL_ARTIFACT_ROOT=/data/closed_loop_3dgs_rl
export GPU_CACHE_ROOT=/tmp/closed-loop-gpu
export PYTHON=/opt/venvs/train/bin/python
export RYNNBRAIN_VENV=/opt/venvs/demo
export VERL_SOURCE_ROOT=/opt/src/verl
export MEGATRON_ENV=/opt/src/megatron-env
# MEGATRON_ENV/mcore and MEGATRON_ENV/bridge/src must contain the installed sources.
export SLOW_MODEL=/data/models/Qwen3.6-35B-A3B
export FAST_BACKBONE=/data/models/Qwen3.5-4B
export INITIAL_CHECKPOINT=/data/checkpoints/openloop-exec4/global_step_010826
export TRAIN_CASES=/data/navigation/train_cases.jsonl
export SAGE_SOURCE_OBJECT_MAP=/data/navigation/source_object_map.json
# Use the SHA256 recorded when the source object map was published.
export SAGE_SOURCE_OBJECT_MAP_SHA256=REPLACE_WITH_PUBLISHED_SHA256
export SAGE_NAVMESH_ROOT=/data/navigation/sage/navmeshes-structural-v1
export SAGE_ROOT=/data/navigation/SAGE-3D
export SAGE_SCENES_ROOT=/data/navigation/sage_gs_root
export SAGE_USD_COLLISION_MASK_ROOT=/data/navigation/sage_usd_rooms_masks
export RYNNBRAIN_NAV_DIR=/opt/src/RynnBrain/navigation
# Install/link your dataset as RYNNBRAIN_NAV_DIR/data on every host.
export PUBLISHED_INPUTS=/data/openloop/inputs/three-source-inputs.json
export PUBLISHED_INDEX=/data/openloop/indexes
export BASE_HEAD_CHECKPOINT=/data/models/action_head_wan22_init_v1
export BASE_QUERY_CHECKPOINT=/data/models/query_initialization
export RAY_ADDRESS=127.0.0.1:6379
# Actual distinct Gym URLs from ALL nodes, in a comma-separated string (default 512).
export BENCH_URLS=''
export SAGE_MANIFEST=/data/navigation/sage/closedloop_subset.json
export SAGE_PROTOCOL_ROOT=/data/navigation/sage
export NAVMESH_ROOT=/data/navigation/navmesh
# Native semantic annotations for R2R/RxR information rewards: <scene>/<scene>.house.
export MP3D_HOUSE_ROOT=/data/navigation/mp3d
export TRAINING_CONFIG=/data/checkpoints/openloop-exec4/experiment_config.json
# PPU: activate your installed SDK before starting services or Ray.
# source /usr/local/PPU_SDK/envsetup.sh cuda
# NVIDIA: activate the matching CUDA runtime and installed training environment.
# Optional custom Mesa installation; otherwise use the system Mesa/EGL packages.
# export MESA_ROOT=/opt/mesa
# Multi-node communication settings follow the actual NIC/transport; not forced by the launcher.
# export NCCL_SOCKET_IFNAME=eth0
