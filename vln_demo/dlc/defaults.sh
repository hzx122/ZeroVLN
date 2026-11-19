#!/usr/bin/env bash
# The sole DLC deployment profile uses all 16 GPUs.
if [[ "${VLN_EXTRA_SERVICES:-1}" != 1 ]]; then
    echo "[defaults] DLC only supports the complete 16-GPU service profile" >&2
    return 2
fi
export VLN_EXTRA_SERVICES=1
export VLN_RENDER_SCENE_CACHE_SIZE=${VLN_RENDER_SCENE_CACHE_SIZE:-128}
# This complete node owns its renderers and models; never inherit another job's URL.
export VLN_HABITAT_RENDER_URL=http://127.0.0.1:7040
export OBJECTNAV_RENDER_URL=http://127.0.0.1:7041
export LEPU_RENDER_URL=http://127.0.0.1:7039
export VLN_SGLANG_35B_URL=http://127.0.0.1:9036/v1
export VLN_SGLANG_27B_URL=http://127.0.0.1:9000/v1
export VLN_SGLANG_122B_URL=http://127.0.0.1:9122/v1
export PI_VLN_URL=http://127.0.0.1:5000
unset VLN_BATCH_MODEL_URL VLN_BATCH_CONTROLLER_MODEL_URL VLN_BATCH_TOKEN VLN_BATCH_OUTPUT
export DEMO_BACKEND=${DEMO_BACKEND:-all}
export DEMO_SPLIT=${DEMO_SPLIT:-val_unseen}
export VLN_REQUIRED_BACKENDS=${VLN_REQUIRED_BACKENDS:-habitat_train,rxr_train_guide,rxr_train_follower,lepu}
if [[ "$VLN_EXTRA_SERVICES" == 1 && ",$VLN_REQUIRED_BACKENDS," != *,lightnav_mujoco,* ]]; then
    export VLN_REQUIRED_BACKENDS="$VLN_REQUIRED_BACKENDS,lightnav_mujoco"
fi
