"""DLC 提交共享小工具（closed_loop_3dgs_rl 专用）。

从 verl/dlc/submit_vln_goal_dlc.py 抽出的 4 个 helper + 常量，单点维护，
两个两节点 submit 脚本（submit_bench_3dgs_dlc / submit_fm_grpo_dlc）import 复用，
不再各自抄一遍。逻辑与 submit_vln_goal_dlc.py 内联版完全一致（已跑通）。

用法（见 submit_bench_3dgs_dlc.py / submit_fm_grpo_dlc.py）：
    from _dlc_common import (
        safe_name, shell_quote, get_dlc_client, user_vpc,
        WORKSPACE, PRIORITY, IMAGE, GPUS_PER_NODE, HC_DIR,
    )
"""

from __future__ import annotations

import os
import re

os.environ.setdefault(
    "ALIBABA_CLOUD_CREDENTIALS_URI",
    "http://localhost:7002/api/v1/credentials/0",
)

from alibabacloud_credentials.client import Client as CredClient  # noqa: E402
from alibabacloud_credentials.models import Config as CredConfig  # noqa: E402
from alibabacloud_pai_dlc20201203.client import Client  # noqa: E402
from alibabacloud_pai_dlc20201203.models import (  # noqa: E402
    CreateJobRequestUserVpc,
)
from alibabacloud_tea_openapi.models import Config  # noqa: E402

# ---- 常量（照 submit_vln_goal_dlc.py 现成值）----
WORKSPACE = "vla_xpu"
PRIORITY = 7
IMAGE = ("registry.example.invalid/"
         "aliyun-wl-gpu02/ppu-training:verl-veomni-eval-lite-zhouyang-0518")
GPUS_PER_NODE = 16
# 两 job CPFS 握手目录（AutoVLA 授权盘，两 pod 都挂了 /data/storage 可见）。
HC_DIR = "/data/zerovln/vln_verl/dlc"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def get_dlc_client() -> Client:
    cred_config = CredConfig(type="credentials_uri")
    cred_client = CredClient(cred_config)
    config = Config(credential=cred_client, endpoint="pai-dlc-vpc.cn-wulanchabu.aliyuncs.com")
    return Client(config)


def user_vpc() -> CreateJobRequestUserVpc:
    # 照 submit_vln_goal_dlc.py 同款 VPC + extended_cidrs（10.230.x.x 覆盖两 job pod IP
    # 网段，trainer pod 直连 bench pod IP:port 走这条路，rubric judge 已验证通）。
    return CreateJobRequestUserVpc(
        vpc_id="vpc-0jl6vz0vw0bf48e7v8v2j",
        switch_id="vsw-0jl7b93vkb5938rqipb81",
        security_group_id="sg-0jlbiowdxq51f35s1bpp",
        extended_cidrs=[
            "127.0.0.1/25",
            "127.0.0.1/20",
            "127.0.0.1/20",
            "127.0.0.1/20",
            "127.0.0.1/26",
            "127.0.0.1/26",
            "127.0.0.1/26",
            "127.0.0.1/28",
            "127.0.0.1/28",
            "127.0.0.1/19",
        ],
    )
