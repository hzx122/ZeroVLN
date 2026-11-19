#!/usr/bin/env python3
"""唯一 DLC 服务提交入口：准备完整 16 卡配置，--submit 普通提交并跟进验收。"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objectnav.categories import BACKEND_CATEGORIES, expected_counts

NEXT_INTEGRITY = Path(
    "/opt/external/user-d5121be6/model/flagos-qwen38-public/runtime/"
    "config/rootfs-integrity.json")
SOURCE_FILES = (
    "vln_demo/dlc/entry.sh", "vln_demo/dlc/attempt.sh", "vln_demo/dlc/defaults.sh",
    "vln_demo/dlc/prepare_pi_tools.sh",
    "vln_demo/dlc/preflight.py", "vln_demo/dlc/supervise.py",
    "vln_demo/dlc/submit.py", "vln_demo/dlc/restart_console.py", "vln_demo/dlc/submission.py", "vln_demo/start_all.sh",
    "vln_demo/vllm/vllm38next", "vln_demo/vllm/launch_next.py",
    "vln_demo/vllm/next_prefix_cache.py", "vln_demo/vllm/vllm-vln-35B.sh",
    "vln_demo/src/run_console.sh", "vln_demo/src/vln_model.py",
    "vln_demo/src/habitat_topdown.py", "vln_demo/src/vln_geometry.py",
    "vln_demo/src/sage/env.py",
    "vln_demo/src/abotn/env.py",
    "vln_demo/src/hsim/env.py",
    "vln_demo/src/rs3d/env.py",
    "vln_demo/src/lightnav_mujoco/env.py",
    "vln_demo/src/vln_gym/env.py",
    "vln_demo/src/vln_gym/server.py", "verl/action_head/render_client.py", "verl/action_head/agent_runtime.py",
    "vln_demo/src/terminal_workspace.py",
    "vln_demo/src/web/js/store.js", "vln_demo/src/web/js/panels/log.js",
    "vln_demo/src/web/js/panels/instr.js", "vln_demo/src/web/js/panels/terminal.js",
    "vln_demo/src/vln_demo.yaml", "vln_demo/src/agent_tools.py", "vln_demo/src/visual_navigation_tools.py",
    "vln_demo/src/remote_habitat/server.py", "vln_demo/src/remote_habitat/client.py",
    "vln_demo/src/remote_habitat/start.sh",
    "vln_demo/src/remote_habitat/policy.py", "vln_demo/src/remote_habitat/worker.py",
    "vln_demo/src/vlnce_training.py", "vln_demo/src/rxr.py",
    "vln_demo/src/lepu/assets.py", "vln_demo/src/lepu/env.py", "vln_demo/src/lepu/floorplan.png",
    "vln_demo/src/mall/assets.py", "vln_demo/src/mall/env.py",
    "vln_demo/src/visual_navigation/planning.py",
    "vln_demo/src/webui.py",
    "vln_demo/src/prompts.py", "vln_demo/dlc/start_extra.sh", "vln_demo/dlc/lightnav_probe.py",
    "vln_demo/vllm/sglang-qwen3.8-27B-tp1.sh",
    "vln_demo/config/pi/project-models.ts",
)
REQUIRED_BACKENDS = "habitat_train,rxr_train_guide,rxr_train_follower,lepu"
TASK_NAME = "VLM+渲染+控制台服务部署"
TASK_DESCRIPTION = (
    "单机 16 卡部署 VLM 推理、4 路 3DGS 渲染、独立 Habitat 渲染与网页控制台（含 pi-agent）服务，"
    "配套 LightNav 导航策略及 MuJoCo 仿真，加载全部已注册仿真资产与训练 split。"
    "模型配置：Next TP8、35B TP2；GPU15 部署 Habitat-gs RGB-D 渲染；挂载 CPFS 持久化会话与交互产物。"
)
MOUNT_ROOTS = (Path("/data/storage"), Path("/data/storage"))


from submission import (CONFIG_BRANCH, git, fingerprint, load_detail, validate_base, write_json,
                        platform_call, create_and_verify, validate_paths, provenance, record_configuration)


def prepare(base_detail, name, output_dir, repo, *, swap_approvers=None, reason=None, objectnav_manifest=None, agent_profile="next", workflow_plan=None):
    base_detail, output_dir, repo = validate_paths(base_detail, name, output_dir, repo, MOUNT_ROOTS)
    detail = load_detail(base_detail)
    validate_base(detail)
    if swap_approvers is not None:
        if detail.get("status") != "Running":
            raise ValueError("swap requires the old task to be Running")
        if not detail.get("spaceId") or detail["spaceId"] == "*":
            raise ValueError("swap requires the old task's explicit spaceId")
        if not swap_approvers or any(not isinstance(a, str) or not a.strip() for a in swap_approvers):
            raise ValueError("swap requires approvers from the space whitelist")
        if not reason or not reason.strip():
            raise ValueError("swap requires a reason")
    if objectnav_manifest is None:
        config = json.loads((repo / "vln_demo/config/objectnav.json").read_text())
        if not config.get("manifest"):
            raise ValueError("ObjectNav data is not bundled; provide --objectnav-manifest with an external normalized catalog")
        objectnav_manifest = Path(config["manifest"])
        if not objectnav_manifest.is_absolute():
            objectnav_manifest = repo / objectnav_manifest
    objectnav_manifest = Path(objectnav_manifest).resolve(strict=True)
    import gzip
    opener = gzip.open if objectnav_manifest.suffix == '.gz' else open
    with opener(objectnav_manifest, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(row.get("dataset") != "objectnav" for row in rows):
        raise ValueError("normalized ObjectNav catalog required")
    category_counts = expected_counts(rows)
    for row in rows:
        for key in ("asset_path", "navmesh"):
            path = Path(row[key]).resolve(strict=True)
            if not any(path.is_relative_to(root) for root in MOUNT_ROOTS):
                raise ValueError("ObjectNav asset must be on DLC-mounted CPFS: " + str(path))
    if agent_profile != "next":
        from model_service import model_profile_files
        model_profile_files(repo, agent_profile)  # validate before creating anything
    source_manifest = provenance(base_detail, detail, repo, (*SOURCE_FILES, *(str(p.relative_to(repo)) for p in (repo/"vln_demo/src/objectnav").glob("*") if p.suffix in (".py", ".sh"))))
    if agent_profile != "next":
        source_manifest["source_sha256"].update({p: fingerprint(repo/p) for p in model_profile_files(repo, agent_profile)})
    source_manifest["next_rootfs_integrity"] = {"path": str(NEXT_INTEGRITY), **fingerprint(NEXT_INTEGRITY)}
    source_manifest["objectnav_manifest"] = {"path": str(objectnav_manifest), **fingerprint(objectnav_manifest)}
    source_manifest["gary_categories"] = category_counts
    if workflow_plan:
        files = [p for p in (repo / "vln_demo/batch").rglob("*")
                 if p.is_file() and p.suffix in (".py", ".mjs", ".json", ".jsonl", ".md")]
        source_manifest["source_sha256"].update({str(p.relative_to(repo)): fingerprint(p) for p in files})
        source_manifest["evaluation_plan"] = json.loads(Path(workflow_plan).read_text())
    run = output_dir / "run"
    config_dir = output_dir / "config-version"
    command = "\n".join((
        "set -euo pipefail",
        "export VLN_WORKFLOW_PLAN=" + shlex.quote(str(workflow_plan or '')),
        "export VLN_AGENT_MODEL_PROFILE=" + shlex.quote(agent_profile),
        "export VLN_RUN_DIR=" + shlex.quote(str(run)),
        "export OBJECTNAV_MANIFEST=" + shlex.quote(str(objectnav_manifest)),
        "export DEMO_TERMINAL_CWD=" + shlex.quote(str(run / "manual-workspace")),
        "export DEMO_BACKEND=all",
        "export DEMO_SPLIT=val_unseen",
        "export VLN_REQUIRED_BACKENDS=" + shlex.quote(
            REQUIRED_BACKENDS + ",lightnav_mujoco," + ",".join(BACKEND_CATEGORIES)),
        "export VLN_EXTRA_SERVICES=1",
        "exec bash " + shlex.quote(str(repo / "vln_demo/dlc/entry.sh")), ""))
    request = {
        "phaseCode": 14, "module": "mfm", "platType": 2,
        "quotaId": detail["quotaId"], "resourceConfig": deepcopy(detail["resourceConfig"]),
        "imageUrl": detail["imageUrl"], "preemptible": detail.get("preemptible", False),
        "imageCategory": detail.get("imageCategory") or "Input",
        "imageId": detail.get("imageId") or "", "name": name,
        "desc": TASK_DESCRIPTION if agent_profile == "next" else TASK_DESCRIPTION.replace("Next TP8", agent_profile + " SGLang TP8"),
        "podCount": 1, "priority": 8, "operatedBy": "local-user-03",
        "configPath": str(config_dir / "launch-config.json"), "workDir": str(repo),
        "parsedProductDir": str(run), "userCommand": command, "downloadCode": False,
        "parentModels": [], "dataSources": deepcopy(detail["dataSources"]),
        "branch": CONFIG_BRANCH,
    }
    if swap_approvers is not None:
        request.update(stopTaskId=detail["taskId"], spaceId=str(detail["spaceId"]),
                       approvers=list(dict.fromkeys(swap_approvers)), reason=reason)
    return record_configuration(output_dir, request, source_manifest)


def submit_created(output, request, repo):
    receipt = create_and_verify(output, request, call=platform_call)
    task_id, job_id = receipt["task_id"], receipt["job_id"]
    follow = output / "follow"
    command = [sys.executable, "-u", str(repo / "vln_demo/dlc/follow_submission.py"),
               "--task-id", task_id, "--job-id", job_id,
               "--run-dir", str(output / "run"), "--output-dir", str(follow), "--interval", "120"]
    with (output / "follower.log").open("x") as log:
        process = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(output / "follower-process.json", {"pid": process.pid, "argv": command})
    return {**receipt, "status": receipt["status"], "follower_pid": process.pid,
            "output_dir": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-profile', choices=('qwen36-35b', 'qwen38-27b', 'qwen35-122b'),
                        help='完整16卡服务的主模型：指定则用SGLang TP8占GPU4–11，省略则用Next')
    parser.add_argument("--base-detail", required=True, type=Path,
                        help="successful official train-platform detail JSON response")
    parser.add_argument("--name", default=TASK_NAME + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                        help="默认用途名称加 UTC 时间戳，不按资产或开发分支命名")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="fresh absolute CPFS directory; task outputs go into its run/ child")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--objectnav-manifest", type=Path, help="external normalized CPFS ObjectNav catalog; not bundled")
    parser.add_argument("--submit", action="store_true",
                        help="准备后普通 create、回查并启动验收跟进；省略则只生成配置")
    parser.add_argument("--swap", action="store_true", help="prepare platform swap approval instead of create")
    parser.add_argument("--approver", action="append", help="verified AD account from the space whitelist")
    parser.add_argument("--reason", help="swap approval reason")
    args = parser.parse_args(argv)
    try:
        if not args.swap and (args.approver or args.reason):
            raise ValueError("--approver and --reason require --swap")
        if args.swap and args.submit:
            raise ValueError("--swap 仅准备审批材料；按训练平台 skill 确认本次审批方案后提交")
        request = prepare(args.base_detail, args.name, args.output_dir, args.repo,
                          swap_approvers=(args.approver or []) if args.swap else None, reason=args.reason, objectnav_manifest=args.objectnav_manifest, agent_profile=args.model_profile or "next")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[prepare] failed: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        return 1
    if args.submit:
        try:
            print(json.dumps(submit_created(args.output_dir.resolve(), request, args.repo.resolve()),
                             ensure_ascii=False, indent=2))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f"[submit] {exc}; 请求与回执已保留，先查询平台，勿盲目重提", file=sys.stderr)
            return 1
        return 0
    filename = "swap.json" if args.swap else "create.json"
    print(json.dumps({"request_json": str(args.output_dir.resolve() / filename),
                      "config_commit": request["commitId"], "priority": 8,
                      "module": "mfm", "submitted": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit("DLC submission entry removed in the open-source edition; use the root GPU run scripts.")
