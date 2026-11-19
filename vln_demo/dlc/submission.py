"""Shared configuration provenance and official-platform I/O for the two submitters."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

CONFIG_BRANCH = "codex/vln-dlc-config"

def git(repo, *args):
    env = dict(os.environ)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        env.pop(key, None)
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
        cwd=repo, env=env, check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    return result.stdout.strip()


def fingerprint(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def load_detail(path):
    envelope = json.loads(path.read_text())
    if not isinstance(envelope, dict):
        raise ValueError("base detail must be the official CLI JSON response")
    response = envelope.get("response")
    http_status = envelope.get("http_status")
    if (type(http_status) is not int or not 200 <= http_status < 300
            or not isinstance(response, dict) or response.get("code") != 2000):
        raise ValueError("base detail is not a successful HTTP/code=2000 response")
    detail = response.get("data")
    if not isinstance(detail, dict):
        raise ValueError("base detail response.data must be a task object")
    return detail


def validate_base(detail, gpus=16, quota=None):
    if detail.get("createdBy") != "local-user-03":
        raise ValueError("base task must belong to local-user-03")
    if detail.get("platType") != 2 or detail.get("podCount") != 1:
        raise ValueError("base task must be Ali phase 2 with exactly one Pod")
    resource = detail.get("resourceConfig")
    if not isinstance(resource, dict):
        raise ValueError("base resourceConfig must be an object")
    gpu_source = resource.get("gpuSource")
    if (not isinstance(gpu_source, dict)
            or type(gpu_source.get("gpu")) is not int
            or gpu_source["gpu"] != gpus):
        raise ValueError("base task must allocate exactly " + str(gpus) + " GPUs per Pod")
    if quota is not None and detail.get("quotaId") != quota:
        raise ValueError("base must use the verified quota")
    for key in ("cpu", "memory"):
        if type(resource.get(key)) is not int or resource[key] <= 0:
            raise ValueError(f"base resourceConfig.{key} must be a positive integer (memory in MB)")
    for key in ("taskId", "quotaId", "imageUrl"):
        if not isinstance(detail.get(key), str) or not detail[key].strip():
            raise ValueError(f"base detail requires nonempty {key}")
    sources = detail.get("dataSources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("base task must have the two CPFS data sources")
    mounts = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("invalid CPFS data source")
        if not isinstance(source.get("dataSourceId"), str) or not source["dataSourceId"]:
            raise ValueError("each CPFS mount needs its authorized dataSourceId")
        mount = source.get("mountPath")
        if not isinstance(mount, str):
            raise ValueError("each CPFS data source needs mountPath")
        mounts.append(mount.rstrip("/"))
    if set(mounts) != {"/data/storage", "/data/storage"}:
        raise ValueError("base task must mount both /data/storage/ and /data/storage/")


def write_json(path, value):
    # Exclusive creation also protects against a concurrent preparation process.
    with path.open("x") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")


def platform_command():
    cli = shutil.which('train-platform')
    return [cli] if cli else [sys.executable, '/opt/external/user-d5121be6/dlc_cli/.agents/skills/train-platform-cli/scripts/train_platform.py']


def platform_call(argv, output):
    """One call only; exclusive receipt prevents accidental reuse after a timeout."""
    with output.open("x") as out, output.with_suffix(".stderr.log").open("x") as err:
        result = subprocess.run([*platform_command(), *argv], stdout=out, stderr=err, timeout=120)
    if result.returncode:
        raise ValueError(f"train-platform {argv[0]} exited {result.returncode}: {output}")
    envelope = json.loads(output.read_text())
    status = envelope.get("http_status")
    response = envelope.get("response", {})
    if (type(status) is not int or not 200 <= status < 300
            or not isinstance(response, dict) or response.get("code") != 2000):
        raise ValueError(f"platform rejected {argv[0]}: inspect {output}")
    return response.get("data")


def create_and_verify(output, request, call=platform_call):
    data = call(["create", "--json-file", str(output / "create.json")],
                         output / "create-response.json")
    tasks = data.get("tasks", [])
    if len(tasks) != 1 or not tasks[0].get("taskId") or not tasks[0].get("externalTaskId"):
        raise ValueError("create receipt does not identify exactly one task/job")
    task_id, job_id = str(tasks[0]["taskId"]), tasks[0]["externalTaskId"]
    receipt = {"submitted": True, "task_id": task_id, "job_id": job_id,
               "name": request["name"], "priority": 8, "module": "mfm"}
    # Persist IDs before readback so a subsequent query failure never hides acceptance.
    write_json(output / "submission-receipt.json", receipt)
    detail = call(["detail", "--task-id", task_id], output / "detail.json")
    for key in ("name", "priority", "module", "podCount", "resourceConfig", "imageUrl", "commitId"):
        if detail.get(key) != request[key]:
            raise ValueError(f"created task readback mismatch: {key}")
    if str(detail.get("taskId")) != task_id or detail.get("jobId") != job_id:
        raise ValueError("created task/job readback mismatch")
    if detail.get("desc", detail.get("description")) != request["desc"]:
        raise ValueError("created task description readback mismatch")
    return {**receipt, "status": detail.get("status"), "output_dir": str(output)}

def validate_paths(base_detail, name, output_dir, repo, mount_roots):
    base_detail, output_dir, repo = map(Path, (base_detail, output_dir, repo))
    if not output_dir.is_absolute() or not repo.is_absolute():
        raise ValueError("--output-dir and --repo must be absolute paths on mounted CPFS")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"output directory already exists; choose a fresh path: {output_dir}")
    output_dir, repo = output_dir.resolve(), repo.resolve()
    if not name.strip() or "\0" in name:
        raise ValueError("--name must be nonempty and cannot contain NUL")
    for path in (repo, output_dir):
        if not any(path.is_relative_to(root) for root in mount_roots):
            raise ValueError(f"path is outside mounted CPFS roots: {path}")
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("--repo must be the repository root")
    return base_detail, output_dir, repo


def provenance(base_detail, detail, repo, source_files):
    return {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "base_task_id": detail["taskId"], "base_job_id": detail.get("jobId"),
        "base_detail_file": str(base_detail.resolve()),
        "base_detail_fingerprint": fingerprint(base_detail),
        "repo": str(repo), "baseline_head": git(repo, "rev-parse", "HEAD"),
        "baseline_branch": git(repo, "branch", "--show-current"),
        "source_sha256": {relative: fingerprint(repo / relative) for relative in source_files},
        "version_scope": "The task runs the shared worktree; this records configuration and selected source hashes, not an immutable checkout of all dependencies.",
    }


def record_configuration(output_dir, request, source_manifest, configs=None):
    output_dir.mkdir(parents=True, exist_ok=False)
    config_dir = output_dir / "config-version"
    config_dir.mkdir()
    files = {"launch-config.json": request, "source-manifest.json": source_manifest, **(configs or {})}
    for name, content in files.items():
        write_json(config_dir / name, content)
    git(config_dir, "init", "--initial-branch=" + CONFIG_BRANCH)
    git(config_dir, "add", "--", *files)
    git(config_dir, "-c", "user.name=VLN DLC Config", "-c", "user.email=vln-dlc@localhost",
        "commit", "-m", "Record VLN DLC launch configuration")
    request["commitId"] = git(config_dir, "rev-parse", "HEAD")
    write_json(output_dir / ("swap.json" if "stopTaskId" in request else "create.json"), request)
    return request
