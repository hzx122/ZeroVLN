#!/usr/bin/env python3
"""Follow one submitted DLC job, accept it, then optionally publish a fixed Git plan.

Only ``train-platform detail`` is used. This tool never creates/stops a task,
changes its priority, or sends notifications. By default it performs no Git
writes. A commit or fast-forward merge plan and its independently supplied
SHA-256 are required to publish after real runtime acceptance. Merge mode never
checks out a branch or changes the index or worktree.
Run under a persistent local process manager; inspect output-dir/status.json.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


TERMINAL_PLATFORM = {"failed", "stopped", "cancelled", "canceled", "deleted",
                     "succeeded", "completed", "terminated", "error"}
TERMINAL_FOLLOW = {"complete", "failed", "needs_attention"}
REQUIRED_RUNTIME_SOURCES = ("vln_demo/dlc/entry.sh", "vln_demo/dlc/attempt.sh",
                            "vln_demo/dlc/supervise.py", "vln_demo/dlc/preflight.py",
                            "vln_demo/start_all.sh", "vln_demo/vllm/vllm-vln-35B.sh")


class NeedsAttention(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise NeedsAttention(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def command(argv, *, cwd=None, timeout=120):
    env = dict(os.environ)
    if argv[0] == "git":
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
            env.pop(key, None)
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                            timeout=timeout, check=False, env=env)
    require(result.returncode == 0,
            f"{argv[0]} {argv[1]} exited {result.returncode}: "
            + (result.stderr or result.stdout)[-3000:])
    return result.stdout


def task_detail(payload, task_id, job_id):
    require(200 <= payload.get("http_status", 0) < 300
            and payload.get("response", {}).get("code") == 2000,
            "official CLI detail request did not succeed")
    detail = payload["response"]["data"]
    require(isinstance(detail, dict), "detail data is not an object")
    require(str(detail.get("taskId")) == str(task_id), "detail taskId changed")
    require(detail.get("jobId") == job_id, "detail jobId changed")
    require(isinstance(detail.get("status"), str), "detail has no task status")
    return detail


def read_ready(run_root, job_id):
    """Return ready status only for a fresh attempt of this exact job/root."""
    path = run_root / "status.json"
    if not path.exists():
        return None
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(run_root), "runtime status escapes requested run root")
    status = json.loads(resolved.read_text())
    if status.get("state") != "ready":
        return None  # A failed attempt is not a terminal platform job.
    hostname = status.get("hostname", "")
    require(hostname == job_id or hostname.startswith(job_id + "-"),
            "ready status belongs to another DLC job")
    require(Path(status.get("run_root", "")).resolve() == run_root,
            "ready status belongs to another run root")
    stamp = datetime.fromisoformat(status["updated_at"].replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "runtime timestamp has no timezone")
    if not -10 <= time.time() - stamp.timestamp() <= 120:
        return None
    return status


def verify_receipt(path, job_id, run_root):
    receipt = json.loads(Path(path).read_text())
    require(receipt.get("passed") is True and receipt.get("state") == "passed",
            "runtime acceptance did not pass")
    require(receipt.get("expected_job_id") == job_id, "acceptance job ID does not match")
    require(Path(receipt.get("run_root", "")).resolve() == Path(run_root).resolve(),
            "acceptance run root does not match")
    checks = receipt.get("checks")
    require(isinstance(checks, list) and checks
            and all(check.get("passed") is True for check in checks),
            "acceptance contains absent or failed checks")
    return receipt


def relative_path(value):
    path = Path(value)
    require(not path.is_absolute() and value not in ("", ".")
            and ".." not in path.parts and str(path) == value,
            f"unsafe repository-relative path: {value!r}")
    return path


def load_plan(path, expected_hash):
    require(re.fullmatch(r"[0-9a-f]{64}", expected_hash or "") is not None,
            "a fixed --commit-plan-sha256 is required")
    require(sha256(path) == expected_hash, "commit plan changed")
    plan = json.loads(Path(path).read_text())
    repo = Path(plan["repo"])
    require(repo.is_absolute(), "plan repo must be absolute")
    require(plan["remote"] == "origin", "plan remote must be origin")
    require(re.fullmatch(r"[0-9a-f]{40,64}", plan["baseline_head"]) is not None,
            "plan baseline_head must be a full Git object ID")
    require(isinstance(plan["message"], str) and plan["message"].strip(), "missing commit message")
    require(isinstance(plan["files"], dict) and plan["files"], "plan has no explicit files")
    for name, digest in plan["files"].items():
        target = repo / relative_path(name)
        require(not target.is_symlink() and target.resolve().is_relative_to(repo.resolve()),
                f"planned source is a symlink or outside repo: {name}")
        require(sha256(target) == digest, f"planned source changed: {name}")
    patch = plan.get("readme_patch")
    if patch:
        relative_path(patch["target"])
        require(Path(patch["target"]).name == "README.md", "patch is not for a README.md")
        require(patch["target"] not in plan["files"], "README patch target also listed as a whole file")
        require(Path(patch["path"]).is_absolute(), "README patch path must be absolute")
        require(sha256(patch["path"]) == patch["sha256"], "README patch changed")
    return plan


def load_merge_plan(path, expected_hash):
    require(Path(path).is_absolute(), "--merge-plan must be absolute")
    require(re.fullmatch(r"[0-9a-f]{64}", expected_hash or "") is not None,
            "a fixed --merge-plan-sha256 is required")
    require(sha256(path) == expected_hash, "merge plan changed")
    plan = json.loads(Path(path).read_text())
    repo = Path(plan["repo"])
    require(repo.is_absolute(), "merge plan repo must be absolute")
    require(plan["remote"] == "origin", "merge plan remote must be origin")
    for field in ("target_head", "source_commit"):
        require(re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", plan[field]) is not None,
                f"merge plan {field} must be a full Git object ID")
    for field in ("source_branch", "target_branch"):
        require(isinstance(plan[field], str) and plan[field] and not plan[field].startswith("-"),
                f"merge plan {field} is invalid")
    require(plan["source_branch"] != plan["target_branch"], "merge branches must differ")
    files = plan["files"]
    require(isinstance(files, dict) and set(REQUIRED_RUNTIME_SOURCES) <= set(files),
            "merge plan is missing required launcher files")
    for name, digest in files.items():
        target = repo / relative_path(name)
        require(not target.is_symlink() and target.resolve().is_relative_to(repo.resolve()),
                f"planned source is a symlink or outside repo: {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"invalid source SHA-256: {name}")
        require(sha256(target) == digest, f"planned source changed: {name}")
    return plan


def verify_runtime_sources(plan, run_root, attempt_dir, job_id):
    """Bind this attempt's preflight fingerprints to the still-present source."""
    root, attempt, repo = Path(run_root).resolve(), Path(attempt_dir).resolve(), Path(plan["repo"]).resolve()
    require(attempt.is_relative_to(root), "preflight attempt escapes this run root")
    path = (attempt / "preflight.json").resolve(strict=True)
    require(path.parent == attempt, "preflight pointer escapes its attempt")
    preflight = json.loads(path.read_text())
    hostname = preflight.get("hostname", "")
    require(preflight.get("passed") is True and
            (hostname == job_id or hostname.startswith(job_id + "-")),
            "preflight is not a passed receipt for this job")
    sources = preflight.get("source_sha256")
    require(isinstance(sources, dict) and sources, "preflight has no source SHA-256 mapping")
    require(all(str(repo / name) in sources for name in REQUIRED_RUNTIME_SOURCES),
            "preflight is missing required launcher fingerprints")
    for source, digest in sources.items():
        source_path = Path(source)
        require(source_path.is_absolute() and source_path.resolve().is_relative_to(repo),
                f"preflight source is outside the repository: {source}")
        require(sha256(source_path) == digest, f"runtime source changed since preflight: {source}")
    for name, digest in plan["files"].items():
        if str(repo / name) in sources:
            require(sources[str(repo / name)] == digest,
                    f"runtime source does not match commit plan: {name}")
    return {"path": str(path), "sha256": sha256(path), "source_sha256": sources}


def commit_verified(plan_path, plan_hash, receipt_path, job_id, run_root, checkpoint):
    """No implicit staging, reset, checkout, pull, rebase, force push, or retries."""
    receipt = verify_receipt(receipt_path, job_id, run_root)
    receipt_hash = sha256(receipt_path)
    plan = load_plan(plan_path, plan_hash)
    runtime = verify_runtime_sources(plan, run_root, receipt["attempt_dir"], job_id)
    repo, baseline, branch = Path(plan["repo"]), plan["baseline_head"], plan["branch"]

    def git(*args):
        return command(["git", *args], cwd=repo).strip()

    def remote_head():
        rows = git("ls-remote", "--heads", "origin", "refs/heads/" + branch).splitlines()
        require(len(rows) == 1, "remote branch missing or ambiguous")
        value, ref = rows[0].split()
        require(ref == "refs/heads/" + branch, "remote ref does not match branch")
        return value

    def unchanged():
        load_plan(plan_path, plan_hash)
        require(git("rev-parse", "HEAD") == baseline, "local HEAD changed")
        require(git("branch", "--show-current") == branch, "local branch changed")

    def blob_hash(revision, name):
        env = dict(os.environ)
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
            env.pop(key, None)
        blob = subprocess.run(["git", "show", revision + ":" + name], cwd=repo,
                              check=True, capture_output=True, env=env).stdout
        return hashlib.sha256(blob).hexdigest()

    unchanged()
    require(Path(git("rev-parse", "--show-toplevel")).resolve() == repo.resolve(),
            "plan repo is not the Git worktree root")
    require(not git("diff", "--cached", "--name-only"), "Git index already has staged changes")
    require(remote_head() == baseline, "remote HEAD changed")
    allowed = set(plan["files"])
    patch = plan.get("readme_patch")
    if patch:
        paths = [line.split("\t", 2)[-1] for line in
                 git("apply", "--numstat", "--", patch["path"]).splitlines()]
        require(paths == [patch["target"]], "README patch modifies unplanned paths")
        git("apply", "--cached", "--check", "--", patch["path"])
        allowed.add(patch["target"])
        # Materialize only the baseline README in a temporary directory. This
        # computes baseline+patch independently of the user's worktree/index.
        with tempfile.TemporaryDirectory(prefix="vln-readme-plan-") as temporary:
            temp_root = Path(temporary)
            target = temp_root / patch["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
                env.pop(key, None)
            baseline_readme = subprocess.run(["git", "show", baseline + ":" + patch["target"]],
                                             cwd=repo, env=env, check=True, capture_output=True).stdout
            target.write_bytes(baseline_readme)
            command(["git", "apply", "--", patch["path"]], cwd=temp_root)
            expected_readme_hash = sha256(target)
    # All read-only guards pass before the first index mutation.
    unchanged()
    require(not git("diff", "--cached", "--name-only"), "Git index changed before staging")
    checkpoint(phase="staging", planned_paths=sorted(allowed))
    git("add", "--", *plan["files"])
    if patch:
        git("apply", "--cached", "--", patch["path"])
    staged = set(git("diff", "--cached", "--name-only", "--no-renames", "-z").split("\0")) - {""}
    require(staged and staged <= allowed, "staged paths do not match the explicit plan")
    # Verify staged bytes too, so concurrent edits cannot substitute untested files.
    for name in plan["files"]:
        require(blob_hash("", name) == plan["files"][name],
                f"staged source differs from tested source: {name}")
    staged_blobs = {name: blob_hash("", name) for name in staged}
    if patch:
        require(blob_hash("", patch["target"]) == expected_readme_hash,
                "cached README differs from baseline plus the authorized patch")
    unchanged()
    require(remote_head() == baseline, "remote HEAD changed before commit")
    require(sha256(receipt_path) == receipt_hash, "acceptance receipt changed")
    verify_receipt(receipt_path, job_id, run_root)
    require(verify_runtime_sources(plan, run_root, receipt["attempt_dir"], job_id) == runtime,
            "runtime preflight changed during Git preparation")
    checkpoint(phase="committing", staged_paths=sorted(staged))
    git("commit", "-m", plan["message"])
    commit = git("rev-parse", "HEAD")
    checkpoint(phase="committed", git_commit=commit)
    require(git("rev-parse", commit + "^") == baseline, "commit parent differs from baseline")
    committed = set(git("diff-tree", "--no-commit-id", "--name-only", "--no-renames",
                        "-r", "-z", commit).split("\0")) - {""}
    require(committed == staged, "commit contains unexpected paths")
    require(all(blob_hash(commit, name) == digest for name, digest in staged_blobs.items()),
            "committed contents changed after staging; local commit retained")
    load_plan(plan_path, plan_hash)
    require(git("branch", "--show-current") == branch, "branch changed during commit")
    require(not git("diff", "--cached", "--name-only"), "index changed during commit")
    require(remote_head() == baseline, "remote HEAD changed before push; local commit retained")
    require(git("rev-parse", "HEAD") == commit, "HEAD changed before push; local commit retained")
    require(sha256(receipt_path) == receipt_hash, "acceptance receipt changed before push")
    verify_receipt(receipt_path, job_id, run_root)
    checkpoint(phase="pushing", git_commit=commit)
    git("push", "origin", commit + ":refs/heads/" + branch)
    require(remote_head() == commit, "push completed but remote HEAD verification failed")
    require(git("rev-parse", "HEAD") == commit, "push verified, but local HEAD changed concurrently")
    checkpoint(phase="published", git_commit=commit, remote_head=commit)
    return commit


def merge_verified(plan_path, plan_hash, receipt_path, job_id, run_root, checkpoint):
    """Publish exactly the accepted source commit by fast-forward, without checkout.

    A failed push leaves the local target ref at the accepted source commit and
    records that checkpoint. Nothing is rolled back or retried automatically.
    """
    receipt = verify_receipt(receipt_path, job_id, run_root)
    receipt_hash = sha256(receipt_path)
    plan = load_merge_plan(plan_path, plan_hash)
    runtime = verify_runtime_sources(plan, run_root, receipt["attempt_dir"], job_id)
    repo = Path(plan["repo"])
    source, baseline = plan["source_commit"], plan["target_head"]
    source_ref, target_ref = ("refs/heads/" + plan[key]
                              for key in ("source_branch", "target_branch"))

    def git(*args):
        return command(["git", *args], cwd=repo).strip()

    for ref in (source_ref, target_ref):
        git("check-ref-format", ref)
    require(Path(git("rev-parse", "--show-toplevel")).resolve() == repo.resolve(),
            "merge plan repo is not the Git worktree root")
    git("merge-base", "--is-ancestor", baseline, source)
    for name, digest in plan["files"].items():
        env = dict(os.environ)
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
            env.pop(key, None)
        blob = subprocess.run(["git", "show", source + ":" + name], cwd=repo,
                              capture_output=True, check=True, env=env).stdout
        require(hashlib.sha256(blob).hexdigest() == digest,
                f"source commit differs from tested source: {name}")

    def remote_refs(expected_target):
        rows = git("ls-remote", "--heads", "origin", source_ref, target_ref).splitlines()
        refs = dict((ref, value) for value, ref in (row.split() for row in rows))
        require(len(rows) == 2 and refs.get(source_ref) == source,
                "remote source branch changed or is missing")
        require(refs.get(target_ref) == expected_target, "remote target branch changed")

    def unchanged(expected_target, expected_remote):
        load_merge_plan(plan_path, plan_hash)
        require(git("symbolic-ref", "HEAD") == source_ref, "local source branch changed")
        require(git("rev-parse", "HEAD") == source, "local source HEAD changed")
        require(git("rev-parse", "--verify", target_ref) == expected_target,
                "local target branch changed")
        require(not git("diff", "--cached", "--name-only"), "Git index has staged changes")
        require("branch " + target_ref not in git("worktree", "list", "--porcelain").splitlines(),
                "target branch is checked out in another worktree")
        require(sha256(receipt_path) == receipt_hash, "acceptance receipt changed")
        verify_receipt(receipt_path, job_id, run_root)
        require(verify_runtime_sources(plan, run_root, receipt["attempt_dir"], job_id) == runtime,
                "runtime preflight changed during merge preparation")
        remote_refs(expected_remote)

    unchanged(baseline, baseline)
    checkpoint(phase="fast_forwarding", git_commit=source, target_branch=plan["target_branch"],
               target_before=baseline)
    unchanged(baseline, baseline)
    git("update-ref", "-m", "DLC runtime acceptance fast-forward", target_ref, source, baseline)
    checkpoint(phase="target_ref_updated", git_commit=source, local_target_head=source)
    unchanged(source, baseline)
    checkpoint(phase="pushing_merge", git_commit=source)
    unchanged(source, baseline)
    git("push", "origin", source + ":" + target_ref)
    unchanged(source, source)
    checkpoint(phase="merged", git_commit=source, remote_head=source,
               target_branch=plan["target_branch"], source_branch=plan["source_branch"])
    return source


class Follower:
    def __init__(self, args):
        self.args = args
        # Preserve programmatic callers of the original commit-only interface.
        self.args.merge_plan = getattr(args, "merge_plan", None)
        self.args.merge_plan_sha256 = getattr(args, "merge_plan_sha256", None)
        require(not (args.commit_plan and args.merge_plan), "commit and merge plans are mutually exclusive")
        self.output = Path(args.output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.root = Path(args.run_dir).resolve()
        self.state = {"task_id": args.task_id, "job_id": args.job_id,
                      "run_root": str(self.root), "started_at": now(),
                      "state": "waiting", "pid": os.getpid(), "commit_enabled": bool(args.commit_plan),
                      "commit_plan": args.commit_plan, "commit_plan_sha256": args.commit_plan_sha256,
                      "merge_enabled": bool(args.merge_plan), "merge_plan": args.merge_plan,
                      "merge_plan_sha256": args.merge_plan_sha256}

    def checkpoint(self, **values):
        previous = (self.state.get("state"), self.state.get("phase"), self.state.get("platform_status"))
        self.state.update(values, updated_at=now())
        write_json(self.output / "status.json", self.state)
        current = (self.state.get("state"), self.state.get("phase"), self.state.get("platform_status"))
        if current != previous or "error" in values:
            with (self.output / "follow.log").open("a") as log:
                log.write(json.dumps({"at": now(), **values}, ensure_ascii=False) + "\n")

    def run(self):
        with (self.output / "follow.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise NeedsAttention("a follower is already active for this output directory")
            previous_path = self.output / "status.json"
            if previous_path.exists():
                previous = json.loads(previous_path.read_text())
                require(previous.get("task_id") == self.args.task_id
                        and previous.get("job_id") == self.args.job_id
                        and previous.get("run_root") == str(self.root)
                        and previous.get("commit_plan") == self.args.commit_plan
                        and previous.get("commit_plan_sha256") == self.args.commit_plan_sha256
                        and previous.get("merge_plan") == self.args.merge_plan
                        and previous.get("merge_plan_sha256") == self.args.merge_plan_sha256,
                        "output directory belongs to another submission")
                if previous.get("state") in TERMINAL_FOLLOW:
                    return 0 if previous["state"] == "complete" else 2
                self.state.update(previous, pid=os.getpid())
            try:
                require(not (self.output / "acceptance").exists(),
                        "prior acceptance exists; inspect it instead of repeating navigation or Git writes")
                if self.args.commit_plan:
                    load_plan(self.args.commit_plan, self.args.commit_plan_sha256)
                if self.args.merge_plan:
                    load_merge_plan(self.args.merge_plan, self.args.merge_plan_sha256)
                self.checkpoint(phase="polling")
                return self.poll()
            except Exception as exc:
                self.checkpoint(state="needs_attention", error=f"{type(exc).__name__}: {exc}")
                return 2

    def poll(self):
        failures = 0
        while True:
            try:
                raw = command(["train-platform", "detail", "--task-id", self.args.task_id], timeout=90)
                payload = json.loads(raw)
                detail = task_detail(payload, self.args.task_id, self.args.job_id)
                write_json(self.output / "detail.json", payload)
                failures = 0
                self.state.pop("error", None)
            except Exception as exc:
                failures += 1
                self.checkpoint(phase="query_retry", query_failures=failures,
                                error=f"{type(exc).__name__}: {exc}")
                require(failures < 10, "ten consecutive official CLI detail failures")
                time.sleep(self.args.interval)
                continue
            platform_status = detail["status"]
            self.checkpoint(phase="polling", platform_status=platform_status, query_failures=0)
            if platform_status.lower() in TERMINAL_PLATFORM:
                self.checkpoint(state="failed", error="platform task ended before acceptance",
                                reason=detail.get("reasonMsg"), fail_info=detail.get("failInfo"))
                return 1
            # Ignore obsolete ready/failed attempt files while the platform is retrying/queuing.
            ready = read_ready(self.root, self.args.job_id) if platform_status.lower() == "running" else None
            if ready:
                break
            time.sleep(self.args.interval)
        acceptance = self.output / "acceptance"
        if self.args.commit_plan or self.args.merge_plan:
            plan = (load_merge_plan(self.args.merge_plan, self.args.merge_plan_sha256)
                    if self.args.merge_plan else load_plan(self.args.commit_plan, self.args.commit_plan_sha256))
            evidence = verify_runtime_sources(plan, self.root, ready["run_dir"], self.args.job_id)
            write_json(self.output / "preflight-sources.json", evidence)
        self.checkpoint(state="verifying", phase="runtime_acceptance")
        verifier = Path(__file__).with_name("verify_runtime.py")
        with (self.output / "verify.log").open("a") as log:
            result = subprocess.run([sys.executable, str(verifier), "--run-dir", str(self.root),
                                     "--output-dir", str(acceptance), "--expected-job-id", self.args.job_id],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=7200, check=False)
        receipt = acceptance / "receipt.json"
        if result.returncode != 0:
            self.checkpoint(state="failed", error=f"runtime acceptance exited {result.returncode}",
                            acceptance_receipt=str(receipt))
            return 1
        try:
            accepted = verify_receipt(receipt, self.args.job_id, self.root)
        except Exception as exc:
            self.checkpoint(state="failed", error=f"invalid runtime acceptance: {exc}",
                            acceptance_receipt=str(receipt))
            return 1
        if self.args.commit_plan or self.args.merge_plan:
            require(Path(accepted["attempt_dir"]).resolve() == Path(ready["run_dir"]).resolve(),
                    "runtime attempt changed during acceptance")
            require(verify_runtime_sources(plan, self.root, accepted["attempt_dir"], self.args.job_id) == evidence,
                    "runtime preflight sources changed during acceptance")
        self.checkpoint(phase="accepted", acceptance_receipt=str(receipt))
        if self.args.commit_plan:
            commit_verified(self.args.commit_plan, self.args.commit_plan_sha256, receipt,
                            self.args.job_id, self.root, self.checkpoint)
        if self.args.merge_plan:
            merge_verified(self.args.merge_plan, self.args.merge_plan_sha256, receipt,
                           self.args.job_id, self.root, self.checkpoint)
        phase = "merged" if self.args.merge_plan else "published" if self.args.commit_plan else "accepted"
        self.checkpoint(state="complete", phase=phase)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=120)
    plans = parser.add_mutually_exclusive_group()
    plans.add_argument("--commit-plan", help="absolute path of an explicitly authorized frozen Git plan")
    plans.add_argument("--merge-plan", help="absolute path of a frozen acceptance-gated fast-forward plan")
    parser.add_argument("--commit-plan-sha256", help="independent SHA-256 of the exact plan bytes")
    parser.add_argument("--merge-plan-sha256", help="independent SHA-256 of the exact merge plan bytes")
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9]+", args.task_id) is not None, "invalid platform task ID")
    require(re.fullmatch(r"dlc[a-z0-9]+", args.job_id) is not None, "invalid DLC job ID")
    require(args.interval >= 10, "poll interval must be at least ten seconds")
    require(bool(args.commit_plan) == bool(args.commit_plan_sha256),
            "--commit-plan and --commit-plan-sha256 must be provided together")
    if args.commit_plan:
        require(Path(args.commit_plan).is_absolute(), "--commit-plan must be absolute")
    require(bool(args.merge_plan) == bool(args.merge_plan_sha256),
            "--merge-plan and --merge-plan-sha256 must be provided together")
    if args.merge_plan:
        require(Path(args.merge_plan).is_absolute(), "--merge-plan must be absolute")
    return Follower(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
