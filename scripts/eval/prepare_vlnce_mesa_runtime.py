#!/usr/bin/env python3
"""Prepare a version-bound, immutable Mesa input seed for Habitat CPU rendering.

Shared storage holds a verified input artifact, never a writable runtime cache.
Copy the seed into each DLC container and apply its environment only to Habitat
processes. System libraries and the shared Python environment are read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
DATA = Path("/data/zerovln")
SEED = DATA / "vln_verl/runtime_inputs/habitat-mesa-25.2.8-ubuntu24.04-amd64"
SYSTEM_LIB = Path("/usr/lib/x86_64-linux-gnu")
GV_LIB = Path("/opt/navigation-runtime/venv/lib")
MESA_VERSION = "25.2.8-0ubuntu0.24.04.2"
SCHEMA = "habitat_mesa_runtime_seed_v1"
ROOTS = {
    "lib/libEGL.so.1": SYSTEM_LIB / "libEGL.so.1",
    "lib/libEGL_mesa.so.0": SYSTEM_LIB / "libEGL_mesa.so.0",
    "lib/libGLdispatch.so.0": SYSTEM_LIB / "libGLdispatch.so.0",
    "lib/libgbm.so.1": SYSTEM_LIB / "libgbm.so.1",
    "lib/libOpenGL.so.0": GV_LIB / "libOpenGL.so.0",
    "dri/swrast_dri.so": SYSTEM_LIB / "dri/swrast_dri.so",
}
BASE_LIBS = re.compile(
    r"^(?:linux-vdso|ld-linux|lib(?:c|m|pthread|dl|rt|resolv|util|anl)\.so(?:\.|$))")

spec = importlib.util.spec_from_file_location(
    "_vlnce_mesa_common", REPO / "verl/scripts/eval/qwen36_4query_ckpt_sweep/common.py")
COMMON = importlib.util.module_from_spec(spec)
spec.loader.exec_module(COMMON)
sha256_file, atomic_json = COMMON.sha256_file, COMMON.atomic_json


def clean_env(library_path=None):
    env = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
        env.pop(key, None)
    env["LC_ALL"] = "C"
    if library_path is not None:
        env["LD_LIBRARY_PATH"] = str(library_path)
    return env


def command(args, library_path=None):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True,
                          env=clean_env(library_path)).stdout


def dependencies(path, library_path=None):
    output = command(["ldd", str(path)], library_path)
    if "not found" in output:
        raise RuntimeError(f"unresolved library for {path}:\n{output}")
    result = {}
    for line in output.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(/\S+)\s+\(", line)
        if match:
            result[match[1]] = Path(match[2])
        else:
            match = re.match(r"\s*(/\S+)\s+\(", line)
            if match:
                result[Path(match[1]).name] = Path(match[1])
    return result


def package_identity(path):
    result = subprocess.run(["dpkg-query", "-S", str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, env=clean_env())
    if result.returncode:
        return None
    package = result.stdout.splitlines()[0].split(": ", 1)[0]
    version = command(["dpkg-query", "-W", "-f=${Version}", package]).strip()
    return {"name": package, "version": version}


def version_tuple(value):
    return tuple(int(part) for part in value.split("."))


def required_glibc(path):
    values = re.findall(r"\bGLIBC_(\d+(?:\.\d+)+)\b",
                        command(["readelf", "--version-info", str(path)]))
    return max(values, key=version_tuple) if values else "0.0"


def source_plan():
    if platform.machine() != "x86_64":
        raise RuntimeError("Mesa seed is bound to x86_64/amd64")
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines()
                   if "=" in line)
    if release.get("ID", "").strip('"') != "ubuntu" or release.get("VERSION_ID", "").strip('"') != "24.04":
        raise RuntimeError("Mesa seed requires the verified Ubuntu 24.04 source system")
    for package in ("libegl-mesa0", "libgbm1", "libgl1-mesa-dri"):
        actual = command(["dpkg-query", "-W", "-f=${Version}", package]).strip()
        if actual != MESA_VERSION:
            raise RuntimeError(f"source Mesa version changed: {package}={actual}")
    sources, skipped = {}, {}
    pending = list(ROOTS.items())
    inspected = set()
    while pending:
        destination, source = pending.pop()
        resolved = source.resolve(strict=True)
        existing = sources.get(destination)
        if existing is not None:
            if existing != resolved:
                raise RuntimeError(f"SONAME collision: {destination}: {existing} / {resolved}")
            continue
        sources[destination] = resolved
        if resolved in inspected:
            continue
        inspected.add(resolved)
        for soname, dependency in dependencies(resolved).items():
            if BASE_LIBS.match(soname):
                skipped[soname] = str(dependency.resolve())
            else:
                pending.append(("lib/" + soname, dependency))
    # swrast is Mesa's dlopen entry; preserve the actual installed driver name
    # as well as the entry alias without following a symlink outside the seed.
    driver = ROOTS["dri/swrast_dri.so"].resolve()
    sources["dri/" + driver.name] = driver
    rows = {}
    for destination, source in sorted(sources.items()):
        rows[destination] = {
            "source_path": str(source),
            "source_requested_paths": [str(path) for name, path in ROOTS.items()
                                       if path.resolve() == source],
            "bytes": source.stat().st_size, "sha256": sha256_file(source),
            "debian_package": package_identity(source),
            "minimum_glibc": required_glibc(source),
        }
    return {
        "schema": SCHEMA, "artifact_kind": "version_bound_read_only_runtime_seed",
        "mesa_version": MESA_VERSION, "source_os": "ubuntu24.04", "arch": "amd64",
        "minimum_glibc": max((row["minimum_glibc"] for row in rows.values()), key=version_tuple),
        "roots": {name: str(path) for name, path in ROOTS.items()},
        "excluded_base_libraries": skipped, "files": rows,
        "environment": {
            "LD_LIBRARY_PATH": "<container-local-seed>/lib:<existing-habitat-libraries>",
            "LIBGL_DRIVERS_PATH": "<container-local-seed>/dri",
            "__EGL_VENDOR_LIBRARY_FILENAMES": "<container-local-seed>/50_mesa.json",
            "EGL_PLATFORM": "surfaceless", "LIBGL_ALWAYS_SOFTWARE": "1",
            "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe", "GALLIUM_DRIVER": "llvmpipe",
        },
    }


def verify_seed(path):
    """Read-only SHA256, ABI and contained dependency checks for submit/preflight."""
    root = Path(path).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("mesa_version") != MESA_VERSION
            or manifest.get("artifact_kind") != "version_bound_read_only_runtime_seed"):
        raise ValueError("not the expected version-bound Habitat Mesa seed")
    if platform.machine() != "x86_64":
        raise ValueError("Habitat Mesa seed requires x86_64")
    libc = os.confstr("CS_GNU_LIBC_VERSION").split()[-1]
    if version_tuple(libc) < version_tuple(manifest["minimum_glibc"]):
        raise ValueError(f"host GLIBC {libc} is below seed minimum {manifest['minimum_glibc']}")
    rows = manifest["files"]
    required = {*ROOTS, "50_mesa.json"}
    if not required <= rows.keys():
        raise ValueError("Mesa seed is missing an EGL/DRI root or ICD descriptor")
    for name, info in rows.items():
        target = root / name
        if Path(name).is_absolute() or not target.resolve().is_relative_to(root) or target.is_symlink():
            raise ValueError(f"unsafe seed member: {name}")
        if not target.is_file() or target.stat().st_size != info["bytes"] or sha256_file(target) != info["sha256"]:
            raise ValueError(f"Mesa seed content mismatch: {target}")
        if BASE_LIBS.match(target.name):
            raise ValueError(f"Mesa seed must not override base GLIBC: {target}")
    expected = set(rows) | {"manifest.json"}
    actual = {str(item.relative_to(root)) for item in root.rglob("*") if item.is_file()}
    if actual != expected:
        raise ValueError(f"unexpected seed files: {sorted(actual - expected)}")
    icd = json.loads((root / "50_mesa.json").read_text())
    if icd != {"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_mesa.so.0"}}:
        raise ValueError("Mesa ICD descriptor must use contained libEGL_mesa.so.0")
    checked = 0
    for name in rows:
        if not name.startswith(("lib/", "dri/")):
            continue
        for soname, resolved in dependencies(root / name, root / "lib").items():
            if BASE_LIBS.match(soname):
                continue
            if not resolved.resolve().is_relative_to(root):
                raise ValueError(f"dependency escapes seed: {name}: {soname} => {resolved}")
        checked += 1
    return {"ok": True, "seed": str(root), "manifest_sha256": sha256_file(root / "manifest.json"),
            "mesa_version": manifest["mesa_version"], "minimum_glibc": manifest["minimum_glibc"],
            "host_glibc": libc, "files": len(rows), "elf_files_checked": checked,
            "bytes": sum(info["bytes"] for info in rows.values())}


def prepare_seed():
    if SEED.exists():
        return verify_seed(SEED)
    plan = source_plan()
    staging = SEED.with_name(SEED.name + f".staging.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for name, info in plan["files"].items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(info["source_path"], target)
        descriptor = staging / "50_mesa.json"
        atomic_json(descriptor, {"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_mesa.so.0"}})
        plan["files"]["50_mesa.json"] = {
            "source_path": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
            "source_requested_paths": ["/usr/share/glvnd/egl_vendor.d/50_mesa.json"],
            "bytes": descriptor.stat().st_size, "sha256": sha256_file(descriptor),
            "debian_package": package_identity(Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")),
        }
        plan["created_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(staging / "manifest.json", plan)
        verify_seed(staging)
        # Publish once; a competing creator must not replace an existing seed.
        staging.rename(SEED)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return verify_seed(SEED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_seed(args.verify_only)
    elif args.dry_run:
        plan = source_plan()
        result = {"dry_run": True, "seed": str(SEED), "files": len(plan["files"]),
                  "bytes": sum(row["bytes"] for row in plan["files"].values()),
                  "minimum_glibc": plan["minimum_glibc"], "plan": plan}
    else:
        result = prepare_seed()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
