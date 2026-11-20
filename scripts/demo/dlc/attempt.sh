#!/usr/bin/env bash
# Source from entry.sh so FD 9 keeps the output-root lease through exec.
prepare_vln_attempt() {
  : "${VLN_RUN_DIR:?Set VLN_RUN_DIR to the persistent task output root}"
  if [[ "$VLN_RUN_DIR" != /* ]]; then
    echo '[entry] VLN_RUN_DIR must be an absolute path' >&2
    return 1
  fi
  mkdir -p -- "$VLN_RUN_DIR"
  VLN_RUN_ROOT=$(cd -- "$VLN_RUN_DIR" && pwd -P)
  # Also excludes a supervisor launched with the old, non-attempt layout.
  exec 9>"$VLN_RUN_ROOT/.supervisor.lock"
  if ! flock -n 9; then
    echo "[entry] another supervisor owns $VLN_RUN_ROOT" >&2
    return 1
  fi
  mkdir -p -- "$VLN_RUN_ROOT/attempts"
  local name legacy_dir='' pointer relative
  for name in logs pids status.json preflight.json bin pi-agent; do
    if [[ -e "$VLN_RUN_ROOT/$name" && ! -L "$VLN_RUN_ROOT/$name" ]]; then
      if [[ -z "$legacy_dir" ]]; then
        legacy_dir=$(mktemp -d "$VLN_RUN_ROOT/attempts/legacy-XXXXXX")
      fi
      mv -- "$VLN_RUN_ROOT/$name" "$legacy_dir/$name"
    fi
  done
  VLN_RUN_DIR=$(mktemp -d "$VLN_RUN_ROOT/attempts/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
  mkdir -p -- "$VLN_RUN_DIR/logs" "$VLN_RUN_DIR/bin" "$VLN_RUN_DIR/pids"
  relative="attempts/${VLN_RUN_DIR##*/}"
  for name in logs pids status.json preflight.json bin pi-agent latest; do
    pointer="$VLN_RUN_ROOT/.${name}.next.$$"
    if [[ "$name" == latest ]]; then
      ln -sfn -- "$relative" "$pointer"
    else
      ln -sfn -- "$relative/$name" "$pointer"
    fi
    mv -Tf -- "$pointer" "$VLN_RUN_ROOT/$name"
  done
  export VLN_RUN_ROOT VLN_RUN_DIR
}
