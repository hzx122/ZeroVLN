#!/usr/bin/env bash
# Run the page's tests. No dependencies beyond node -- the DOM is shim.mjs.
#
#   ./run.sh                    # against the committed fixtures
#   ./run.sh 5000               # refresh the fixtures from a live console first
#
# The modules are copied to a scratch directory next to a {"type":"module"}
# package.json rather than dropping one into src/web/js: that tree is served to
# the browser, and a stray package.json in it would be reachable at
# /static/package.json.
set -u
cd "$(dirname "$0")/../../../vln_demo/src/web/test"
HERE=$PWD
WEB=$(cd .. && pwd)
TMP=${TMPDIR:-/tmp}/vlnweb-test.$$
trap 'rm -rf "$TMP"' EXIT

if [ $# -ge 1 ]; then
  PORT=$1
  echo "refreshing fixtures from 127.0.0.1:$PORT"
  curl -fsS "http://127.0.0.1:$PORT/config.json"   -o fixtures/config.json || exit 1
  curl -fsS "http://127.0.0.1:$PORT/status?since=0" -o fixtures/status.json || exit 1
fi

mkdir -p "$TMP"
cp -r "$WEB/js" "$TMP/js"
cp -r "$WEB/css" "$TMP/css"          # css.mjs reads the stylesheets
printf '{"type":"module"}\n' > "$TMP/package.json"

pass=0; fail=0
for t in boot interact params tabs dock view map css lock reference request terminal status action-head; do
  if out=$(node "$HERE/$t.mjs" "$TMP" "$HERE/fixtures/config.json" \
                "$HERE/fixtures/status.json" 2>&1); then
    pass=$((pass + 1)); printf 'PASS  %-9s %s\n' "$t" "$(printf '%s' "$out" | tail -1)"
  else
    fail=$((fail + 1)); printf 'FAIL  %-9s\n' "$t"; printf '%s\n' "$out" | sed 's/^/      /'
  fi
done
if out=$(PYTHONPATH="$WEB/..${PYTHONPATH:+:$PYTHONPATH}" \
         python3 "$HERE/frame_pipeline.py" 2>&1); then
  pass=$((pass + 1)); printf 'PASS  %-9s %s\n' "frame" "$(printf '%s' "$out" | tail -1)"
else
  fail=$((fail + 1)); printf 'FAIL  %-9s\n' "frame"; printf '%s\n' "$out" | sed 's/^/      /'
fi
echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
