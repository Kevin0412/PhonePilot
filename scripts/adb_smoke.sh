#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${RUNS:-5}"
CONFIG="${CONFIG:-examples/adb_smoke_config.yaml}"
PLAN_FILE="${PLAN_FILE:-examples/adb_wlan_intent.yaml}"
TASK="${TASK:-打开 WLAN 设置页}"
PROFILE="${PROFILE:-profiles/android_settings.json}"
OUT_BASE="${OUT_BASE:-runs/adb_smoke}"
RESET_HOME="${RESET_HOME:-1}"

cd "$ROOT" || exit 1

mkdir -p "$OUT_BASE"

if ! adb get-state >/dev/null 2>&1; then
  echo "ADB device is not ready. Check: adb devices -l" >&2
  exit 1
fi

echo "PhonePilot ADB smoke"
echo "runs=$RUNS config=$CONFIG plan_file=$PLAN_FILE task=$TASK out=$OUT_BASE reset_home=$RESET_HOME"

pass=0
fail=0
start_ts="$(date '+%F %T')"

for i in $(seq 1 "$RUNS"); do
  echo
  echo "[$i/$RUNS] $(date '+%T')"
  if [ "$RESET_HOME" = "1" ]; then
    adb shell input keyevent KEYCODE_HOME >/dev/null 2>&1
    sleep 0.8
  fi
  if python -m phonepilot.cli run \
    --config "$CONFIG" \
    --executor adb \
    --llm mock \
    --plan-file "$PLAN_FILE" \
    --task "$TASK" \
    --profile "$PROFILE" \
    --run-dir "$OUT_BASE"; then
    pass=$((pass + 1))
    echo "[$i/$RUNS] PASS"
  else
    fail=$((fail + 1))
    echo "[$i/$RUNS] FAIL"
  fi
  sleep 1
done

end_ts="$(date '+%F %T')"
summary="$OUT_BASE/summary_$(date '+%Y%m%d_%H%M%S').txt"
{
  echo "start=$start_ts"
  echo "end=$end_ts"
  echo "runs=$RUNS"
  echo "pass=$pass"
  echo "fail=$fail"
  echo "config=$CONFIG"
  echo "plan_file=$PLAN_FILE"
  echo "task=$TASK"
  echo "profile=$PROFILE"
  echo "reset_home=$RESET_HOME"
} | tee "$summary"

if [ "$fail" -eq 0 ]; then
  exit 0
fi
exit 2
