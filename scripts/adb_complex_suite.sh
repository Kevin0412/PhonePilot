#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-examples/adb_smoke_config.yaml}"
OUT_BASE="${OUT_BASE:-runs/adb_complex}"
PROFILE="${PROFILE:-profiles/android_settings.json}"
RESET_PACKAGES="${RESET_PACKAGES:-com.android.settings com.android.documentsui}"

cd "$ROOT" || exit 1
mkdir -p "$OUT_BASE"

if ! adb get-state >/dev/null 2>&1; then
  echo "ADB device is not ready. Check: adb devices -l" >&2
  exit 1
fi

CASES=(
  "tap_wlan|打开设置并进入 WLAN 页面|examples/adb_settings_wlan_tap.yaml|"
  "search_wlan|在设置中搜索并打开 WLAN 页面|examples/adb_settings_search_wlan.yaml|--auto-confirm-medium"
)

pass=0
fail=0
summary="$OUT_BASE/summary_$(date '+%Y%m%d_%H%M%S').txt"

echo "PhonePilot ADB complex suite"
echo "config=$CONFIG out=$OUT_BASE reset_packages=$RESET_PACKAGES"

for case in "${CASES[@]}"; do
  IFS='|' read -r name task plan extra <<< "$case"
  echo
  echo "[$name] $(date '+%T')"
  for pkg in $RESET_PACKAGES; do
    adb shell am force-stop "$pkg" >/dev/null 2>&1
  done
  adb shell input keyevent KEYCODE_HOME >/dev/null 2>&1
  sleep 0.8
  # shellcheck disable=SC2086
  if python -m phonepilot.cli run \
    --config "$CONFIG" \
    --executor adb \
    --llm mock \
    --plan-file "$plan" \
    --task "$task" \
    --profile "$PROFILE" \
    --run-dir "$OUT_BASE/$name" \
    $extra; then
    pass=$((pass + 1))
    echo "[$name] PASS"
  else
    fail=$((fail + 1))
    echo "[$name] FAIL"
  fi
done

{
  echo "end=$(date '+%F %T')"
  echo "pass=$pass"
  echo "fail=$fail"
  echo "config=$CONFIG"
  echo "out=$OUT_BASE"
  echo "reset_packages=$RESET_PACKAGES"
} | tee "$summary"

if [ "$fail" -eq 0 ]; then
  exit 0
fi
exit 2
