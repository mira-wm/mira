#!/usr/bin/env bash
# Run the versioned strict audit and deterministic report after the pinned pilot audit succeeds.
set -euo pipefail

run_root=${1:?Usage: watch_cs2_strict_audit_report.sh RUN_ROOT PROJECT PYTHON TRAIN_COMMIT EVAL_COMMIT REPORT_COMMIT}
project_dir=${2:?}
python_bin=${3:?}
training_commit=${4:?}
evaluator_commit=${5:?}
report_commit=${6:?}
status_file=$run_root/strict_audit_report_status.tsv
pinned_watcher_pid=$(cat "$run_root/final_audit_watcher.pid")

printf '%s\twatching\tpinned_audit_watcher_pid=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pinned_watcher_pid" >>"$status_file"
while kill -0 "$pinned_watcher_pid" 2>/dev/null; do
  sleep 60
done

if ! tail -n 1 "$run_root/final_audit_status.tsv" | grep -q $'\taudit\tcomplete$'; then
  printf '%s\tblocked\tpinned_audit_incomplete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"$status_file"
  exit 1
fi
if ! "$python_bin" - "$run_root/audit.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if payload.get("status") == "pass" else 1)
PY
then
  printf '%s\tblocked\tpinned_audit_not_pass\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"$status_file"
  exit 1
fi

cd "$project_dir"
observed_commit=$(git rev-parse HEAD)
if [[ "$observed_commit" != "$report_commit" || -n "$(git status --porcelain=v1)" ]]; then
  printf '%s\tblocked\treport_source_identity_mismatch\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"$status_file"
  exit 1
fi

printf '%s\tstrict_audit\trunning\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
set +e
"$python_bin" scripts/audit_cs2_rebuttal_run.py "$run_root" \
  --expected-training-commit "$training_commit" \
  --expected-evaluator-commit "$evaluator_commit" \
  --output "$run_root/audit_strict.json"
audit_result=$?
set -e
if [[ $audit_result -ne 0 ]]; then
  printf '%s\tstrict_audit\tfailed_exit_%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$audit_result" >>"$status_file"
  exit "$audit_result"
fi
printf '%s\tstrict_audit\tcomplete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"

printf '%s\treport\trunning\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
set +e
"$python_bin" scripts/render_cs2_rebuttal_report.py "$run_root" \
  --audit "$run_root/audit_strict.json" \
  --output "$run_root/evaluation/rebuttal_report.md"
report_result=$?
set -e
if [[ $report_result -ne 0 ]]; then
  printf '%s\treport\tfailed_exit_%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$report_result" >>"$status_file"
  exit "$report_result"
fi
sha256sum "$run_root/audit.json" "$run_root/audit_strict.json" \
  "$run_root/evaluation/rebuttal_report.md" >"$run_root/provenance/final_report_artifacts.sha256"
printf '%s\n' "$report_commit" >"$run_root/provenance/report_code_commit.txt"
printf '%s\treport\tcomplete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$status_file"
