#!/usr/bin/env bash
# Re-emit the most useful lines of a failed build log as GitHub Actions
# annotations.
#
# Why: the Actions log download endpoint is not reachable from every
# environment that needs to triage a red run, but the *annotations* of a check
# run are available through the REST API. Annotations are also what humans see
# inline on the run page, so this is useful beyond CI triage.
#
# Usage: emit-failure-annotations.sh <log-file> [max-lines]
set -uo pipefail

LOG_FILE="${1:-}"
MAX_LINES="${2:-40}"

if [[ -z "${LOG_FILE}" || ! -f "${LOG_FILE}" ]]; then
  echo "::error::build failed (no log captured at '${LOG_FILE:-<unset>}')"
  exit 1
fi

# Collapse CR, strip ANSI escapes, and keep lines that look like diagnostics.
sanitize() {
  sed -e 's/\r$//' -e 's/\x1b\[[0-9;]*[A-Za-z]//g' "$1" \
    | grep -E 'error(\[E[0-9]+\])?:|error: could not|panicked at|note: required by|--> |warning: unused' \
    | grep -v '^\s*$' \
    | tail -n "${MAX_LINES}" \
    | cut -c1-900 \
    | sed -e 's/%/%25/g' -e 's/\r//g' -e 's/\n/ /g'
}

mapfile -t LINES < <(sanitize "${LOG_FILE}")

if [[ ${#LINES[@]} -eq 0 ]]; then
  # Fall back to the raw tail so a silent failure still explains itself.
  mapfile -t LINES < <(tail -n "${MAX_LINES}" "${LOG_FILE}" | cut -c1-900 | sed -e 's/%/%25/g')
fi

for line in "${LINES[@]}"; do
  echo "::error::${line}"
done

echo "::error::(full log captured as an artifact: build-logs)"
exit 1
