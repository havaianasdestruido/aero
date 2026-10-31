#!/usr/bin/env bash
# Fetch the bootable demo media that ships with the static site.
#
# The site is fully static: everything a visitor needs is baked into `dist/` at
# build time. This script pulls a freely redistributable DOS floppy image so the
# deployed emulator has something real to boot without asking the visitor for a
# Windows ISO first.
#
# Design rules:
# - Best effort. A missing demo image must never fail the site build: the host
#   page also contains a built-in (generated) boot demo, and visitors can always
#   supply their own disk image.
# - Never commit the image to git (see .gitignore). It is a build artifact.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/apps/web/public/demo"
CACHE_DIR="${ROOT_DIR}/test-images/cache"

# FreeDOS 1.4 "Floppy edition" boot floppy. FreeDOS is GPL-2.0-or-later and
# explicitly redistributable; see demo/README.md in the published site.
ZIP_URL="https://www.ibiblio.org/pub/micro/pc-stuff/freedos/files/distributions/1.4/FD14-FloppyEdition.zip"
ZIP_SHA256="45b1fa7c52dd996c3bfa5e352ffcd410781b952a6ad629f15a4c9ec4bbaefc5a"
ZIP_PATH="${CACHE_DIR}/FD14-FloppyEdition.zip"
OUT_IMG="${OUT_DIR}/freedos.img"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

if [[ -f "${OUT_IMG}" ]]; then
  echo "demo media already present: ${OUT_IMG}"
  exit 0
fi

if [[ "${AERO_SKIP_DEMO_MEDIA:-0}" == "1" ]]; then
  echo "AERO_SKIP_DEMO_MEDIA=1: skipping demo media download"
  exit 0
fi

fetch() {
  if [[ ! -f "${ZIP_PATH}" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
      echo "warning: curl not found; skipping demo media" >&2
      return 1
    fi
    echo "downloading FreeDOS floppy edition..."
    curl -L --fail --retry 4 --retry-delay 2 --max-time 300 -o "${ZIP_PATH}.tmp" "${ZIP_URL}" || return 1
    if ! printf '%s  %s\n' "${ZIP_SHA256}" "${ZIP_PATH}.tmp" | sha256sum -c -; then
      echo "warning: FreeDOS zip checksum mismatch; skipping demo media" >&2
      rm -f "${ZIP_PATH}.tmp"
      return 1
    fi
    mv "${ZIP_PATH}.tmp" "${ZIP_PATH}"
  fi
  return 0
}

if ! fetch; then
  echo "warning: could not download FreeDOS media; the site will ship without it" >&2
  exit 0
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "warning: unzip not found; skipping demo media" >&2
  exit 0
fi

TMP_IMG="${OUT_IMG}.tmp"
if unzip -l "${ZIP_PATH}" | grep -q '144m/x86BOOT.img'; then
  unzip -p "${ZIP_PATH}" 144m/x86BOOT.img > "${TMP_IMG}"
else
  echo "warning: FreeDOS boot floppy not found inside the zip; skipping demo media" >&2
  rm -f "${TMP_IMG}"
  exit 0
fi

# A bootable floppy image must be exactly one 1.44 MiB floppy worth of sectors.
SIZE_BYTES="$(wc -c < "${TMP_IMG}")"
if [[ "${SIZE_BYTES}" -lt 368640 ]]; then
  echo "warning: FreeDOS boot floppy looks truncated (${SIZE_BYTES} bytes); skipping demo media" >&2
  rm -f "${TMP_IMG}"
  exit 0
fi

mv "${TMP_IMG}" "${OUT_IMG}"
echo "demo media ready: ${OUT_IMG} (${SIZE_BYTES} bytes)"
