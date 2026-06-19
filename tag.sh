#!/usr/bin/env bash
set -euo pipefail

tag="${TAG:-$(date +%Y.%-m.%-d)}"
remote="${REMOTE:-origin}"

git rev-parse --is-inside-work-tree >/dev/null
if [ "${1:-}" = "--check" ]; then
  set -x
  gh run list --branch "$tag" --limit 1
  exit 0
fi
git tag -d "$tag" 2>/dev/null || true
git push "$remote" ":refs/tags/$tag" 2>/dev/null || true
git tag "$tag"
git push "$remote" "$tag"
