#!/usr/bin/env bash
set -euo pipefail
TASK=${1:?usage: docker/build.sh <task1|task2|task3> <checkpoint_dir> [image_name]}
CKPT=${2:?usage: docker/build.sh <task1|task2|task3> <checkpoint_dir> [image_name]}
IMAGE=${3:-scala-${TASK}}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
case "$TASK" in
  task1) MEMBERS="cavity_resenc scar_resenc scar_surface" ;;
  task2) MEMBERS="cavity_resenc cavity_mednext" ;;
  task3) MEMBERS="ct_resenc ct_stunet" ;;
  *) echo "unknown task $TASK"; exit 1 ;;
esac
CTX=$(mktemp -d "${TMPDIR:-/tmp}/scala_build_XXXX")
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/docker" "$CTX/models"
cp "$REPO/docker/Dockerfile" "$REPO/docker/requirements.txt" "$CTX/docker/"
cp -r "$REPO/scala" "$CTX/scala"
find "$CTX/scala" -name '__pycache__' -type d -prune -exec rm -rf {} +
for m in $MEMBERS; do cp -r "$CKPT/$m" "$CTX/models/$m"; done
docker build --build-arg TASK="$TASK" -f "$CTX/docker/Dockerfile" -t "$IMAGE" "$CTX"
echo "built $IMAGE"
