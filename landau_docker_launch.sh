#!/bin/bash
# Run the GYSELA compression mini-app inside the baked docker image (see
# Dockerfile). The binary and PDI config live in the image, so only the work
# dir (run config + outputs) is mounted. Use it as the Landau2X2V dataset's
# `launcher` parameter; the benchmark invokes it with the minimal contract:
#
#   landau_docker_launch.sh <n_ranks> <config> <work_dir>
#
# The work dir is mounted at its same host path so the absolute paths in the
# run config resolve identically inside. The container runs as the current
# host user (outputs not root-owned). Overridable via env:
#   GYSELA_IMAGE (default gysela-compression:latest)
#   GYSELA_BIN   (default /opt/gysela/compression_app)
#   GYSELA_PDI   (default /opt/gysela/pdi_out.yaml)
set -euo pipefail

n_ranks=$1; config=$2; work_dir=$3

IMAGE="${GYSELA_IMAGE:-ghcr.io/gyselax/benchmarks/landau2x2v:latest}"
BIN="${GYSELA_BIN:-/opt/gysela/compression_app}"
PDI="${GYSELA_PDI:-/opt/gysela/pdi_out.yaml}"

exec docker run --rm \
    --user ":$(id -g)" \
    -v "${work_dir}:/work" \
    --workdir "/work" \
    "${IMAGE}" \
    mpirun --allow-run-as-root -n "${n_ranks}" "${BIN}" "${config}" "${PDI}"
