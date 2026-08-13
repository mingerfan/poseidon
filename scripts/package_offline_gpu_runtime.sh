#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 OUTPUT_DIR [BUILD_DIR]" >&2
    echo "  BUILD_DIR defaults to build-runtime-gpu-api-release." >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=$1
build_dir=${2:-"${repo_root}/build-runtime-gpu-api-release"}
if [[ "${build_dir}" != /* ]]; then
    build_dir="${repo_root}/${build_dir}"
fi
timestamp=$(date +%Y%m%d-%H%M%S)
package_name="poseidon-${timestamp}"
staging_root=$(mktemp -d "${TMPDIR:-/tmp}/poseidon-package.XXXXXX")
staging_dir="${staging_root}/${package_name}"
archive_path="${output_dir}/${package_name}.tar.zst"

cleanup() {
    rm -rf -- "${staging_root}"
}
trap cleanup EXIT

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file is missing: $1" >&2
        exit 1
    fi
}

copy_repo_path() {
    local relative_path=$1
    local destination_parent="${staging_dir}/$(dirname "${relative_path}")"
    mkdir -p "${destination_parent}"
    rsync -a "${repo_root}/${relative_path}" "${destination_parent}/"
}

required_source_files=(
    "cmake/FindGMP.cmake"
    "third_party/rmm/cmake/thirdparty/get_spdlog.cmake"
    "third_party/rmm/cmake/thirdparty/get_cccl.cmake"
    "third_party/rmm/cmake/thirdparty/get_nvtx.cmake"
)

for relative_path in "${required_source_files[@]}"; do
    require_file "${repo_root}/${relative_path}"
done

if [[ ! -d "${build_dir}" ]]; then
    echo "Build directory is missing: ${build_dir}" >&2
    exit 1
fi

mkdir -p "${staging_dir}"
rsync -a \
    --exclude='/.git/' \
    --exclude='/.agents/' \
    --exclude='/.codex/' \
    --exclude='/.venv/' \
    --exclude='/.vscode/' \
    --exclude='/build*/' \
    --exclude='/test-results/' \
    --exclude='/scripts/' \
    --exclude='/thirdparty/' \
    --exclude='/third_party/cpm/cccl/' \
    --exclude='/third_party/cpm/fmt/' \
    --exclude='/third_party/cpm/nvtx3/' \
    --exclude='/third_party/ckks-runtime/third_party/dacapo/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.nsys-rep' \
    --exclude='*.sqlite' \
    --exclude='*.sqlite-shm' \
    --exclude='*.sqlite-wal' \
    "${repo_root}/" "${staging_dir}/"

copy_repo_path "scripts/package_offline_gpu_runtime.sh"

asset_root="third_party/ckks-runtime/third_party/dacapo/review_artifacts/mlp/65536"
copy_repo_path "${asset_root}/1x4-current"
copy_repo_path "${asset_root}/1x8"
copy_repo_path "${asset_root}/profiles"

build_name=$(basename "${build_dir}")
mkdir -p "${staging_dir}/${build_name}"
rsync -a \
    --exclude='/Testing/' \
    --exclude='*.nsys-rep' \
    --exclude='*.sqlite' \
    --exclude='*.sqlite-shm' \
    --exclude='*.sqlite-wal' \
    "${build_dir}/" "${staging_dir}/${build_name}/"

for relative_path in "${required_source_files[@]}"; do
    require_file "${staging_dir}/${relative_path}"
done
require_file "${staging_dir}/${asset_root}/1x4-current/mlp.optimized._hecate_MLP.runtime-plan.json"
require_file "${staging_dir}/${asset_root}/1x4-current/mlp.optimized._hecate_MLP.bundle/manifest.json"
require_file "${staging_dir}/${asset_root}/1x8/input.json"
require_file "${staging_dir}/${asset_root}/profiles/operator-spec.json"

(
    cd "${staging_dir}"
    find . -type f ! -name SHA256SUMS -print0 \
        | sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)

mkdir -p "${output_dir}"
if [[ -e "${archive_path}" ]]; then
    echo "Refusing to overwrite existing archive: ${archive_path}" >&2
    exit 1
fi

tar -C "${staging_root}" --zstd -cf "${archive_path}" "${package_name}"
archive_listing="${staging_root}/archive-listing.txt"
tar --zstd -tf "${archive_path}" >"${archive_listing}"

if grep -Eq '(^|/)(test-results|__pycache__)(/|$)|\.nsys-rep$|\.sqlite(-shm|-wal)?$' \
        "${archive_listing}"; then
    echo "Archive validation failed: excluded output was included" >&2
    exit 1
fi

sha256sum "${archive_path}"
du -h "${archive_path}"
