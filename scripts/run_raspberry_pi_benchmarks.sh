#!/usr/bin/env bash
set -Eeuo pipefail

RPI_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RPI_PYTHON_BIN="${RPI_PYTHON_BIN:-python3}"
RPI_VENV_DIR="${RPI_VENV_DIR:-${RPI_PROJECT_ROOT}/.venv-rpi}"
RPI_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RPI_RESULTS_DIR="${RPI_RESULTS_DIR:-${RPI_PROJECT_ROOT}/results/raspberry-pi/${RPI_RUN_ID}}"

CUBE_DATASET="${RPI_PROJECT_ROOT}/datasets/cubes-on-conveyor-belt"
ROAD_DATASET="${RPI_PROJECT_ROOT}/datasets/vehicle-coco-examples"
CUBE_MODEL="${RPI_PROJECT_ROOT}/models/yolo26n-cube.onnx"
ROAD_MODEL="${RPI_PROJECT_ROOT}/models/yolo26n-coco.onnx"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "Required file not found: $1"
}

require_directory() {
    [[ -d "$1" ]] || fail "Required directory not found: $1"
}

install_venv_support_if_needed() {
    if ! command -v dpkg-query >/dev/null 2>&1; then
        return
    fi

    local package_status
    package_status="$(dpkg-query -W -f='${Status}' python3-venv 2>/dev/null || true)"
    if [[ "$package_status" == "install ok installed" ]]; then
        return
    fi

    echo "Installing the Raspberry Pi OS python3-venv system package..."
    if [[ "$(id -u)" -eq 0 ]]; then
        apt-get update
        apt-get install -y python3-venv
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y python3-venv
    else
        fail "python3-venv is missing and sudo is unavailable"
    fi
}

record_environment() {
    local destination="$1"
    {
        echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        if [[ -r /proc/device-tree/model ]]; then
            echo "device_model=$(tr -d '\0' </proc/device-tree/model)"
        else
            echo "device_model=unknown"
        fi
        if [[ -r /etc/os-release ]]; then
            # shellcheck disable=SC1091
            source /etc/os-release
            echo "operating_system=${PRETTY_NAME:-unknown}"
        fi
        echo "architecture=$(uname -m)"
        echo "kernel=$(uname -sr)"
        echo "cpu_count=$(nproc)"
        echo "memory_total_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
        echo "git_commit=$(git -C "$RPI_PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "cube_model_sha256=$(sha256sum "$CUBE_MODEL" | awk '{print $1}')"
        echo "road_model_sha256=$(sha256sum "$ROAD_MODEL" | awk '{print $1}')"
        if [[ -r /sys/devices/system/cpu/cpufreq/policy0/scaling_governor ]]; then
            echo "cpu_governor=$(</sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
        fi
        "${RPI_VENV_DIR}/bin/python" -c \
            'import cv2, numpy, onnxruntime, platform, yaml; print(f"python={platform.python_version()}\nnumpy={numpy.__version__}\nopencv={cv2.__version__}\nonnxruntime={onnxruntime.__version__}\npyyaml={yaml.__version__}")'
        if command -v vcgencmd >/dev/null 2>&1; then
            echo "temperature=$(vcgencmd measure_temp)"
            echo "throttled=$(vcgencmd get_throttled)"
        fi
    } >"$destination"
}

command -v "$RPI_PYTHON_BIN" >/dev/null 2>&1 \
    || fail "Python executable not found: $RPI_PYTHON_BIN"

"$RPI_PYTHON_BIN" -c \
    'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
    || fail "Python 3.11 or newer is required"

case "$(uname -m)" in
    aarch64 | arm64) ;;
    armv6l | armv7l)
        fail "A 64-bit Raspberry Pi OS installation is required for ONNX Runtime"
        ;;
    *)
        echo "WARNING: $(uname -m) is not a Raspberry Pi ARM64 architecture; continuing."
        ;;
esac

require_file "${RPI_PROJECT_ROOT}/config.yaml"
require_file "${RPI_PROJECT_ROOT}/configs/road.yaml"
require_file "$CUBE_MODEL"
require_file "$ROAD_MODEL"
require_file "$CUBE_DATASET/testing/bounding_boxes.labels"
require_directory "$ROAD_DATASET"

if [[ ! -x "${RPI_VENV_DIR}/bin/python" ]]; then
    install_venv_support_if_needed
    echo "Creating virtual environment: $RPI_VENV_DIR"
    "$RPI_PYTHON_BIN" -m venv "$RPI_VENV_DIR"
fi

echo "Installing inference-only project dependencies..."
"${RPI_VENV_DIR}/bin/python" -m pip install --upgrade pip
"${RPI_VENV_DIR}/bin/python" -m pip install -e "$RPI_PROJECT_ROOT"

mkdir -p "$RPI_RESULTS_DIR"
record_environment "$RPI_RESULTS_DIR/environment-before.txt"

cd "$RPI_PROJECT_ROOT"

run_benchmark() {
    local dataset_name="$1"
    local mode="$2"
    local config_path="$3"
    local dataset_path="$4"
    local split="$5"
    local output_path="$RPI_RESULTS_DIR/$dataset_name/$mode"

    echo "Running ${dataset_name}/${mode}..."
    "${RPI_VENV_DIR}/bin/python" app.py \
        --mode "$mode" \
        --config "$config_path" \
        --dataset "$dataset_path" \
        --split "$split" \
        --output "$output_path" \
        --metrics-only
}

run_benchmark cubes opencv config.yaml "$CUBE_DATASET" testing
run_benchmark cubes yolo config.yaml "$CUBE_DATASET" testing
run_benchmark road opencv configs/road.yaml "$ROAD_DATASET" val
run_benchmark road yolo configs/road.yaml "$ROAD_DATASET" val

record_environment "$RPI_RESULTS_DIR/environment-after.txt"

echo "All four benchmarks completed."
echo "Results: $RPI_RESULTS_DIR"
