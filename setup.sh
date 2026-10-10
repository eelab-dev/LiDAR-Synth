#!/usr/bin/env bash
# Prepare this checkout; native GPU drivers/toolchains are system prerequisites.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode="--noise-only"
case "${1:-$mode}" in
    --noise-only|--full) mode="${1:-$mode}" ;;
    -h|--help)
        cat <<'HELP'
Usage: ./setup.sh [--noise-only | --full]
  --noise-only  Install Python package and build the Rust sensor extension (default).
  --full        Also install OpenUSD, build Rust pixelization and all SYCL executables.
Overrides: PYTHON, VENV_DIR, ONEAPI_ROOT, CUDA_HOME, HDF5_DIR, SYCL_TARGETS, BUILD_JOBS.
Full prerequisites: Git, CMake >=3.22, HDF5 development libraries, Rust/Cargo,
oneAPI icx/icpx and a compatible SYCL GPU runtime (CUDA plugin for NVIDIA).
No system packages or GPU drivers are installed by this script.
HELP
        exit 0 ;;
    *) echo "Unknown setup option: $1" >&2; exit 2 ;;
esac

python_bin="${PYTHON:-python3}"
venv_dir="${VENV_DIR:-$repo_dir/.venv}"
command -v "$python_bin" >/dev/null || { echo "Python not found: $python_bin" >&2; exit 1; }
command -v cargo >/dev/null || { echo "Rust/Cargo is required. Install the Rust toolchain first." >&2; exit 1; }
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Python >=3.11 is required"'
if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$python_bin" -m venv "$venv_dir"
fi
python_env="$venv_dir/bin/python"
"$python_env" -m pip install 'pip>=23' 'setuptools>=68' 'maturin>=1.7,<2'
package_spec="$repo_dir"
if [[ "$mode" == "--full" ]]; then package_spec="$repo_dir[scene]"; fi
"$python_env" -m pip install -e "$package_spec"
# pip builds a wheel through maturin in an isolated build environment.
"$python_env" -m pip install --no-deps "$repo_dir/src/spad_noise_injection/rust"
"$python_env" -c 'import _synthetic_sensor; print("Rust sensor extension loaded")'

if [[ "$mode" == "--full" ]]; then
    for executable in git cmake; do
        command -v "$executable" >/dev/null || { echo "Missing prerequisite: $executable" >&2; exit 1; }
    done
    if [[ -n "${ONEAPI_ROOT:-}" && -f "$ONEAPI_ROOT/setvars.sh" ]]; then
        set +u
        source "$ONEAPI_ROOT/setvars.sh" >/dev/null
        set -u
    elif ! command -v icpx >/dev/null; then
        for candidate in /opt/intel/oneapi "$HOME/intel/oneapi"; do
            if [[ -f "$candidate/setvars.sh" ]]; then
                set +u
                source "$candidate/setvars.sh" >/dev/null
                set -u
                break
            fi
        done
    fi
    command -v icpx >/dev/null || { echo "icpx not found. Set ONEAPI_ROOT or source your oneAPI environment." >&2; exit 1; }
    if [[ -n "${CUDA_HOME:-}" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    fi
    # Pin the exact dependency used by the source simulator.
    tinyusdz_revision=a04ee0bcbd1a930e30cc40938fcee3526a6fa8eb
    tinyusdz_dir="$repo_dir/src/monte_carlo_simulation/simulator/external/tinyusdz"
    if [[ ! -d "$tinyusdz_dir/.git" ]]; then
        mkdir -p "$(dirname "$tinyusdz_dir")"
        git init "$tinyusdz_dir"
        git -C "$tinyusdz_dir" remote add origin https://github.com/lighttransport/tinyusdz.git
    fi
    if ! git -C "$tinyusdz_dir" cat-file -e "$tinyusdz_revision^{commit}"; then
        git -C "$tinyusdz_dir" fetch --depth 1 origin "$tinyusdz_revision"
    fi
    if [[ -n "$(git -C "$tinyusdz_dir" status --porcelain)" ]]; then
        echo "tinyusdz has local changes; refusing to replace them." >&2
        exit 1
    fi
    git -C "$tinyusdz_dir" checkout --detach "$tinyusdz_revision"
    cmake_extra=()
    if [[ -n "${HDF5_DIR:-}" ]]; then
        # A custom HDF5 installation must also be discoverable when running later.
        export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,$HDF5_DIR/lib"
        cmake_extra+=("-DHDF5_ROOT=$HDF5_DIR" "-DCMAKE_BUILD_RPATH=$HDF5_DIR/lib")
    fi
    cargo build --release --locked --manifest-path "$repo_dir/src/monte_carlo_simulation/pixelization/Cargo.toml"
    cmake -S "$repo_dir/src/monte_carlo_simulation/simulator" -B "$repo_dir/src/monte_carlo_simulation/simulator/build" \
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
        -DSYCL_TARGETS="${SYCL_TARGETS:-nvptx64-nvidia-cuda,spir64_x86_64}" "${cmake_extra[@]}"
    cmake --build "$repo_dir/src/monte_carlo_simulation/simulator/build" --parallel "${BUILD_JOBS:-2}"
fi
echo "Setup complete. Use: $venv_dir/bin/synthetic-data --help"
if [[ "$mode" == "--full" ]]; then
    echo "Before native runs: source $repo_dir/activate.sh --full"
fi
