#!/usr/bin/env bash
# Source this file to activate Python and, optionally, the native SYCL runtime.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source ./activate.sh [--full]" >&2
    exit 2
fi

_synthetic_activate() {
    local repo_dir venv_dir candidate
    repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    venv_dir="${VENV_DIR:-$repo_dir/.venv}"
    if [[ ! -f "$venv_dir/bin/activate" ]]; then
        echo "Run ./setup.sh first." >&2
        return 1
    fi
    if [[ "${1:-}" == "--full" ]]; then
        candidate="${ONEAPI_ROOT:-}/setvars.sh"
        if [[ ! -f "$candidate" ]]; then
            for candidate in /opt/intel/oneapi/setvars.sh "$HOME/intel/oneapi/setvars.sh"; do
                if [[ -f "$candidate" ]]; then break; fi
            done
        fi
        if [[ ! -f "$candidate" ]]; then
            echo "Set ONEAPI_ROOT to the oneAPI installation containing setvars.sh." >&2
            return 1
        fi
        # Intel's environment script configures both GPU and CPU SYCL runtimes.
        source "$candidate" >/dev/null || return
        if [[ -n "${CUDA_HOME:-}" ]]; then
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        fi
    elif [[ -n "${1:-}" ]]; then
        echo "Use: source ./activate.sh [--full]" >&2
        return 2
    fi
    source "$venv_dir/bin/activate"
}
if _synthetic_activate "$@"; then
    unset -f _synthetic_activate
else
    unset -f _synthetic_activate
    return 1
fi
