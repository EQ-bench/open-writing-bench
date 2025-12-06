#!/usr/bin/env python
"""
install_llamacpp.py

Install llama-cpp-python for the current environment, preferring prebuilt wheels
(CUDA, Metal, CPU) when possible, and otherwise falling back to a source build.

Features:

- Detects OS, architecture, Python version, CUDA version, and rough GPU presence.
- Knows about official prebuilt indices:
    * CPU:  https://abetlen.github.io/llama-cpp-python/whl/cpu
    * CUDA: https://abetlen.github.io/llama-cpp-python/whl/cu121 .. cu125
    * Metal: https://abetlen.github.io/llama-cpp-python/whl/metal
- Optionally knows about community wheels:
    * https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/<AVX>/<cuda_tag|cpu>
- If a suitable prebuilt wheel is found (and pip can install it), use it.
- Otherwise, attempt source builds with appropriate CMake flags for CUDA / Metal / ROCm,
  finally falling back to a plain CPU source build.

Usage (examples):

    # Auto-detect and install (CPU or GPU, preferring GPU):
    python install_llamacpp.py

    # Force CUDA backend, allow community wheels:
    python install_llamacpp.py --backend cuda --allow-unofficial-wheels

    # Force CPU-only, no community wheels:
    python install_llamacpp.py --backend cpu

    # Pick specific llama-cpp-python version, with extras:
    python install_llamacpp.py --version 0.3.16 --extras server

    # Dry-run to see what it would do:
    python install_llamacpp.py --dry-run

Environment overrides (optional):

    LLAMACPP_CUDA_TAG / LLAMA_CPP_PYTHON_CUDA_TAG
        Override official CUDA tag (e.g. "cu121", "cu124").

    LLAMACPP_JLLLLLL_CUDA_TAG
        Override jllllll CUDA tag (e.g. "cu117").

    LLAMACPP_AVX_LEVEL / LLAMA_CPP_PYTHON_AVX_LEVEL
        Override AVX level for jllllll wheels: BASIC, AVX, AVX2, AVX512.

    CMAKE_ARGS
        Existing CMake options will be merged with required flags for GPU backends.

"""

import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

# Official llama-cpp-python wheel indexes (documented by the maintainer).
# CPU wheels:
OFFICIAL_CPU_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
# Metal wheels (macOS 11+; Python 3.10–3.12 as of docs).
OFFICIAL_METAL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/metal"
# CUDA wheels: Python 3.10–3.12 and CUDA 12.1–12.5 as of docs.
OFFICIAL_CUDA_INDEX_TEMPLATE = "https://abetlen.github.io/llama-cpp-python/whl/{cuda_tag}"
OFFICIAL_CUDA_TAGS = {
    (12, 1): "cu121",
    (12, 2): "cu122",
    (12, 3): "cu123",
    (12, 4): "cu124",
    (12, 5): "cu125",
}

# Community wheels from jllllll/llama-cpp-python-cuBLAS-wheels.
# Requirements from their README:
#   - Windows x64, Linux x64, or macOS 11.0+
#   - CUDA 11.6–12.2
#   - CPython 3.8–3.11
JLLLLLL_BASE_INDEX = "https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels"
JLLLLLL_CUDA_TAGS = {
    (11, 6): "cu116",
    (11, 7): "cu117",
    (11, 8): "cu118",
    (12, 0): "cu120",
    (12, 1): "cu121",
    (12, 2): "cu122",
}


@dataclass
class EnvInfo:
    os: str
    arch: str
    python_version: Tuple[int, int, int]
    mac_version: Optional[Tuple[int, int, int]]
    cuda_version: Optional[Tuple[int, int]]
    has_nvidia_gpu: bool
    has_amd_gpu: bool


def parse_major_minor(s: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"(\d+)\.(\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return None


def detect_cuda_version() -> Optional[Tuple[int, int]]:
    # Environment hints first
    for var in ("CUDA_VERSION", "LLAMA_CUDA_VERSION", "LLAMACPP_CUDA_VERSION"):
        v = os.environ.get(var)
        if v:
            parsed = parse_major_minor(v)
            if parsed:
                return parsed

    # nvcc --version is the most reliable
    if shutil.which("nvcc"):
        try:
            out = subprocess.check_output(
                ["nvcc", "--version"],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            pass
        else:
            m = re.search(r"release\s+(\d+)\.(\d+)", out)
            if m:
                try:
                    return int(m.group(1)), int(m.group(2))
                except ValueError:
                    pass

    # Fallback: parse CUDA Version from nvidia-smi output
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi"],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            pass
        else:
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
            if m:
                try:
                    return int(m.group(1)), int(m.group(2))
                except ValueError:
                    pass

    return None


def detect_env() -> EnvInfo:
    os_name = platform.system()
    arch_raw = platform.machine() or platform.processor() or ""
    arch = arch_raw.lower()
    if arch == "amd64":
        arch = "x86_64"

    py_ver = sys.version_info
    python_version = (py_ver.major, py_ver.minor, py_ver.micro)

    mac_version: Optional[Tuple[int, int, int]] = None
    if os_name == "Darwin":
        ver = platform.mac_ver()[0]
        if ver:
            parts = ver.split(".")
            try:
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                patch = int(parts[2]) if len(parts) > 2 else 0
                mac_version = (major, minor, patch)
            except ValueError:
                mac_version = None

    cuda_version = detect_cuda_version()
    has_nvidia_gpu = cuda_version is not None or bool(shutil.which("nvidia-smi"))
    has_amd_gpu = bool(shutil.which("rocminfo") or shutil.which("hipcc"))

    return EnvInfo(
        os=os_name,
        arch=arch,
        python_version=python_version,
        mac_version=mac_version,
        cuda_version=cuda_version,
        has_nvidia_gpu=has_nvidia_gpu,
        has_amd_gpu=has_amd_gpu,
    )


def detect_x86_avx_level(env: EnvInfo) -> str:
    """
    Best-effort detection of AVX support level on x86_64:
        returns one of: "BASIC", "AVX", "AVX2", "AVX512" (case-sensitive).
    """
    if not env.arch.startswith("x86"):
        return "basic"

    flags = ""
    if env.os == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                data = f.read()
            m = re.search(r"flags\s*: (.+)", data)
            if m:
                flags = m.group(1).lower()
        except OSError:
            flags = ""
    elif env.os == "Darwin":
        out_parts: List[str] = []
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", key],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                out_parts.append(out)
            except Exception:
                pass
        flags = " ".join(out_parts).lower()
    else:
        flags = ""

    if "avx512" in flags:
        return "AVX512"
    if "avx2" in flags:
        return "AVX2"
    if "avx" in flags:
        return "AVX"
    return "basic"


def get_avx_level(env: EnvInfo) -> str:
    override = os.environ.get("LLAMACPP_AVX_LEVEL") or os.environ.get(
        "LLAMA_CPP_PYTHON_AVX_LEVEL"
    )
    if override:
        override = override.upper()
        if override in ("BASIC", "AVX", "AVX2", "AVX512"):
            return override
    return detect_x86_avx_level(env)


def pick_official_cuda_tag(env: EnvInfo) -> Optional[str]:
    """
    Pick the best official CUDA tag given detected CUDA version.
    """
    override = os.environ.get("LLAMACPP_CUDA_TAG") or os.environ.get(
        "LLAMA_CPP_PYTHON_CUDA_TAG"
    )
    if override:
        return override

    v = env.cuda_version
    if not v:
        return None
    if v in OFFICIAL_CUDA_TAGS:
        return OFFICIAL_CUDA_TAGS[v]
    if v[0] == 12:
        # choose nearest supported minor version
        candidates = sorted(
            OFFICIAL_CUDA_TAGS.keys(),
            key=lambda k: abs((k[0] - v[0]) * 10 + (k[1] - v[1])),
        )
        if candidates:
            return OFFICIAL_CUDA_TAGS[candidates[0]]
    return None


def pick_jllllll_cuda_tag(env: EnvInfo) -> Optional[str]:
    """
    Pick the best jllllll CUDA tag given detected CUDA version.
    """
    override = os.environ.get("LLAMACPP_JLLLLLL_CUDA_TAG")
    if override:
        return override

    v = env.cuda_version
    if not v:
        return None
    if v in JLLLLLL_CUDA_TAGS:
        return JLLLLLL_CUDA_TAGS[v]
    if v[0] == 12:
        candidates = sorted(
            JLLLLLL_CUDA_TAGS.keys(),
            key=lambda k: abs((k[0] - v[0]) * 10 + (k[1] - v[1])),
        )
        if candidates:
            return JLLLLLL_CUDA_TAGS[candidates[0]]
    return None


def supports_official_cuda(env: EnvInfo) -> bool:
    """
    Official CUDA wheels: Python 3.10–3.12 and CUDA 12.1–12.5 (as of docs).
    """
    py_major, py_minor, _ = env.python_version
    if (py_major, py_minor) < (3, 10) or (py_major, py_minor) > (3, 12):
        return False
    if env.cuda_version is None:
        return False
    tag = pick_official_cuda_tag(env)
    return tag is not None


def supports_official_metal(env: EnvInfo) -> bool:
    """
    Official Metal wheels: macOS 11+ and Python 3.10–3.12 (as of docs).
    """
    if env.os != "Darwin":
        return False
    if not env.mac_version:
        return False
    if env.mac_version < (11, 0, 0):
        return False
    py_major, py_minor, _ = env.python_version
    if (py_major, py_minor) < (3, 10) or (py_major, py_minor) > (3, 12):
        return False
    return True


def supports_jllllll_cuda(env: EnvInfo) -> bool:
    """
    Community CUDA wheels (jllllll):
        - OS: Windows x64, Linux x64, macOS 11+.
        - Arch: x86_64 only.
        - Python: 3.8–3.11.
        - CUDA: 11.6–12.2.
    """
    if env.cuda_version is None:
        return False
    if env.arch != "x86_64":
        return False
    if env.os not in ("Linux", "Windows", "Darwin"):
        return False
    py_major, py_minor, _ = env.python_version
    if py_major != 3 or py_minor < 8 or py_minor > 11:
        return False
    v = env.cuda_version
    # CUDA 11.6–12.2
    if v[0] == 11 and v[1] >= 6:
        return True
    if v[0] == 12 and v[1] <= 2:
        return True
    return False


def supports_jllllll_cpu(env: EnvInfo) -> bool:
    """
    Community CPU-only wheels (jllllll):
        - OS: Windows x64, Linux x64, macOS 11+.
        - Arch: x86_64 only.
        - Python: 3.8–3.11.
    """
    if env.arch != "x86_64":
        return False
    if env.os not in ("Linux", "Windows", "Darwin"):
        return False
    py_major, py_minor, _ = env.python_version
    if py_major != 3 or py_minor < 8 or py_minor > 11:
        return False
    return True


def dist_name_from_spec(project_spec: str) -> str:
    base = project_spec.split("==", 1)[0]
    base = base.split("[", 1)[0]
    return base


def run_pip_install(
    project_spec: str,
    extra_index_url: Optional[str] = None,
    only_binary: bool = False,
    prefer_binary: bool = True,
    force_reinstall: bool = False,
    env_overrides: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> bool:
    cmd: List[str] = [sys.executable, "-m", "pip", "install", project_spec]
    if prefer_binary:
        cmd.append("--prefer-binary")
    if only_binary:
        dist_name = dist_name_from_spec(project_spec)
        cmd.extend(["--only-binary", dist_name])
    if extra_index_url:
        cmd.extend(["--extra-index-url", extra_index_url])
    if force_reinstall:
        cmd.extend(["--upgrade", "--force-reinstall", "--no-cache-dir"])

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    print(f"-> Running: {' '.join(shlex.quote(c) for c in cmd)}")
    if env_overrides:
        for k, v in env_overrides.items():
            print(f"   env {k}={v}")
    if dry_run:
        print("   (dry run, not executing)")
        return False

    proc = subprocess.run(cmd, env=env)
    return proc.returncode == 0


def build_project_spec(
    base_name: str,
    extras: Optional[str],
    version: Optional[str],
) -> str:
    name = base_name
    if extras:
        extras_norm = ",".join(
            e.strip() for e in extras.split(",") if e.strip()
        )
        if extras_norm:
            name = f"{name}[{extras_norm}]"
    if version:
        return f"{name}=={version}"
    return name


def try_official_cpu_wheel(env: EnvInfo, project_spec: str, args) -> bool:
    py_major, py_minor, _ = env.python_version
    if (py_major, py_minor) < (3, 8):
        print("Skipping official CPU wheel: Python < 3.8.")
        return False
    print("Trying official CPU prebuilt wheel index...")
    return run_pip_install(
        project_spec,
        extra_index_url=OFFICIAL_CPU_INDEX,
        only_binary=True,
        prefer_binary=True,
        force_reinstall=args.force_reinstall,
        dry_run=args.dry_run,
    )


def try_official_cuda_wheel(env: EnvInfo, project_spec: str, args) -> bool:
    if not supports_official_cuda(env):
        print(
            "Skipping official CUDA wheels: environment does not meet "
            "documented requirements."
        )
        return False
    cuda_tag = pick_official_cuda_tag(env)
    if not cuda_tag:
        print("Could not determine official CUDA tag for this environment.")
        return False
    index = OFFICIAL_CUDA_INDEX_TEMPLATE.format(cuda_tag=cuda_tag)
    print(f"Trying official CUDA prebuilt wheel index '{index}'...")
    return run_pip_install(
        project_spec,
        extra_index_url=index,
        only_binary=True,
        prefer_binary=True,
        force_reinstall=args.force_reinstall,
        dry_run=args.dry_run,
    )


def try_official_metal_wheel(env: EnvInfo, project_spec: str, args) -> bool:
    if not supports_official_metal(env):
        print(
            "Skipping official Metal wheels: environment does not meet "
            "documented requirements."
        )
        return False
    print(f"Trying official Metal prebuilt wheel index '{OFFICIAL_METAL_INDEX}'...")
    return run_pip_install(
        project_spec,
        extra_index_url=OFFICIAL_METAL_INDEX,
        only_binary=True,
        prefer_binary=True,
        force_reinstall=args.force_reinstall,
        dry_run=args.dry_run,
    )


def try_jllllll_cuda_wheel(env: EnvInfo, project_spec: str, args) -> bool:
    if not supports_jllllll_cuda(env):
        print(
            "Skipping jllllll CUDA wheels: environment does not meet their "
            "documented requirements."
        )
        return False
    cuda_tag = pick_jllllll_cuda_tag(env)
    if not cuda_tag:
        print("Could not determine jllllll CUDA tag for this environment.")
        return False
    avx = get_avx_level(env)
    index = f"{JLLLLLL_BASE_INDEX}/{avx}/{cuda_tag}"
    print(f"Trying community CUDA wheels from '{index}'...")
    return run_pip_install(
        project_spec,
        extra_index_url=index,
        only_binary=True,
        prefer_binary=True,
        force_reinstall=args.force_reinstall,
        dry_run=args.dry_run,
    )


def try_jllllll_cpu_wheel(env: EnvInfo, project_spec: str, args) -> bool:
    if not supports_jllllll_cpu(env):
        print(
            "Skipping jllllll CPU wheels: environment does not meet their "
            "documented requirements."
        )
        return False
    avx = get_avx_level(env)
    index = f"{JLLLLLL_BASE_INDEX}/{avx}/cpu"
    print(f"Trying community CPU wheels from '{index}'...")
    return run_pip_install(
        project_spec,
        extra_index_url=index,
        only_binary=True,
        prefer_binary=True,
        force_reinstall=args.force_reinstall,
        dry_run=args.dry_run,
    )


def try_source_cpu(env: EnvInfo, project_spec: str, args) -> bool:
    print("Falling back to building from source (CPU-only)...")
    return run_pip_install(
        project_spec,
        extra_index_url=None,
        only_binary=False,
        prefer_binary=False,
        force_reinstall=args.force_reinstall,
        dry_run=args.dry_run,
    )


def merge_cmake_args(existing: str, extra: str) -> str:
    existing = (existing or "").strip()
    if not existing:
        return extra
    return existing + " " + extra


def try_source_cuda(env: EnvInfo, project_spec: str, args) -> bool:
    if env.cuda_version is None:
        print("Skipping source CUDA build: no CUDA detected.")
        return False
    print("Trying to build from source with CUDA support (GGML_CUDA=on)...")
    cmake_args = merge_cmake_args(
        os.environ.get("CMAKE_ARGS", ""),
        "-DGGML_CUDA=on",
    )
    env_overrides = {
        "CMAKE_ARGS": cmake_args,
        "FORCE_CMAKE": "1",
    }
    return run_pip_install(
        project_spec,
        extra_index_url=None,
        only_binary=False,
        prefer_binary=False,
        force_reinstall=args.force_reinstall,
        env_overrides=env_overrides,
        dry_run=args.dry_run,
    )


def try_source_metal(env: EnvInfo, project_spec: str, args) -> bool:
    if env.os != "Darwin":
        print("Skipping source Metal build: not on macOS.")
        return False
    print("Trying to build from source with Metal support (GGML_METAL=on)...")
    extra_flags = "-DGGML_METAL=on"
    # For Apple Silicon specifically, hint at arm64 build flags
    if env.arch == "arm64":
        extra_flags += (
            " -DCMAKE_OSX_ARCHITECTURES=arm64"
            " -DCMAKE_APPLE_SILICON_PROCESSOR=arm64"
        )
    cmake_args = merge_cmake_args(os.environ.get("CMAKE_ARGS", ""), extra_flags)
    env_overrides = {
        "CMAKE_ARGS": cmake_args,
        "FORCE_CMAKE": "1",
    }
    return run_pip_install(
        project_spec,
        extra_index_url=None,
        only_binary=False,
        prefer_binary=False,
        force_reinstall=args.force_reinstall,
        env_overrides=env_overrides,
        dry_run=args.dry_run,
    )


def try_source_rocm(env: EnvInfo, project_spec: str, args) -> bool:
    if not env.has_amd_gpu:
        print("Skipping source ROCm build: no AMD ROCm tooling detected.")
        return False
    print("Trying to build from source with ROCm hipBLAS support (GGML_HIPBLAS=on)...")
    cmake_args = merge_cmake_args(
        os.environ.get("CMAKE_ARGS", ""),
        "-DGGML_HIPBLAS=on",
    )
    env_overrides = {
        "CMAKE_ARGS": cmake_args,
        "FORCE_CMAKE": "1",
    }
    return run_pip_install(
        project_spec,
        extra_index_url=None,
        only_binary=False,
        prefer_binary=False,
        force_reinstall=args.force_reinstall,
        env_overrides=env_overrides,
        dry_run=args.dry_run,
    )


STRATEGY_FUNCS = {
    "official-cpu-prebuilt": try_official_cpu_wheel,
    "official-cuda-prebuilt": try_official_cuda_wheel,
    "official-metal-prebuilt": try_official_metal_wheel,
    "jllllll-cuda-prebuilt": try_jllllll_cuda_wheel,
    "jllllll-cpu-prebuilt": try_jllllll_cpu_wheel,
    "source-cpu": try_source_cpu,
    "source-cuda": try_source_cuda,
    "source-metal": try_source_metal,
    "source-rocm": try_source_rocm,
}


def detect_default_backend(env: EnvInfo) -> str:
    """
    Auto mode priority:
        1. CUDA if any CUDA detected.
        2. Metal on macOS 11+.
        3. ROCm if AMD ROCm tooling detected.
        4. CPU as fallback.
    """
    if env.cuda_version is not None:
        return "cuda"
    if env.os == "Darwin" and env.mac_version and env.mac_version >= (11, 0, 0):
        return "metal"
    if env.has_amd_gpu:
        return "rocm"
    return "cpu"


def build_strategies_for_backend(
    backend: str,
    allow_unofficial: bool,
) -> List[str]:
    strategies: List[str] = []
    if backend == "cuda":
        strategies.append("official-cuda-prebuilt")
        if allow_unofficial:
            strategies.append("jllllll-cuda-prebuilt")
        strategies.append("source-cuda")
        strategies.append("official-cpu-prebuilt")
        if allow_unofficial:
            strategies.append("jllllll-cpu-prebuilt")
        strategies.append("source-cpu")
    elif backend == "metal":
        strategies.append("official-metal-prebuilt")
        strategies.append("source-metal")
        strategies.append("official-cpu-prebuilt")
        if allow_unofficial:
            strategies.append("jllllll-cpu-prebuilt")
        strategies.append("source-cpu")
    elif backend == "rocm":
        strategies.append("source-rocm")
        strategies.append("official-cpu-prebuilt")
        if allow_unofficial:
            strategies.append("jllllll-cpu-prebuilt")
        strategies.append("source-cpu")
    else:  # cpu
        strategies.append("official-cpu-prebuilt")
        if allow_unofficial:
            strategies.append("jllllll-cpu-prebuilt")
        strategies.append("source-cpu")

    # De-duplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for s in strategies:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install llama-cpp-python for the current environment, "
            "preferring prebuilt wheels (CUDA, Metal, CPU) when possible, "
            "otherwise falling back to a source build."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "cuda", "metal", "rocm"],
        default="auto",
        help="Which backend to target (default: auto-detect).",
    )
    parser.add_argument(
        "--version",
        help="Specific llama-cpp-python version to install, e.g. 0.3.16.",
    )
    parser.add_argument(
        "--extras",
        help="Comma-separated pip extras to enable, e.g. 'server' or 'server,all'.",
    )
    parser.add_argument(
        "--allow-unofficial-wheels",
        action="store_true",
        help=(
            "Also try community wheels from "
            "jllllll/llama-cpp-python-cuBLAS-wheels."
        ),
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Use --upgrade --force-reinstall --no-cache-dir when installing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually running pip.",
    )

    args = parser.parse_args(argv)

    env = detect_env()
    print("Detected environment:")
    print(f"  OS:            {env.os}")
    print(f"  Arch:          {env.arch}")
    print(
        f"  Python:        "
        f"{env.python_version[0]}.{env.python_version[1]}.{env.python_version[2]}"
    )
    if env.mac_version:
        print(f"  macOS:         {'.'.join(str(x) for x in env.mac_version)}")
    if env.cuda_version:
        print(f"  CUDA:          {env.cuda_version[0]}.{env.cuda_version[1]}")
    else:
        print("  CUDA:          not detected")
    print(f"  NVIDIA GPU:    {env.has_nvidia_gpu}")
    print(f"  AMD ROCm GPU:  {env.has_amd_gpu}")

    backend = args.backend
    if backend == "auto":
        backend = detect_default_backend(env)
        print(f"Auto-detected preferred backend: {backend}")

    if backend == "cpu" and args.backend != "cpu":
        # For example, auto-detected backend may be CPU even if user requested auto.
        print("No GPU backend detected; falling back to CPU.")

    project_spec = build_project_spec(
        "llama-cpp-python",
        args.extras,
        args.version,
    )
    strategies = build_strategies_for_backend(
        backend,
        args.allow_unofficial_wheels,
    )

    print(f"\nInstall target: {project_spec}")
    print(f"Strategy order: {', '.join(strategies)}\n")

    for name in strategies:
        func = STRATEGY_FUNCS[name]
        print(f"=== Strategy: {name} ===")
        success = func(env, project_spec, args)
        if success:
            print(f"\nInstall succeeded using strategy: {name}")
            return 0
        else:
            print(f"Strategy {name} did not succeed.\n")

    print("All installation strategies failed. Please inspect the errors above and install manually.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
