#!/usr/bin/env python3
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from urllib.error import HTTPError, URLError

GITHUB_API_BASE = "https://api.github.com/repos/vllm-project/vllm"
PYPI_JSON_URL = "https://pypi.org/pypi/vllm/json"
PYPI_PACKAGE_NAME = "vllm"


@dataclass
class EnvInfo:
    python_major: int
    python_minor: int
    python_tag: str
    arch: str
    cuda_version: Optional[str]  # e.g. "12.9"
    cuda_tag: Optional[str]      # e.g. "129"


@dataclass
class WheelInfo:
    name: str
    url: str
    version: str
    cuda_tag: Optional[str]
    python_tag: str
    abi_tag: str
    platform_tag: str
    source: str  # "pypi" or "github"


def http_get_json(url: str) -> dict:
    headers = {"User-Agent": "vllm-auto-installer/0.1"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def detect_cuda_version() -> Optional[str]:
    # Try PyTorch first
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore
    else:
        v = getattr(torch.version, "cuda", None)
        if v:
            m = re.match(r"(\d+)\.(\d+)", str(v))
            if m:
                major = int(m.group(1))
                minor = int(m.group(2))
                return f"{major}.{minor}"

    # Try nvcc
    for cmd in (["nvcc", "--version"], ["nvcc", "-V"]):
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        if proc.returncode == 0:
            m = re.search(r"release\s+(\d+)\.(\d+)", proc.stdout)
            if m:
                major = int(m.group(1))
                minor = int(m.group(2))
                return f"{major}.{minor}"

    # Try nvidia-smi
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if proc.returncode == 0:
        m = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", proc.stdout)
        if m:
            major = int(m.group(1))
            minor = int(m.group(2))
            return f"{major}.{minor}"

    return None


def cuda_version_to_tag(version: str) -> str:
    m = re.match(r"(\d+)\.(\d+)", version.strip())
    if not m:
        raise RuntimeError(f"Unrecognized CUDA version string: {version!r}")
    major = int(m.group(1))
    minor = int(m.group(2))
    tag = major * 10 + minor
    return f"{tag:03d}"


def detect_env() -> EnvInfo:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("This script currently supports Linux only for GPU installs.")

    major, minor = sys.version_info[:2]
    if major != 3 or minor < 10:
        raise RuntimeError("vLLM GPU wheels currently require Python 3.10–3.13.")

    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        arch = "x86_64"
    elif arch in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        raise RuntimeError(
            f"Unsupported architecture {arch!r}. vLLM NVIDIA wheels are built for x86_64 and aarch64."
        )

    cuda_version = detect_cuda_version()
    if cuda_version is None:
        raise RuntimeError(
            "Could not detect a usable CUDA version (no CUDA-enabled torch, nvcc, or nvidia-smi). "
            "This script only handles NVIDIA GPU installs."
        )
    cuda_tag = cuda_version_to_tag(cuda_version)
    python_tag = f"cp{major}{minor}"

    return EnvInfo(
        python_major=major,
        python_minor=minor,
        python_tag=python_tag,
        arch=arch,
        cuda_version=cuda_version,
        cuda_tag=cuda_tag,
    )


def parse_wheel_filename(filename: str, url: str, source: str) -> Optional[WheelInfo]:
    if not filename.endswith(".whl"):
        return None
    name_no_ext = filename[:-4]
    parts = name_no_ext.split("-")
    if len(parts) < 5:
        return None
    dist = parts[0]
    if dist != PYPI_PACKAGE_NAME:
        return None
    version_part = parts[1]
    python_tag = parts[-3]
    abi_tag = parts[-2]
    platform_tag = parts[-1]

    version = version_part
    cuda_tag: Optional[str] = None
    if "+" in version_part:
        base, local = version_part.split("+", 1)
        version = base
        m = re.search(r"cu(\d+)", local)
        if m:
            cuda_tag = m.group(1)

    return WheelInfo(
        name=filename,
        url=url,
        version=version,
        cuda_tag=cuda_tag,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        source=source,
    )


def get_latest_pypi_version() -> str:
    data = http_get_json(PYPI_JSON_URL)
    info = data.get("info", {})
    version = info.get("version")
    if not version:
        raise RuntimeError("Could not determine latest vLLM version from PyPI metadata.")
    return str(version)


def get_pypi_wheels(version: str) -> List[WheelInfo]:
    data = http_get_json(PYPI_JSON_URL)
    releases = data.get("releases", {})
    if version not in releases:
        raise RuntimeError(f"Version {version} not found on PyPI for {PYPI_PACKAGE_NAME}.")
    wheels: List[WheelInfo] = []
    for file in releases[version]:
        if file.get("packagetype") != "bdist_wheel":
            continue
        filename = file.get("filename")
        url = file.get("url")
        if not filename or not url:
            continue
        wheel = parse_wheel_filename(filename, url, source="pypi")
        if wheel is not None:
            wheels.append(wheel)
    return wheels


def get_github_wheels(version: str) -> List[WheelInfo]:
    tag = f"v{version}"
    url = f"{GITHUB_API_BASE}/releases/tags/{tag}"
    data = http_get_json(url)
    assets = data.get("assets", [])
    wheels: List[WheelInfo] = []
    for asset in assets:
        name = asset.get("name") or ""
        if not name.endswith(".whl") or not name.startswith(f"{PYPI_PACKAGE_NAME}-"):
            continue
        dl_url = asset.get("browser_download_url")
        if not dl_url:
            continue
        wheel = parse_wheel_filename(name, dl_url, source="github")
        if wheel is not None:
            wheels.append(wheel)
    return wheels


def wheel_supports_env(wheel: WheelInfo, env: EnvInfo) -> bool:
    pt = wheel.platform_tag.lower()
    # OS
    if "linux" not in pt and "manylinux" not in pt:
        return False
    # Arch
    if env.arch == "x86_64":
        if not any(x in pt for x in ("x86_64", "amd64")):
            return False
    elif env.arch == "aarch64":
        if not any(x in pt for x in ("aarch64", "arm64")):
            return False
    # Python
    major = env.python_major
    minor = env.python_minor
    py_tag = wheel.python_tag
    abi_tag = wheel.abi_tag
    exact = f"cp{major}{minor}"
    if py_tag == exact:
        return True
    if py_tag.startswith("cp") and abi_tag == "abi3":
        try:
            base = int(py_tag[2:])
        except ValueError:
            return False
        if major == 3 and minor >= base:
            return True
    return False


def find_best_prebuilt_wheel(env: EnvInfo, version: str, verbose: bool = False) -> Optional[WheelInfo]:
    wheels: List[WheelInfo] = []
    # PyPI wheels
    try:
        pypi_wheels = get_pypi_wheels(version)
        if verbose:
            print(f"Found {len(pypi_wheels)} candidate wheels on PyPI for vllm=={version}.")
        wheels.extend(pypi_wheels)
    except (HTTPError, URLError, RuntimeError) as e:
        if verbose:
            print(f"Warning: failed to query PyPI for vllm=={version}: {e}")

    # GitHub wheels (CUDA-variant builds)
    try:
        gh_wheels = get_github_wheels(version)
        if verbose:
            print(f"Found {len(gh_wheels)} candidate wheels on GitHub for vllm=={version}.")
        wheels.extend(gh_wheels)
    except (HTTPError, URLError, RuntimeError) as e:
        if verbose:
            print(f"Warning: failed to query GitHub for vllm=={version}: {e}")

    if not wheels:
        if verbose:
            print("No wheels found on PyPI or GitHub for this version.")
        return None

    compatible = [w for w in wheels if wheel_supports_env(w, env)]
    if not compatible:
        if verbose:
            print("No wheels compatible with current Python/platform.")
        return None

    # Prefer exact CUDA match if we know the CUDA tag
    if env.cuda_tag:
        cuda_matches = [w for w in compatible if w.cuda_tag == env.cuda_tag]
        if cuda_matches:
            # Prefer GitHub CUDA-specific wheels, then PyPI, then name
            cuda_matches.sort(key=lambda w: (w.source != "github", w.source != "pypi", w.name))
            best = cuda_matches[0]
            if verbose:
                print(f"Selected wheel {best.name} with CUDA tag cu{best.cuda_tag}.")
            return best

    # Fallback: wheels without explicit CUDA tag (main variant)
    main_variant = [w for w in compatible if w.cuda_tag is None]
    if main_variant:
        main_variant.sort(key=lambda w: (w.source != "pypi", w.name))
        best = main_variant[0]
        if verbose:
            print(f"Selected default wheel {best.name} (no explicit CUDA tag).")
        return best

    # Final fallback: any compatible wheel
    compatible.sort(key=lambda w: (w.source != "github", w.name))
    best = compatible[0]
    if verbose:
        print(f"Selected wheel {best.name} (fallback selection).")
    return best


def get_torch_index_url(cuda_tag: str) -> str:
    return f"https://download.pytorch.org/whl/cu{cuda_tag}"


def run_subprocess(cmd: List[str], dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def install_prebuilt_wheel(wheel: WheelInfo, env: EnvInfo, dry_run: bool = False) -> None:
    if not env.cuda_tag:
        raise RuntimeError("CUDA tag is unknown; cannot choose PyTorch wheel index.")

    extra_index = get_torch_index_url(env.cuda_tag)
    cmd: List[str] = [sys.executable, "-m", "pip", "install", "-U"]

    if wheel.source == "github":
        target = wheel.url
    else:
        target = f"{PYPI_PACKAGE_NAME}=={wheel.version}"

    cmd.append(target)
    cmd.extend(["--extra-index-url", extra_index])

    run_subprocess(cmd, dry_run=dry_run)


def build_from_source(version: str, env: EnvInfo, dry_run: bool = False) -> None:
    if not env.cuda_tag:
        raise RuntimeError("CUDA tag is unknown; cannot choose PyTorch wheel index for source build.")

    extra_index = get_torch_index_url(env.cuda_tag)
    cmd: List[str] = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"{PYPI_PACKAGE_NAME}=={version}",
        "--no-binary",
        PYPI_PACKAGE_NAME,
        "--extra-index-url",
        extra_index,
    ]
    run_subprocess(cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Install vLLM for NVIDIA GPUs by selecting the best prebuilt wheel for the "
            "current environment, or building from source if no wheel exists."
        )
    )
    parser.add_argument(
        "--version",
        default="latest",
        help=(
            "vLLM version to install, e.g. 0.12.0. "
            "Default: latest version published on PyPI."
        ),
    )
    parser.add_argument(
        "--force-source",
        action="store_true",
        help="Always build vLLM from source (sdist) instead of using any prebuilt wheels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run, but do not execute them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging about wheel discovery and selection.",
    )
    args = parser.parse_args()

    env = detect_env()
    print(
        f"Detected environment: Python {env.python_major}.{env.python_minor} "
        f"on Linux/{env.arch}, CUDA {env.cuda_version} (cu{env.cuda_tag})."
    )

    if args.version == "latest":
        version = get_latest_pypi_version()
        if args.verbose:
            print(f"Latest vLLM version on PyPI: {version}")
    else:
        version = args.version.lstrip("v")

    print(f"Target vLLM version: {version}")

    if args.force_source:
        print("Installing vLLM by building from source (sdist).")
        build_from_source(version, env, dry_run=args.dry_run)
        return

    print("Searching for prebuilt vLLM wheels matching this environment...")
    wheel = find_best_prebuilt_wheel(env, version, verbose=args.verbose)

    if wheel is None:
        print("No matching prebuilt wheel found. Falling back to source build.")
        build_from_source(version, env, dry_run=args.dry_run)
        return

    cuda_desc = f"cu{wheel.cuda_tag}" if wheel.cuda_tag else "default CUDA variant"
    print(
        f"Using prebuilt wheel: {wheel.name} "
        f"({wheel.source}, CUDA variant: {cuda_desc})"
    )
    install_prebuilt_wheel(wheel, env, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
