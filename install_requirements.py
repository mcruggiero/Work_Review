#!/usr/bin/env python3
"""
ScoreCard Pipeline - Requirements Installer & Checker

This script checks all dependencies and installs missing Python packages.
It will NOT modify system-wide installations (CUDA, Python, etc.)

Usage:
    python install_requirements.py           # Check only (dry run)
    python install_requirements.py --install # Check and install missing packages

Outputs clear errors that can be shared for troubleshooting.
"""

import sys
import os
import subprocess
import argparse
from typing import Optional, Tuple, List

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum versions required
MIN_PYTHON_VERSION = (3, 9)
MIN_CUDA_VERSION = 11  # Major version

# Required packages with minimum versions (None = any version)
# Format: (package_name, import_name, min_version, pip_install_name)
REQUIRED_PACKAGES = [
    # Core data science
    ("numpy", "numpy", "1.20.0", "numpy"),
    ("pandas", "pandas", "1.3.0", "pandas"),
    ("scikit-learn", "sklearn", "1.0.0", "scikit-learn"),

    # NLP
    ("spacy", "spacy", "3.5.0", "spacy"),
    ("beautifulsoup4", "bs4", None, "beautifulsoup4"),
    ("tiktoken", "tiktoken", None, "tiktoken"),

    # ML / Deep Learning
    ("torch", "torch", "2.0.0", "torch"),
    ("sentence-transformers", "sentence_transformers", None, "sentence-transformers"),

    # Database / Search
    ("pyodbc", "pyodbc", None, "pyodbc"),
    # Pin to 8.13.0 - must match local Elasticsearch server version
    ("elasticsearch", "elasticsearch", "8.13.0", "elasticsearch==8.13.0"),

    # API clients
    ("openai", "openai", "1.0.0", "openai"),

    # Thinc (spaCy backend) - with CUDA support
    ("thinc", "thinc", "8.1.0", "thinc"),
]

# Optional but recommended for GPU
GPU_PACKAGES = [
    ("cupy-cuda12x", "cupy", None, "cupy-cuda12x"),
]

# spaCy model required
SPACY_MODEL = "en_core_web_trf"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def color(text: str, c: str) -> str:
    """Wrap text in color codes"""
    return f"{c}{text}{Colors.END}"

def section(title: str):
    """Print a section header"""
    print(f"\n{color('=' * 70, Colors.BLUE)}")
    print(f" {color(title, Colors.BOLD)}")
    print(f"{color('=' * 70, Colors.BLUE)}")

def ok(msg: str):
    print(f"  {color('✓', Colors.GREEN)} {msg}")

def warn(msg: str):
    print(f"  {color('⚠', Colors.YELLOW)} {msg}")

def error(msg: str):
    print(f"  {color('✗', Colors.RED)} {msg}")

def info(msg: str):
    print(f"  {color('ℹ', Colors.BLUE)} {msg}")

def run_pip(args: List[str], capture: bool = True) -> Tuple[int, str, str]:
    """Run pip command and return (returncode, stdout, stderr)"""
    cmd = [sys.executable, "-m", "pip"] + args
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode, result.stdout, result.stderr
    else:
        result = subprocess.run(cmd, timeout=300)
        return result.returncode, "", ""

def get_package_version(import_name: str) -> Optional[str]:
    """Get installed version of a package"""
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None

def version_tuple(v) -> Tuple:
    """Convert version string to tuple for comparison.

    Handles:
    - Standard versions: "1.2.3" -> (1, 2, 3)
    - Local versions: "2.1.2+cu121" -> (2, 1, 2)
    - Tuples (elasticsearch): (8, 13, 0) -> (8, 13, 0)
    """
    # If already a tuple (some packages return tuple for __version__)
    if isinstance(v, tuple):
        return v[:3]

    try:
        # Strip local version suffix (e.g., +cu121, +cpu)
        v = str(v).split("+")[0]
        # Strip pre-release tags (e.g., -beta, -rc1)
        v = v.split("-")[0]
        return tuple(int(x) for x in v.split(".")[:3])
    except:
        return (0, 0, 0)

# =============================================================================
# CHECK FUNCTIONS
# =============================================================================

def check_python() -> bool:
    """Check Python version"""
    section("1. PYTHON VERSION")

    current = sys.version_info[:2]
    if current >= MIN_PYTHON_VERSION:
        ok(f"Python {current[0]}.{current[1]} (>= {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} required)")
        return True
    else:
        error(f"Python {current[0]}.{current[1]} - NEED {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+")
        error("Cannot fix: Python is a system-wide installation")
        return False

def check_cuda() -> Tuple[bool, Optional[int]]:
    """Check CUDA availability (read-only, no modifications)"""
    section("2. CUDA / GPU (System-wide - Read Only)")

    cuda_version = None

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')[0]
            ok(f"GPU detected: {gpu_info}")
        else:
            warn("nvidia-smi failed - no GPU or drivers not installed")
            return False, None
    except FileNotFoundError:
        warn("nvidia-smi not found - no GPU or drivers not installed")
        return False, None
    except Exception as e:
        warn(f"GPU check error: {e}")
        return False, None

    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = int(torch.version.cuda.split('.')[0])
            ok(f"PyTorch CUDA: {torch.version.cuda}")
            ok(f"GPU count: {torch.cuda.device_count()}")
        else:
            warn("PyTorch installed but CUDA not available")
    except ImportError:
        info("PyTorch not yet installed (will check later)")
    except Exception as e:
        warn(f"PyTorch CUDA check error: {e}")

    # Check CUDA environment
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        ok(f"CUDA_HOME: {cuda_home}")
    else:
        info("CUDA_HOME not set (may still work via PyTorch)")

    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'nvidia' in ld_path or 'cuda' in ld_path.lower():
        ok(f"LD_LIBRARY_PATH includes NVIDIA/CUDA paths")

    return True, cuda_version

def format_version(v) -> str:
    """Format version for display (handles tuples and strings)"""
    if isinstance(v, tuple):
        return ".".join(str(x) for x in v)
    return str(v)

def check_packages(install: bool = False) -> Tuple[List[str], List[str]]:
    """Check required Python packages"""
    section("3. PYTHON PACKAGES")

    missing = []
    outdated = []
    install_commands = []

    for pkg_name, import_name, min_ver, pip_name in REQUIRED_PACKAGES:
        version = get_package_version(import_name)
        version_display = format_version(version) if version else None

        if version is None:
            error(f"{pkg_name}: NOT INSTALLED")
            missing.append(pkg_name)
            install_commands.append(pip_name)
        elif min_ver and version_tuple(version) < version_tuple(min_ver):
            warn(f"{pkg_name}: {version_display} (need >= {min_ver})")
            outdated.append(pkg_name)
            install_commands.append(f"{pip_name}>={min_ver}")
        else:
            ok(f"{pkg_name}: {version_display}")

    # Install if requested
    if install and install_commands:
        print(f"\n  {color('Installing/upgrading packages...', Colors.BOLD)}")
        for pkg in install_commands:
            info(f"pip install --user --upgrade {pkg}")
            code, out, err = run_pip(["install", "--user", "--upgrade", pkg], capture=False)
            if code != 0:
                error(f"Failed to install {pkg}")
                if err:
                    print(f"    {err[:200]}")
    elif install_commands:
        print(f"\n  {color('To install missing packages, run:', Colors.YELLOW)}")
        print(f"    pip install --user --upgrade {' '.join(install_commands)}")

    return missing, outdated

def check_gpu_packages(install: bool = False, cuda_version: Optional[int] = None) -> bool:
    """Check GPU-specific packages (CuPy, thinc CUDA)"""
    section("4. GPU PACKAGES (for spaCy acceleration)")

    all_ok = True

    # Determine CUDA version suffix
    if cuda_version and cuda_version >= 12:
        cuda_suffix = "cuda12x"
    elif cuda_version and cuda_version >= 11:
        cuda_suffix = "cuda11x"
    else:
        cuda_suffix = "cuda12x"  # Default guess
        info(f"Assuming CUDA 12.x (detected: {cuda_version or 'unknown'})")

    # Check CuPy
    cupy_version = get_package_version("cupy")
    if cupy_version:
        ok(f"CuPy: {cupy_version}")

        # Test if it actually works
        try:
            import cupy as cp
            x = cp.array([1, 2, 3])
            _ = cp.sum(x)
            ok("CuPy GPU test: PASSED")
        except Exception as e:
            error(f"CuPy GPU test: FAILED - {e}")
            all_ok = False
    else:
        error("CuPy: NOT INSTALLED")
        all_ok = False
        if install:
            info(f"pip install --user cupy-{cuda_suffix}")
            run_pip(["install", "--user", f"cupy-{cuda_suffix}"], capture=False)
        else:
            print(f"    {color(f'pip install --user cupy-{cuda_suffix}', Colors.YELLOW)}")

    # Check thinc CUDA support
    print()
    info("Checking thinc (spaCy backend) CUDA support...")
    try:
        import spacy
        result = spacy.prefer_gpu()
        if result:
            ok("spacy.prefer_gpu() returned True - GPU acceleration available!")
        else:
            warn("spacy.prefer_gpu() returned False - thinc CUDA not configured")
            all_ok = False

            thinc_cuda_pkg = f"thinc[{cuda_suffix}]"
            if install:
                info(f"Installing thinc with CUDA support...")
                info(f"pip install --user --upgrade '{thinc_cuda_pkg}'")
                code, _, _ = run_pip(["install", "--user", "--upgrade", thinc_cuda_pkg], capture=False)
                if code == 0:
                    ok("thinc CUDA installed - please restart Python and re-run this script")
            else:
                cmd = f"pip install --user --upgrade '{thinc_cuda_pkg}'"
                print(f"    {color(cmd, Colors.YELLOW)}")
    except Exception as e:
        error(f"spaCy GPU check failed: {e}")
        all_ok = False

    return all_ok

def check_spacy_model(install: bool = False) -> bool:
    """Check spaCy model is installed"""
    section("5. SPACY MODEL")

    try:
        import spacy
        try:
            nlp = spacy.load(SPACY_MODEL)
            ok(f"Model '{SPACY_MODEL}' is installed")
            return True
        except OSError:
            error(f"Model '{SPACY_MODEL}' NOT INSTALLED")
            if install:
                info(f"Downloading {SPACY_MODEL}...")
                code, _, _ = run_pip(["install", "--user", f"https://github.com/explosion/spacy-models/releases/download/{SPACY_MODEL}-3.8.0/{SPACY_MODEL}-3.8.0-py3-none-any.whl"], capture=False)
                if code != 0:
                    # Try python -m spacy download
                    subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL])
            else:
                print(f"    {color(f'python -m spacy download {SPACY_MODEL}', Colors.YELLOW)}")
            return False
    except ImportError:
        error("spaCy not installed - cannot check model")
        return False

def check_config_files() -> bool:
    """Check that required config files exist"""
    section("6. CONFIGURATION FILES")

    # Try to import the config to get resolved paths
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from scorecard.config import _resolve_path

        files = [
            ("Model Matrix", _resolve_path("prompts/model_matrix.json")),
            ("SQL Query", _resolve_path("prompts/sql_query.txt")),
            ("GPT Prompt", _resolve_path("prompts/GPT_Prompt.txt")),
        ]

        all_ok = True
        for name, path in files:
            if os.path.exists(path):
                ok(f"{name}: {path}")
            else:
                error(f"{name}: NOT FOUND at {path}")
                all_ok = False

        return all_ok
    except ImportError as e:
        warn(f"Could not import scorecard.config: {e}")

        # Fallback: check relative paths
        base = os.path.dirname(os.path.abspath(__file__))
        files = [
            ("Model Matrix", os.path.join(base, "prompts", "model_matrix.json")),
            ("SQL Query", os.path.join(base, "prompts", "sql_query.txt")),
            ("GPT Prompt", os.path.join(base, "prompts", "GPT_Prompt.txt")),
        ]

        all_ok = True
        for name, path in files:
            if os.path.exists(path):
                ok(f"{name}: {path}")
            else:
                error(f"{name}: NOT FOUND at {path}")
                all_ok = False

        return all_ok

def check_connectivity() -> bool:
    """Check database and API connectivity (optional)"""
    section("7. CONNECTIVITY (Optional)")

    info("Skipping connectivity checks - will fail gracefully at runtime")
    info("Required services: SQL Server, Elasticsearch, OpenAI API")
    return True

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ScoreCard Pipeline Requirements Checker")
    parser.add_argument("--install", action="store_true",
                       help="Install missing packages (uses --user flag, no sudo needed)")
    args = parser.parse_args()

    print(color("\n╔══════════════════════════════════════════════════════════════════════╗", Colors.BOLD))
    print(color("║          SCORECARD PIPELINE - REQUIREMENTS CHECK                     ║", Colors.BOLD))
    print(color("╚══════════════════════════════════════════════════════════════════════╝", Colors.BOLD))

    if args.install:
        print(color("\n  MODE: Install missing packages (--user flag, no sudo)", Colors.GREEN))
    else:
        print(color("\n  MODE: Check only (use --install to fix issues)", Colors.YELLOW))

    errors = []
    warnings = []

    # 1. Python version (cannot fix)
    if not check_python():
        errors.append("Python version too old - cannot fix automatically")

    # 2. CUDA/GPU (cannot fix, just report)
    gpu_ok, cuda_version = check_cuda()
    if not gpu_ok:
        warnings.append("GPU not available - will use CPU (slower)")

    # 3. Python packages (can fix)
    missing, outdated = check_packages(install=args.install)
    if missing:
        errors.append(f"Missing packages: {', '.join(missing)}")
    if outdated:
        warnings.append(f"Outdated packages: {', '.join(outdated)}")

    # 4. GPU packages (can fix)
    if gpu_ok:
        if not check_gpu_packages(install=args.install, cuda_version=cuda_version):
            warnings.append("GPU packages not fully configured - spaCy will use CPU")
    else:
        info("Skipping GPU package checks (no GPU detected)")

    # 5. spaCy model (can fix)
    if not check_spacy_model(install=args.install):
        errors.append(f"spaCy model '{SPACY_MODEL}' not installed")

    # 6. Config files (cannot fix, just report)
    if not check_config_files():
        errors.append("Configuration files missing")

    # 7. Connectivity
    check_connectivity()

    # =============================================================================
    # SUMMARY
    # =============================================================================
    section("SUMMARY")

    if not errors and not warnings:
        print(color("\n  ✓ ALL CHECKS PASSED - Ready to run pipeline!\n", Colors.GREEN))
        return 0

    if warnings:
        print(color(f"\n  ⚠ WARNINGS ({len(warnings)}):", Colors.YELLOW))
        for w in warnings:
            print(f"    - {w}")

    if errors:
        print(color(f"\n  ✗ ERRORS ({len(errors)}):", Colors.RED))
        for e in errors:
            print(f"    - {e}")

        if not args.install:
            print(color("\n  To fix issues, run:", Colors.BOLD))
            print(f"    python {os.path.basename(__file__)} --install")

        return 1

    print(color("\n  Pipeline should work, but with reduced functionality.", Colors.YELLOW))
    return 0

if __name__ == "__main__":
    sys.exit(main())
