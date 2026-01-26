#!/usr/bin/env python3
"""
GPU/CuPy/spaCy Diagnostic Script
Run this to diagnose why spaCy can't find CuPy even though it's installed.

Usage:
    python diagnose_gpu.py
"""

import sys
import os

def section(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def main():
    section("1. PYTHON ENVIRONMENT")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"sys.path:")
    for p in sys.path[:5]:
        print(f"  - {p}")
    if len(sys.path) > 5:
        print(f"  ... and {len(sys.path) - 5} more")

    section("2. CUDA ENVIRONMENT VARIABLES")
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES']
    for var in cuda_vars:
        val = os.environ.get(var, '<not set>')
        # Truncate long values
        if len(val) > 80:
            val = val[:80] + '...'
        print(f"{var}: {val}")

    section("3. NVIDIA-SMI")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"nvidia-smi failed: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi not found in PATH")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

    section("4. TORCH (if installed)")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed")
    except Exception as e:
        print(f"PyTorch error: {e}")

    section("5. CUPY - DETAILED CHECK")
    print("Checking for various CuPy packages...")

    cupy_packages = [
        'cupy',
        'cupy_cuda12x',
        'cupy_cuda121',
        'cupy_cuda120',
        'cupy_cuda11x',
        'cupy_cuda118',
        'cupy_cuda117',
    ]

    found_cupy = None
    for pkg in cupy_packages:
        try:
            mod = __import__(pkg)
            print(f"  ✓ {pkg}: {getattr(mod, '__version__', 'version unknown')}")
            if found_cupy is None:
                found_cupy = pkg
        except ImportError:
            print(f"  ✗ {pkg}: not installed")
        except Exception as e:
            print(f"  ? {pkg}: error - {e}")

    # Try importing cupy directly
    print("\nDirect CuPy import test:")
    try:
        import cupy as cp
        print(f"  ✓ import cupy as cp: SUCCESS")
        print(f"  CuPy version: {cp.__version__}")
        print(f"  CuPy file: {cp.__file__}")

        # Try a simple GPU operation
        print("\n  Testing GPU operation...")
        x = cp.array([1, 2, 3])
        y = cp.sum(x)
        print(f"  ✓ cp.sum([1,2,3]) = {y}")

        # Check CUDA runtime version CuPy sees
        print(f"\n  CuPy CUDA info:")
        print(f"    cuda.runtime.runtimeGetVersion: {cp.cuda.runtime.runtimeGetVersion()}")

    except ImportError as e:
        print(f"  ✗ import cupy failed: {e}")
    except Exception as e:
        print(f"  ✗ CuPy error: {e}")

    section("6. SPACY CHECK")
    try:
        import spacy
        print(f"spaCy version: {spacy.__version__}")
        print(f"spaCy file: {spacy.__file__}")

        # Check what spaCy thinks about GPU
        print("\nspaCy GPU detection:")
        try:
            from spacy.util import use_gpu, get_cuda_stream
            print(f"  use_gpu function exists: True")
        except ImportError as e:
            print(f"  use_gpu import failed: {e}")

        # Check if spacy.prefer_gpu() works
        print("\nTrying spacy.prefer_gpu()...")
        try:
            result = spacy.prefer_gpu()
            print(f"  spacy.prefer_gpu() returned: {result}")
        except Exception as e:
            print(f"  spacy.prefer_gpu() failed: {e}")

        # Check if spacy.require_gpu() works
        print("\nTrying spacy.require_gpu()...")
        try:
            result = spacy.require_gpu()
            print(f"  spacy.require_gpu() returned: {result}")
        except Exception as e:
            print(f"  ✗ spacy.require_gpu() failed: {e}")

    except ImportError:
        print("spaCy not installed")
    except Exception as e:
        print(f"spaCy error: {e}")

    section("7. SPACY'S CUPY DETECTION (the actual issue)")
    print("Checking what spaCy sees when it tries to import CuPy...")
    try:
        # This is what spaCy does internally
        from thinc.api import require_gpu, prefer_gpu
        print("  ✓ thinc.api imports work")

        # Check thinc's GPU config
        from thinc.util import has_cupy, get_cupy
        print(f"  thinc.util.has_cupy(): {has_cupy()}")

        if has_cupy():
            cp = get_cupy()
            print(f"  thinc.util.get_cupy() returned: {type(cp)}")
        else:
            print("  thinc thinks CuPy is NOT available")

            # Let's see why
            print("\n  Debugging thinc's CuPy detection...")
            try:
                from thinc.backends import cupy_ops
                print("  ✓ thinc.backends.cupy_ops imported")
            except ImportError as e:
                print(f"  ✗ thinc.backends.cupy_ops import failed: {e}")

    except ImportError as e:
        print(f"  thinc import failed: {e}")
    except Exception as e:
        print(f"  thinc error: {e}")

    section("8. INSTALLED PACKAGES (relevant)")
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            relevant = ['cupy', 'spacy', 'thinc', 'torch', 'cuda', 'nvidia']
            print("Packages containing: cupy, spacy, thinc, torch, cuda, nvidia")
            for line in lines:
                if any(r in line.lower() for r in relevant):
                    print(f"  {line}")
    except Exception as e:
        print(f"pip list failed: {e}")

    section("9. RECOMMENDATIONS")
    print("""
Based on the output above, common fixes are:

1. If CuPy works but thinc/spaCy can't find it:
   pip install --upgrade thinc[cuda12x]
   # or for CUDA 11.x:
   pip install --upgrade thinc[cuda11x]

2. If CuPy version doesn't match CUDA:
   pip uninstall cupy-cuda12x cupy-cuda11x cupy  # remove all
   pip install cupy-cuda12x  # match your CUDA version

3. If spaCy is outdated:
   pip install --upgrade spacy

4. Nuclear option - reinstall everything:
   pip install --upgrade spacy[cuda12x] thinc[cuda12x]

5. If nothing works, skip GPU for spaCy:
   # In your code, don't call spacy.require_gpu()
   # Just use: nlp = spacy.load("en_core_web_trf")
   # It will use CPU, which is slower but works
""")

    section("DONE")
    print("Copy all output above and share it for further diagnosis.")

if __name__ == "__main__":
    main()
