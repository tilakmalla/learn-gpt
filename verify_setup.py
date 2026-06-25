"""
GPT Learning Project - Setup Verification Script
=================================================
Run this after setup to verify everything works correctly.

Usage:
    python verify_setup.py
"""

import sys
import os

def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print('='*60)

def print_check(name, passed, details=""):
    status = "✅" if passed else "❌"
    print(f"{status} {name}")
    if details:
        print(f"   {details}")

def main():
    print_header("GPT Learning Environment Verification")
    
    all_passed = True
    
    # -------------------------------------------------------------------------
    # Check 1: Python version
    # -------------------------------------------------------------------------
    py_version = sys.version_info
    py_ok = py_version.major >= 3 and py_version.minor >= 9
    print_check(
        "Python Version", 
        py_ok,
        f"Python {py_version.major}.{py_version.minor}.{py_version.micro}"
    )
    all_passed &= py_ok
    
    # -------------------------------------------------------------------------
    # Check 2: Virtual environment
    # -------------------------------------------------------------------------
    in_venv = sys.prefix != sys.base_prefix
    venv_path = sys.prefix if in_venv else "Not in venv!"
    print_check("Virtual Environment", in_venv, venv_path)
    all_passed &= in_venv
    
    # -------------------------------------------------------------------------
    # Check 3: PyTorch
    # -------------------------------------------------------------------------
    try:
        import torch
        torch_ok = True
        torch_version = torch.__version__
    except ImportError:
        torch_ok = False
        torch_version = "Not installed"
    print_check("PyTorch", torch_ok, f"Version: {torch_version}")
    all_passed &= torch_ok
    
    # -------------------------------------------------------------------------
    # Check 4: MPS (Metal Performance Shaders) for M1
    # -------------------------------------------------------------------------
    if torch_ok:
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        
        if mps_available:
            # Test actual computation
            try:
                device = torch.device("mps")
                x = torch.randn(100, 100, device=device)
                y = torch.matmul(x, x)
                mps_works = True
                mps_detail = "GPU acceleration ready!"
            except Exception as e:
                mps_works = False
                mps_detail = f"MPS error: {e}"
        else:
            mps_works = False
            mps_detail = f"Built: {mps_built}, Available: {mps_available}"
        
        print_check("MPS (M1 GPU)", mps_works, mps_detail)
        # MPS not required - CPU fallback is fine
    
    # -------------------------------------------------------------------------
    # Check 5: NumPy
    # -------------------------------------------------------------------------
    try:
        import numpy as np
        numpy_ok = True
        numpy_version = np.__version__
    except ImportError:
        numpy_ok = False
        numpy_version = "Not installed"
    print_check("NumPy", numpy_ok, f"Version: {numpy_version}")
    all_passed &= numpy_ok
    
    # -------------------------------------------------------------------------
    # Check 6: Matplotlib
    # -------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib_ok = True
        matplotlib_version = matplotlib.__version__
    except ImportError:
        matplotlib_ok = False
        matplotlib_version = "Not installed"
    print_check("Matplotlib", matplotlib_ok, f"Version: {matplotlib_version}")
    all_passed &= matplotlib_ok
    
    # -------------------------------------------------------------------------
    # Check 7: tqdm
    # -------------------------------------------------------------------------
    try:
        import tqdm
        tqdm_ok = True
        tqdm_version = tqdm.__version__
    except ImportError:
        tqdm_ok = False
        tqdm_version = "Not installed"
    print_check("tqdm", tqdm_ok, f"Version: {tqdm_version}")
    all_passed &= tqdm_ok
    
    # -------------------------------------------------------------------------
    # Check 8: Dataset
    # -------------------------------------------------------------------------
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "input.txt")
    dataset_exists = os.path.exists(dataset_path)
    
    if dataset_exists:
        with open(dataset_path, 'r') as f:
            text = f.read()
        dataset_detail = f"{len(text):,} characters, {len(set(text))} unique"
    else:
        dataset_detail = f"Expected at: {dataset_path}"
    
    print_check("Shakespeare Dataset", dataset_exists, dataset_detail)
    all_passed &= dataset_exists
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_header("Summary")
    
    if all_passed:
        print("🎉 All checks passed! You're ready for Phase 2.")
        print("\nRecommended device for training:")
        if torch_ok:
            if torch.backends.mps.is_available():
                print("   device = torch.device('mps')  # M1 GPU")
            else:
                print("   device = torch.device('cpu')  # CPU fallback")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        return 1
    
    # -------------------------------------------------------------------------
    # Quick benchmark
    # -------------------------------------------------------------------------
    if torch_ok and torch.backends.mps.is_available():
        print_header("Quick MPS Benchmark")
        import time
        
        device = torch.device("mps")
        sizes = [1000, 2000, 4000]
        
        for size in sizes:
            x = torch.randn(size, size, device=device)
            
            # Warmup
            _ = torch.matmul(x, x)
            torch.mps.synchronize()  # Wait for GPU
            
            # Timed run
            start = time.perf_counter()
            for _ in range(10):
                _ = torch.matmul(x, x)
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            
            gflops = (2 * size**3 * 10) / elapsed / 1e9
            print(f"   {size}x{size} matmul: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")
        
        print("\n   Your M1 is ready for transformer training! 🚀")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
