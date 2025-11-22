"""
Build script for creating a Windows executable of the Video Activity Recognition app.

This script uses PyInstaller to bundle the entire application including:
- All Python dependencies (OpenCV, MediaPipe, scikit-learn, etc.)
- Source code from Entrega2 and Entrega3
- Model files (.joblib) and data files (features.csv)

Usage:
    python build_exe.py
"""
import sys
import subprocess
import shutil
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.resolve()
ENTREGA2_DIR = PROJECT_ROOT / "Entrega2"
ENTREGA3_DIR = PROJECT_ROOT / "Entrega3"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def check_requirements():
    """Verify all required files exist before building."""
    print("[Build] Checking requirements...")
    
    required_files = [
        ENTREGA3_DIR / "experiments" / "models" / "label_encoder.joblib",
        ENTREGA2_DIR / "experiments" / "results" / "features.csv",
        ENTREGA2_DIR / "src" / "features" / "feature_engineering.py",
        ENTREGA3_DIR / "src" / "online" / "ui_app.py",
    ]
    
    # At least one model file should exist
    model_files = [
        ENTREGA3_DIR / "experiments" / "models" / "svm_reduced.joblib",
        ENTREGA3_DIR / "experiments" / "models" / "svm_full.joblib",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path)
    
    if not any(f.exists() for f in model_files):
        missing_files.extend(model_files)
        print("  [ERROR] At least one SVM model file is required!")
    
    if missing_files:
        print("  [ERROR] Missing required files:")
        for f in missing_files:
            print(f"    - {f.relative_to(PROJECT_ROOT)}")
        return False
    
    print("  [OK] All required files found")
    return True


def install_pyinstaller():
    """Install PyInstaller if not already installed."""
    print("[Build] Checking PyInstaller...")
    try:
        import PyInstaller
        print(f"  [OK] PyInstaller {PyInstaller.__version__} is installed")
        return True
    except ImportError:
        print("  [INFO] PyInstaller not found, installing...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "pyinstaller"
            ])
            print("  [OK] PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to install PyInstaller: {e}")
            return False


def create_pyinstaller_command():
    """Generate the PyInstaller command with all necessary options."""
    
    # Base command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=VideoActivityRecognition",
        "--onefile",  # Single executable
        # --windowed removed for debugging (shows console for error messages)
        "--noconfirm",  # Overwrite without asking
        
        # Entry point
        str(PROJECT_ROOT / "app_entry.py"),
    ]
    
    # Add hidden imports for packages that PyInstaller might miss
    hidden_imports = [
        # sklearn internals - comprehensive list
        "sklearn.utils._typedefs",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._partition_nodes",
        "sklearn.tree._utils",
        "sklearn.utils._weight_vector",
        # sklearn pipeline and preprocessing
        "sklearn.pipeline",
        "sklearn.preprocessing",
        "sklearn.preprocessing._data",
        "sklearn.preprocessing._encoders",
        "sklearn.preprocessing._label",
        # sklearn feature selection
        "sklearn.feature_selection",
        "sklearn.feature_selection._univariate_selection",
        # sklearn svm
        "sklearn.svm",
        "sklearn.svm._classes",
        "sklearn.svm._libsvm",
        "sklearn.svm._liblinear",
        # sklearn metrics
        "sklearn.metrics",
        "sklearn.metrics._classification",
        # OpenCV
        "cv2",
        "cv2.data",
        # MediaPipe
        "mediapipe",
        "mediapipe.python",
        # Other packages
        "numpy",
        "numpy.core._methods",
        "numpy.lib.format",
        "pandas",
        "pandas._libs.tslibs.timedeltas",
        "joblib",
        "xgboost",
        "scipy",
        "scipy.special.cython_special",
        "scipy.sparse.csgraph._validation",
    ]
    
    for module in hidden_imports:
        cmd.extend(["--hidden-import", module])
    
    # Add data files - Models from Entrega3
    models_dir = ENTREGA3_DIR / "experiments" / "models"
    if models_dir.exists():
        cmd.extend([
            "--add-data",
            f"{models_dir};Entrega3/experiments/models"
        ])
    
    # Add data files - Results from Entrega3
    results_dir = ENTREGA3_DIR / "experiments" / "results"
    if results_dir.exists():
        cmd.extend([
            "--add-data",
            f"{results_dir};Entrega3/experiments/results"
        ])
    
    # Add data files - Results from Entrega2 (features.csv)
    e2_results = ENTREGA2_DIR / "experiments" / "results"
    if e2_results.exists():
        cmd.extend([
            "--add-data",
            f"{e2_results};Entrega2/experiments/results"
        ])
    
    # Add entire source directories as they contain code
    cmd.extend([
        "--add-data", f"{ENTREGA2_DIR / 'src'};Entrega2/src",
        "--add-data", f"{ENTREGA3_DIR / 'src'};Entrega3/src",
    ])
    
    # Collect binaries for mediapipe and opencv
    # Note: opencv-python is the package name, but imports as cv2
    cmd.extend([
        "--collect-all", "mediapipe",
        "--collect-binaries", "cv2",
        "--collect-data", "cv2",
    ])
    
    return cmd


def clean_build_dirs():
    """Remove old build artifacts."""
    print("[Build] Cleaning old build directories...")
    
    dirs_to_clean = [BUILD_DIR, DIST_DIR]
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"  [OK] Removed {dir_path.name}/")
            except Exception as e:
                print(f"  [WARNING] Could not remove {dir_path.name}/: {e}")


def build_executable():
    """Execute PyInstaller to build the executable."""
    print("\n" + "=" * 60)
    print("Building executable...")
    print("=" * 60)
    
    cmd = create_pyinstaller_command()
    
    print(f"\n[Build] Running PyInstaller...")
    print(f"Command: {' '.join(cmd[:3])} ...")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False
        )
        
        print("\n" + "=" * 60)
        print("Build completed successfully!")
        print("=" * 60)
        
        # Find the executable
        exe_path = DIST_DIR / "VideoActivityRecognition.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"\n[Success] Executable created:")
            print(f"  Location: {exe_path}")
            print(f"  Size: {size_mb:.1f} MB")
            print(f"\nYou can now run the executable directly:")
            print(f"  {exe_path}")
        else:
            print(f"\n[Warning] Build completed but executable not found at expected location")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Build failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during build: {e}")
        return False


def main():
    """Main build process."""
    print("=" * 60)
    print("Video Activity Recognition - Executable Builder")
    print("=" * 60)
    print()
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n[FAILED] Please fix the missing files and try again.")
        return 1
    
    # Step 2: Install PyInstaller
    if not install_pyinstaller():
        print("\n[FAILED] Could not install PyInstaller.")
        return 1
    
    # Step 3: Clean old builds
    clean_build_dirs()
    
    # Step 4: Build executable
    if not build_executable():
        print("\n[FAILED] Build process failed.")
        return 1
    
    print("\n" + "=" * 60)
    print("All done! 🎉")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
