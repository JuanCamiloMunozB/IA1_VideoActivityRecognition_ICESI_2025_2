"""
Entry point for the Video Activity Recognition executable.

This script handles the setup needed when running as a bundled executable,
including path configuration and environment setup.
"""
import sys
import os
from pathlib import Path


def setup_executable_environment():
    """Configure environment for running as a PyInstaller executable."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        application_path = Path(sys._MEIPASS)
        
        # Set PROJECT_ROOT to the bundled resources directory
        os.environ['PROJECT_ROOT'] = str(application_path)
        
        print(f"[Executable] Running from: {application_path}")
    else:
        # Running as normal Python script
        application_path = Path(__file__).parent.resolve()
        os.environ['PROJECT_ROOT'] = str(application_path)
        
        print(f"[Development] Running from: {application_path}")


def main():
    """Main entry point for the executable."""
    # Setup environment
    setup_executable_environment()
    
    # Import and run the actual application
    # We import here (after setup) to ensure paths are configured
    try:
        from Entrega3.src.online.ui_app import run_realtime_app
        
        print("=" * 60)
        print("Video Activity Recognition - Real-time HAR System")
        print("=" * 60)
        print("Presiona 'q' en la ventana de video para salir")
        print()
        
        # Run the application
        run_realtime_app(camera_index=0)
        
    except KeyboardInterrupt:
        print("\n[Executable] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Failed to run application: {e}")
        import traceback
        traceback.print_exc()
        
        # Write error to file for debugging
        error_log = Path("error_log.txt")
        with open(error_log, "w") as f:
            f.write(f"Error running VideoActivityRecognition\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"{e}\n\n")
            f.write(traceback.format_exc())
        
        print(f"\nError details written to: {error_log.absolute()}")
        
        # Only call input() if we have a console
        if sys.stdin and sys.stdin.isatty():
            input("\nPresiona Enter para cerrar...")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
