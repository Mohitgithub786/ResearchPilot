import subprocess
import sys
import time
import urllib.request
import webbrowser
import os

def check_health():
    try:
        req = urllib.request.Request("http://127.0.0.1:8085/health")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except:
        return False

def main():
    print("=== ResearchPilot Startup ===")
    
    # 1. Check/Create Virtual Environment
    if not os.path.exists(".venv"):
        print("-> Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"])
    
    python_exe = os.path.join(".venv", "Scripts", "python.exe") if os.name == 'nt' else os.path.join(".venv", "bin", "python")
    
    if not os.path.exists(python_exe):
        print(f"Error: Could not find Python at {python_exe}. Please delete the .venv folder and try again.")
        sys.exit(1)

    # 2. Install dependencies
    print("-> Installing/Verifying dependencies (this may take a moment)...")
    subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt"], stdout=subprocess.DEVNULL)
    
    # 3. Start Backend
    print("-> Starting FastAPI backend on port 8085...")
    backend = subprocess.Popen([python_exe, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8085"])
    
    # 4. Wait for Backend
    print("-> Waiting for backend to become ready (this can take over a minute if loading models for the first time)...", end="")
    sys.stdout.flush()
    
    while not check_health():
        time.sleep(3)
        print(".", end="")
        sys.stdout.flush()
        if backend.poll() is not None:
            print("\n[!] Backend crashed or failed to start!")
            sys.exit(1)
            
    print("\n-> Backend is healthy and running!")
    
    # 5. Start Frontend
    print("-> Starting Gradio frontend...")
    frontend = subprocess.Popen([python_exe, "gradio_app/app.py"])
    
    # Give frontend a couple of seconds to bind to the port
    time.sleep(3)
    
    # 6. Open Browser
    print("-> Opening UI in your default browser...")
    webbrowser.open("http://127.0.0.1:7860")
    
    print("\n=== ResearchPilot is running! ===")
    print("Press Ctrl+C in this terminal to shut down both the backend and frontend.\n")
    
    # 7. Keep script alive and handle shutdown
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nShutting down ResearchPilot...")
        backend.terminate()
        frontend.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
