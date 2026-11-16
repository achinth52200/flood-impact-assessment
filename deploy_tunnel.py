"""
Start Streamlit + expose via ngrok tunnel.
Run: python deploy_tunnel.py
"""
import subprocess, sys, time, os
from pathlib import Path

def main():
    token = os.environ.get("NGROK_AUTHTOKEN", "")

    # Start Streamlit in background
    print("Starting Streamlit…")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless=true", "--server.port=8501"],
        cwd=str(Path(__file__).parent),
    )
    time.sleep(4)

    # Open ngrok tunnel
    from pyngrok import ngrok, conf
    if token:
        conf.get_default().auth_token = token

    try:
        tunnel = ngrok.connect(8501, "http")
        print("\n" + "="*55)
        print(f"  🌊  Flood Assessment App — PUBLIC URL:")
        print(f"  👉  {tunnel.public_url}")
        print("="*55)
        print("\nShare this link. Press Ctrl+C to stop.\n")
        proc.wait()
    except Exception as e:
        print(f"\nngrok error: {e}")
        print("Without ngrok auth, the tunnel has a 2-hour limit.")
        print("Get a free token at https://dashboard.ngrok.com/")
        print("\nApp is still running at http://localhost:8501")
        proc.wait()

if __name__ == "__main__":
    main()
