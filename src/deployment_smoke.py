"""Bounded headless server health check; no training or external publication."""

from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


def server_smoke(timeout=45):
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless=true", "--server.address=127.0.0.1", f"--server.port={port}"],
            cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                        if response.status == 200 and response.read() == b"ok":
                            return {"health": "passed", "scope": "HTTP health only; UI rendering is covered by AppTest, not this probe"}
                except (URLError, TimeoutError, ConnectionError):
                    time.sleep(0.2)
            log.seek(0)
            raise RuntimeError("Streamlit health check failed. " + log.read().decode("utf-8", errors="replace")[-3000:])
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    print(server_smoke())
