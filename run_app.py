"""Entry point to launch the Streamlit UI."""

import os
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv

# Load config to check telemetry setting
load_dotenv()
from sqlai.config import load_app_config
from sqlai.utils.logging import _disable_telemetry


def main() -> None:
    # Disable telemetry if configured (prevents PostHog SSL errors at root cause)
    _disable_telemetry()
    
    app_path = Path(__file__).parent / "src" / "sqlai" / "ui" / "app.py"
    project_src = str(Path(__file__).parent / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()

