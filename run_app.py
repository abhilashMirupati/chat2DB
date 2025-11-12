"""Entry point to launch the Streamlit UI."""

import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
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

