"""Entry point to launch the Streamlit UI.

Recommended workflow:
1. Pre-load cache: python scripts/prewarm_metadata.py
2. Validate cache: python scripts/validate_cache.py
3. Launch UI: python run_app.py

The UI will work without pre-loaded cache, but startup will be slower.
"""

import os

# Disable all analytics BEFORE any other imports (prevents PostHog SSL errors)
os.environ["LANGCHAIN_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LLAMA_INDEX_ANALYTICS_ENABLED"] = "false"
os.environ["POSTHOG_DISABLE"] = "true"
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["OPENAI_TELEMETRY_OPTOUT"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_TELEMETRY"] = "false"
os.environ["DO_NOT_TRACK"] = "1"

# Hard monkeypatch PostHog so it CANNOT send anything
try:
    import posthog
    posthog.disabled = True
except Exception:
    pass

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

