# run_app.py
# Purpose: thin entry point that loads config and runs the Tkinter client.
# Author: Colile Sibanda

from __future__ import annotations

from desktop.app import PhishLensApp
from phishlens.config.loader import load_config


def main() -> None:
    """Load config, build the desktop window and run it.

    Inputs: none.
    Output: none; blocks until the window is closed. The API must already be
    serving at `config.desktop.api_base_url`; if it is not, the window still
    opens and reports the service as unreachable when Analyse is clicked.
    """
    config = load_config()
    app = PhishLensApp(
        api_base_url=config.desktop.api_base_url,
        timeout=config.desktop.request_timeout_seconds,
    )
    app.run()


if __name__ == "__main__":
    main()
