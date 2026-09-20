# run_api.py
# Purpose: thin entry point that loads config and serves the FastAPI app.
# Author: Colile Sibanda

from __future__ import annotations

import uvicorn

from phishlens.config.loader import load_config
from phishlens.serving.app import create_app


def main() -> None:
    """Load config, build the app and serve it with uvicorn.

    Inputs: none.
    Output: none; blocks, serving the API on `config.serving.host:port`.
    """
    config = load_config()
    app = create_app(config)
    uvicorn.run(app, host=config.serving.host, port=config.serving.port)


if __name__ == "__main__":
    main()
