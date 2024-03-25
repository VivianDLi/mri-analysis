"""
Contains project-level constants used to configure paths and logging.

Paths are configured using the `.env` file in the project root.
"""

import logging
import os
import pathlib
from loguru import logger

# path constants
SRC_PATH = pathlib.Path(__file__).parent
"""Path to the project source code. """

PROJECT_PATH = SRC_PATH.parent
"""Path to the project root."""

# loading .env variables
if not os.path.exists(PROJECT_PATH / ".env"):
    logger.debug(
        "No `.env` file found in project root. Checking for env vars..."
    )
    # If no `.env` file found, check for an env var
    if os.environ.get("DATA_PATH") is not None:
        logger.debug("Found env var `DATA_PATH`:.")
        DATA_PATH = os.environ.get("DATA_PATH")
    else:
        logger.debug("No env var `DATA_PATH` found. Setting default...")
        DATA_PATH = str(SRC_PATH / "data")
        os.environ["DATA_PATH"] = str(DATA_PATH)
else:
    import dotenv  # lazy import to avoid dependency on dotenv

    dotenv.load_dotenv(PROJECT_PATH / ".env")
    DATA_PATH = os.environ.get("DATA_PATH")

logger.info(f"DATA_PATH: {DATA_PATH}")
# Set default environment paths as fallback if not specified in .env file
if os.environ.get("ROOT_DIR") is None:
    ROOT_DIR = str(PROJECT_PATH)
    os.environ["ROOT_DIR"] = str(ROOT_DIR)
if os.environ.get("DATA_PATH") is None:
    DATA_PATH = str(PROJECT_PATH / "data")
    os.environ["DATA_PATH"] = str(DATA_PATH)
if os.environ.get("METADATA_PATH") is None:
    METADATA_PATH = str(PROJECT_PATH / "metadata")
    os.environ["METADATA_PATH"] = str(METADATA_PATH)
if os.environ.get("MODELS_PATH") is None:
    MODELS_PATH = str(PROJECT_PATH / "models")
    os.environ["MODELS_PATH"] = str(MODELS_PATH)
if os.environ.get("RESULTS_PATH") is None:
    RESULTS_PATH = str(PROJECT_PATH / "results")
    os.environ["RESULTS_PATH"] = str(RESULTS_PATH)

# check for brain plotting path
BRAINPLOT_PATH = os.environ.get("BRAIN_PLOTTING_PATH")

# hydra constants
HYDRA_CONFIG_PATH = SRC_PATH / "config"

# logging constants
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s [in %(funcName)s at %(pathname)s:%(lineno)d]"
)
DEFAULT_LOG_FILE = PROJECT_PATH / "logs" / "default_log.log"
DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_LEVEL = logging.DEBUG  # verbose logging per default
