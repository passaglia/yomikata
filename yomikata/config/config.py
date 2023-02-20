# config.py

import json
import logging.config
import sys
from pathlib import Path

from rich.logging import RichHandler

# Base and Config Directories
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
YOMIKATA_DIR = Path(BASE_DIR, "yomikata")
CONFIG_DIR = Path(YOMIKATA_DIR, "config")

# Data Directories
DATA_DIR = Path(BASE_DIR, "data")

RAW_DATA_DIR = Path(DATA_DIR, "raw")
SENTENCE_DATA_DIR = Path(DATA_DIR, "sentence_data")
READING_DATA_DIR = Path(DATA_DIR, "reading_data")

TRAIN_DATA_DIR = Path(SENTENCE_DATA_DIR, "train")
VAL_DATA_DIR = Path(SENTENCE_DATA_DIR, "val")
TEST_DATA_DIR = Path(SENTENCE_DATA_DIR, "test")

# Logs Directory
LOGS_DIR = Path(BASE_DIR, "logs")

# Model Storage Directory
STORES_DIR = Path(BASE_DIR, "stores")
RUN_REGISTRY = Path(STORES_DIR, "runs")

# Default model directories
DBERT_DIR = Path(YOMIKATA_DIR, "dbert-artifacts")

# Create dirs
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
SENTENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
VAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
READING_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
STORES_DIR.mkdir(parents=True, exist_ok=True)
RUN_REGISTRY.mkdir(parents=True, exist_ok=True)

# Special tokens reserved
ASCII_SPACE_TOKEN = "\U0000FFFF"  # this is used to replace the usual space characters before sending text to mecab, because mecab uses the usual space to separate words.

# Seed # Currently seeds not actually set anywhere
SEED = 1271297

# Training parameters
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
assert TRAIN_SIZE + VAL_SIZE + TEST_SIZE == 1

# Heteronym list
with open(Path(CONFIG_DIR, "heteronyms.json")) as fp:
    HETERONYMS = json.load(fp)

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)
