"""sudachi.py
Data processing script for sudachi dictionary
"""

import warnings
from pathlib import Path

import pandas as pd

from yomikata.config import config, logger

warnings.filterwarnings("ignore")


def sudachi_data():
    sudachi_file = list(Path(config.RAW_DATA_DIR, "sudachi").glob("*.csv"))

    df = pd.DataFrame()

    for file in sudachi_file:
        logger.info(file.name)
        # Load file
        df = pd.concat(
            [
                df,
                pd.read_csv(
                    file,
                    header=None,
                ),
            ]
        )

    df["surface"] = df[0].astype(str).str.strip()
    df["kana"] = df[11].astype(str).str.strip()
    df["type"] = df[5].astype(str).str.strip()
    df = df[df["kana"] != "*"]
    df = df[df["surface"] != df["kana"]]
    df = df[df["type"] != "補助記号"]

    df = df[["surface", "kana"]]

    df.to_csv(Path(config.READING_DATA_DIR, "sudachi.csv"), index=False)

    logger.info("✅ Processed sudachi data!")


if __name__ == "__main__":
    sudachi_data()
