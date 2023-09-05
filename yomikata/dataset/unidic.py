"""unidic.py
Data processing script for unidic dictionary 
Download: lex.csv from the full download https://clrd.ninjal.ac.jp/unidic/download.html)
"""

import warnings
from pathlib import Path

import pandas as pd

from yomikata.config import config, logger

warnings.filterwarnings("ignore")


def unidic_data():
    """Extract, load and transform the unidic data"""

    # Extract sentences from the data files
    unidic_file = list(Path(config.RAW_DATA_DIR, "unidic").glob("*.csv"))[0]

    # Load file
    df = pd.read_csv(
        unidic_file,
        header=None,
        names="surface id1 id2 id3 pos1 pos2 pos3 pos4 cType "
        "cForm lForm lemma orth orthBase pron pronBase goshu iType iForm fType "
        "fForm iConType fConType type kana kanaBase form formBase aType aConType "
        "aModType lid lemma_id".split(" "),
    )

    df["surface"] = df["surface"].astype(str).str.strip()
    df["kana"] = df["kana"].astype(str).str.strip()
    df = df[df["kana"] != "*"]
    df = df[df["surface"] != df["kana"]]
    df = df[["surface", "kana"]]

    df.to_csv(Path(config.READING_DATA_DIR, "unidic.csv"), index=False)

    logger.info("✅ Processed unidic data!")


if __name__ == "__main__":
    unidic_data()
