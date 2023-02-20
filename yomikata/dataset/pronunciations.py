from pathlib import Path

import jaconv
import pandas as pd
from tqdm import tqdm

from yomikata import utils
from yomikata.config import config, logger


def pronunciation_data():
    data_files = list(Path(config.READING_DATA_DIR).glob("*.csv"))

    df = pd.DataFrame()

    for file in data_files:
        if (file.name == "all.csv") or (file.name == "ambiguous.csv"):
            continue
        output_df = pd.read_csv(file)
        df = pd.concat([df, output_df])

    df["surface"] = df["surface"].astype(str).str.strip()
    df["kana"] = df["kana"].astype(str).str.strip()

    tqdm.pandas()

    df["kana"] = df["kana"].progress_apply(utils.standardize_text)
    df["surface"] = df["surface"].progress_apply(utils.standardize_text)
    df["kana"] = df.progress_apply(lambda row: jaconv.kata2hira(row["kana"]), axis=1)
    df = df[df["surface"] != df["kana"]]
    df = df[df["kana"] != ""]

    df = df[df["surface"].progress_apply(utils.has_kanji)]

    df = df.loc[~df["surface"].str.contains(r"[〜〜（）\)\(\*]\.")]

    df = df[["surface", "kana"]]
    df = df.drop_duplicates()

    df.to_csv(Path(config.READING_DATA_DIR, "all.csv"), index=False)

    logger.info("✅ Merged all the pronunciation data!")

    # merged_df = (
    #     df.groupby("surface")["kana"]
    #     .apply(list)
    #     .reset_index(name="pronunciations")
    # )

    # ambiguous_df = merged_df[merged_df["pronunciations"].apply(len) > 1]
    # ambiguous_df.to_csv(Path(config.READING_DATA_DIR, "ambiguous.csv"), index=False)


if __name__ == "__main__":
    pronunciation_data()
