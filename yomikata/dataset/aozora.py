"""aozora.py
Data processing script for aozora bunko file from https://github.com/ndl-lab/huriganacorpus-aozora
"""

import warnings
from pathlib import Path

import pandas as pd
from pandas.errors import ParserError
from speach import ttlig

from yomikata import utils
from yomikata.config import config, logger
from yomikata.dataset.repair_long_vowels import repair_long_vowels

warnings.filterwarnings("ignore")


def read_file(file: str):
    # logger.info("reading file")
    with open(file) as f:
        rows = [line.rstrip("\n").rstrip("\r").split("\t")[0:3] for line in f.readlines()]
    df = pd.DataFrame(rows, columns=["word", "furigana", "type"])

    # logger.info("removing unused rows")
    # remove unused rows
    df = df[~df["type"].isin(["[入力 読み]", "分かち書き"])]
    df = df[~pd.isna(df["word"])]
    df = df[~pd.isnull(df["word"])]
    df = df[df["word"] != ""]

    # logger.info("organizing into sentences")
    # now organize remaining rows into sentences
    gyou_df = pd.DataFrame(columns=["sentence", "furigana", "sentenceid"])
    sentence = ""
    furigana = ""
    sentenceid = None
    gyous = []
    for row in df.itertuples():
        if row.type in ["[入力文]"]:
            sentence = row.word
        elif row.type in ["漢字"]:
            furigana += ttlig.RubyToken.from_furi(
                row.word, repair_long_vowels(row.furigana, row.word)
            ).to_code()
        elif row.word.split(":")[0] in ["行番号"]:
            if sentenceid:  # this handles the first row
                gyous.append([sentence, furigana, sentenceid])
            sentenceid = file.name + "_" + row.word.split(":")[1].strip()
            sentence = None
            furigana = ""
        else:
            furigana += row.word

    # last row handling
    gyous.append([sentence, furigana, sentenceid])

    # make dataframe
    gyou_df = pd.DataFrame(gyous, columns=["sentence", "furigana", "sentenceid"])
    gyou_df = gyou_df[~pd.isna(gyou_df.sentence)]

    # logger.info("cleaning rows")
    # clean rows
    gyou_df["furigana"] = gyou_df["furigana"].apply(utils.standardize_text)
    gyou_df["sentence"] = gyou_df["sentence"].apply(
        lambda s: utils.standardize_text(s.replace("|", "").replace(" ", "").replace("※", ""))
    )

    # logger.info("removing errors")
    # remove non-matching rows
    gyou_df = gyou_df[gyou_df["sentence"] == gyou_df["furigana"].apply(utils.remove_furigana)]

    # remove known errors
    error_ids = []
    gyou_df = gyou_df[~gyou_df["sentenceid"].isin(error_ids)]

    # remove duplicates
    gyou_df = gyou_df.drop_duplicates()

    return gyou_df


def aozora_data():
    """Extract, load and transform the aozora data"""

    # Extract sentences from the data files
    files = list(Path(config.RAW_DATA_DIR, "aozora").glob("*/*/*.txt"))

    with open(Path(config.SENTENCE_DATA_DIR, "aozora.csv"), "w") as f:
        f.write("sentence,furigana,sentenceid\n")

    for i, file in enumerate(files):
        logger.info(f"{i+1}/{len(files)} {file.name}")
        try:
            df = read_file(file)
        except ParserError:
            logger.error(f"Parser error on {file}")

        df.to_csv(
            Path(config.SENTENCE_DATA_DIR, "aozora.csv"),
            mode="a",
            index=False,
            header=False,
        )

    logger.info("✅ Saved all aozora data!")


if __name__ == "__main__":
    aozora_data()
