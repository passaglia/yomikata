from pathlib import Path

import pandas as pd

from yomikata.config import config, logger

pronunciation_df = pd.read_csv(Path(config.READING_DATA_DIR, "all.csv"))
pronunciation_df = pronunciation_df.groupby("surface")["kana"].apply(list)


def repair_long_vowels(kana: str, kanji: str = None) -> str:
    """Clean and normalize text

    Args:
        kana (str): input string
        kanji (str): input string, optional

    Returns:
        str: a cleaned string
    """

    reading = kana
    indices_of_dash = [pos for pos, char in enumerate(reading) if char == "ー"]

    # get rid of non-ambiguous dashes
    for index_of_dash in indices_of_dash:
        char_before_dash = reading[index_of_dash - 1]
        if char_before_dash in "ぬつづむるくぐすずゆゅふぶぷ":
            reading = reading[:index_of_dash] + "う" + reading[index_of_dash + 1 :]
        elif char_before_dash in "しじみいきぎひびちぢぃ":
            reading = reading[:index_of_dash] + "い" + reading[index_of_dash + 1 :]

    indices_of_not_dash = [pos for pos, char in enumerate(reading) if char != "ー"]
    if len(indices_of_not_dash) != len(reading):
        if not kanji:
            logger.info("Disambiguating this dash requires kanji")
            logger.info(f"Left dash in {reading}")
        else:
            try:
                candidate_pronunciations = list(pronunciation_df[kanji])
            except KeyError:
                candidate_pronunciations = []

            candidate_pronunciations = list(set(candidate_pronunciations))

            candidate_pronunciations = [
                x for x in candidate_pronunciations if len(x) == len(reading)
            ]
            candidate_pronunciations = [
                x
                for x in candidate_pronunciations
                if all([x[i] == reading[i] for i in indices_of_not_dash])
            ]

            if len(candidate_pronunciations) == 1:
                reading = candidate_pronunciations[0]
            else:
                pass
                # logger.warning(f"Left dashes in {kanji} {reading}")

    return reading
