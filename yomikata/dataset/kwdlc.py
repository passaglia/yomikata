"""kwdlc.py
Data processing script for KWDLC files directly in the repository format
KWDLC repository: https://github.com/ku-nlp/KWDLC
"""

import warnings
from pathlib import Path

import pandas as pd
from speach import ttlig

from yomikata import utils
from yomikata.config import config, logger

warnings.filterwarnings("ignore")


def read_knp_file(filename: str):
    with open(filename) as f:
        contents = f.readlines()

    ids = []
    sentences = []
    furiganas = []
    sentence = ""
    furigana = ""
    for row in contents:
        first_word = row.split(" ")[0]
        if first_word in ["*", "+"]:
            pass
        elif first_word == "#":
            sentence_id = row.split(" ")[1].split("S-ID:")[1]
        elif first_word == "EOS\n":
            sentence = utils.standardize_text(sentence)
            furigana = utils.standardize_text(furigana)
            if sentence == utils.remove_furigana(furigana):
                sentences.append(sentence)
                furiganas.append(furigana)
                ids.append(sentence_id)
            else:
                logger.warning(
                    f"Dropping mismatched line \n Sentence: {sentence} \n  Furigana: {furigana}"
                )
            sentence = ""
            furigana = ""
        else:
            words = row.split(" ")
            sentence += words[0]
            if words[0] == words[1]:
                furigana += words[0]
            else:
                furigana += ttlig.RubyToken.from_furi(words[0], words[1]).to_code()

    assert len(ids) == len(sentences)
    assert len(sentences) == len(furiganas)
    return ids, sentences, furiganas  # readings


def kwdlc_data():
    """Extract, load and transform the kwdlc data"""

    # Extract sentences from the data files
    knp_files = list(Path(config.RAW_DATA_DIR, "kwdlc").glob("**/*.knp"))

    all_ids = []
    all_sentences = []
    all_furiganas = []
    for knp_file in knp_files:
        ids, sentences, furiganas = read_knp_file(knp_file)
        all_ids += ids
        all_sentences += sentences
        all_furiganas += furiganas

    # construct dataframe
    df = pd.DataFrame(
        list(zip(all_sentences, all_furiganas, all_ids)),  # all_readings, all_furiganas)),
        columns=["sentence", "furigana", "sentenceid"],
    )

    # remove known errors
    error_ids = [
        "w201106-0000547376-1",
        "w201106-0001768070-1-01",
        "w201106-0000785999-1",
        "w201106-0001500842-1",
        "w201106-0000704257-1",
        "w201106-0002300346-3",
        "w201106-0001779669-3",
        "w201106-0000259203-1",
    ]

    df = df[~df["sentenceid"].isin(error_ids)]
    df = df.drop_duplicates()
    df["furigana"] = df["furigana"].apply(utils.standardize_text)
    df["sentence"] = df["sentence"].apply(utils.standardize_text)
    # Test
    assert (df["sentence"] == df["furigana"].apply(utils.remove_furigana)).all()

    # Output
    df.to_csv(Path(config.SENTENCE_DATA_DIR, "kwdlc.csv"), index=False)

    logger.info("✅ Saved kwdlc data!")


if __name__ == "__main__":
    kwdlc_data()
