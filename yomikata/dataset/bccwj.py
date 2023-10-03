"""bccwj.py
Data processing script for files downloaded from Chuunagon search
Chuunagon URL: https://chunagon.ninjal.ac.jp/

Download with the settings
文脈中の区切り記号 |
文脈中の文区切り記号 #
前後文脈の語数 10
検索対象（固定長・可変長） 両方
共起条件の範囲 文境界をまたがない

ダウンロードオプション
システム Linux
文字コード UTF-8
改行コード LF
出力ファイルが一つの場合は Zip 圧縮を行わない 検索条件式ごとに出力ファイルを分割する
インラインタグを使用  CHECK BOTH 語彙素読み AND 発音形出現形語種 BOX
(発音形出現形 is the actual pronounced one, but displays e.g. よう　れい　as よー　れー)
タグの区切り記号 :
"""

import warnings
from pathlib import Path

import jaconv
import pandas as pd
from speach.ttlig import RubyToken
from tqdm import tqdm

from yomikata import utils
from yomikata.config import config, logger

warnings.filterwarnings("ignore")

SENTENCE_SPLIT_CHAR = "#"
WORD_SPLIT_CHAR = "|"
READING_SEP_CHAR = ":"


def read_bccwj_file(filename: str):
    """ """

    df = pd.read_csv(filename, sep="\t")

    df["前文脈"] = df["前文脈"].fillna("")
    df["後文脈"] = df["後文脈"].fillna("")
    df["full_text"] = (
        df["前文脈"] + df["キー"] + "[" + df["語彙素読み"] + ":" + df["発音形出現形"] + "]" + df["後文脈"]
    )

    def get_sentences(row):
        sentences = row["full_text"].split(SENTENCE_SPLIT_CHAR)
        furigana_sentences = []
        for sentence in sentences:
            words_with_readings = sentence.split(WORD_SPLIT_CHAR)
            furigana_sentence = ""
            for word_with_reading in words_with_readings:
                word = word_with_reading.split("[")[0]
                form, reading = jaconv.kata2hira(
                    word_with_reading.split("[")[1].split("]")[0]
                ).split(READING_SEP_CHAR)

                if (
                    not utils.has_kanji(word)
                    or reading == jaconv.kata2hira(word)
                    or form == ""
                    or reading == ""
                ):
                    furigana_sentence += word
                else:
                    if ("ー" in reading) and ("ー" not in form):
                        indexes_of_dash = [pos for pos, char in enumerate(reading) if char == "ー"]
                        for index_of_dash in indexes_of_dash:
                            if len(reading) == len(form):
                                dash_reading = form[index_of_dash]
                            else:
                                char_before_dash = reading[index_of_dash - 1]
                                if char_before_dash in "ねめせぜれてでけげへべぺ":
                                    digraphA = char_before_dash + "え"
                                    digraphB = char_before_dash + "い"
                                    if digraphA in form and digraphB not in form:
                                        dash_reading = "え"
                                    elif digraphB in form and digraphA not in form:
                                        dash_reading = "い"
                                    else:
                                        logger.warning(f"Leaving dash in {word} {form} {reading}")
                                        dash_reading = "ー"
                                elif char_before_dash in "ぬつづむるくぐすずゆゅふぶぷ":
                                    dash_reading = "う"
                                elif char_before_dash in "しじみいきぎひびち":
                                    dash_reading = "い"
                                elif char_before_dash in "そぞのこごもろとどよょおほぼぽ":
                                    digraphA = char_before_dash + "お"
                                    digraphB = char_before_dash + "う"
                                    if digraphA in form and digraphB not in form:
                                        dash_reading = "お"
                                    elif digraphB in form and digraphA not in form:
                                        dash_reading = "う"
                                    else:
                                        if digraphA in word and digraphB not in word:
                                            dash_reading = "お"
                                        elif digraphB in word and digraphA not in word:
                                            dash_reading = "う"
                                        else:
                                            logger.warning(
                                                f"Leaving dash in {word} {form} {reading}"
                                            )
                                            dash_reading = "ー"
                                else:
                                    logger.warning(f"Leaving dash in {word} {form} {reading}")
                                    dash_reading = "ー"
                            reading = (
                                reading[:index_of_dash]
                                + dash_reading
                                + reading[index_of_dash + 1 :]
                            )
                    furigana_sentence += RubyToken.from_furi(word, reading).to_code()

            furigana_sentences.append(furigana_sentence)

        furigana_sentences = [utils.standardize_text(sentence) for sentence in furigana_sentences]
        sentences = [utils.remove_furigana(sentence) for sentence in furigana_sentences]
        try:
            rowid = row["サンプル ID"]
        except KeyError:
            rowid = row["講演 ID"]
        if len(furigana_sentences) == 1:
            ids = [rowid]
        else:
            ids = [rowid + "_" + str(i) for i in range(len(furigana_sentences))]

        sub_df = pd.DataFrame(
            {"sentence": sentences, "furigana": furigana_sentences, "sentenceid": ids}
        )

        sub_df = sub_df[sub_df["sentence"] != sub_df["furigana"]]

        return sub_df

    output_df = pd.DataFrame()
    for i, row in tqdm(df.iterrows()):
        try:
            output_df = pd.concat([output_df, get_sentences(row)])
        except AttributeError:
            continue

    return output_df


def bccwj_data():
    """Extract, load and transform the bccwj data"""

    # Extract sentences from the data files
    bccwj_files = list(Path(config.RAW_DATA_DIR, "bccwj").glob("*.txt"))

    df = pd.DataFrame()

    for bccwj_file in bccwj_files:
        logger.info(bccwj_file.name)
        df = pd.concat([df, read_bccwj_file(bccwj_file)])

    # remove known errors
    error_ids = []

    df = df[~df["sentenceid"].isin(error_ids)]
    df = df[df["sentence"] != ""]
    df = df.drop_duplicates()
    df["furigana"] = df["furigana"].apply(utils.standardize_text)
    df["sentence"] = df["sentence"].apply(utils.standardize_text)
    assert (df["sentence"] == df["furigana"].apply(utils.remove_furigana)).all()

    # Output
    df.to_csv(Path(config.SENTENCE_DATA_DIR, "bccwj.csv"), index=False)

    logger.info("✅ Saved bccwj data!")


def bccwj_subset(bccwj_file):
    """Extract, load and transform a subset of the bccwj data"""

    df = read_bccwj_file(bccwj_file)

    # remove known errors
    error_ids = []

    df = df[~df["sentenceid"].isin(error_ids)]
    df = df.drop_duplicates()
    df["furigana"] = df["furigana"].apply(utils.standardize_text)
    df["sentence"] = df["sentence"].apply(utils.standardize_text)

    # Output
    df.to_csv(
        Path(config.SENTENCE_DATA_DIR, bccwj_file.name.split(".")[0] + ".csv"),
        index=False,
    )

    logger.info("✅ Saved bccwj " + bccwj_file.name.split(".")[0] + " data!")


if __name__ == "__main__":
    bccwj_data()
