from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from speach.ttlig import RubyFrag, RubyToken

from yomikata import utils
from yomikata.config import config, logger
from yomikata.dictionary import Dictionary


def train_val_test_split(X, y, train_size, val_size, test_size):
    """Split dataset into data splits."""
    assert (train_size + val_size + test_size) == 1
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, train_size=val_size / (test_size + val_size)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def filter_simple(input_file, output_file, heteronyms) -> None:
    """This filters out sentences which don't contain any heteronyms"""

    df = pd.read_csv(input_file)  # load
    logger.info(f"Prefilter size: {len(df)}")

    df = df[df["sentence"].str.contains(r"|".join(heteronyms))]
    logger.info(f"Postfilter size: {len(df)}")

    df.to_csv(output_file, index=False)


def filter_dictionary(input_file, output_file, heteronyms, dictionary) -> None:
    """This filters out sentences which contain heteronyms only as part of a compound which is known to the dictionary"""
    df = pd.read_csv(input_file)  # load
    logger.info(f"Prefilter size: {len(df)}")

    df["contains_heteronym"] = df["sentence"].apply(
        lambda s: not set(
            [dictionary.token_to_surface(m) for m in dictionary.tagger(s)]
        ).isdisjoint(heteronyms)
    )

    df = df[df["contains_heteronym"]]
    logger.info(f"Postfilter size: {len(df)}")

    df.to_csv(output_file, index=False)


def regroup_furigana(s, heteronym, heteronym_dict, dictionary, verbose=False):
    rubytokens = utils.parse_furigana(s)
    output_tokens = []
    for token in rubytokens.groups:
        if isinstance(token, RubyFrag):
            # this is a token with furigana
            if heteronym in token.text and token.text != heteronym:
                # it includes the heteronym but is not exactly the heteronym
                # if len(dictionary.tagger(token.text)) > 1:
                # it is not in the dictionary, so we try to regroup it
                # note this dictionary check is not foolproof: sometimes words are in the dictionary and found here,
                # but in a parse of the whole sentence the word will be split in two.
                # commented this out since actually even if it is part of dictionary, it will go through the training and so we might as well try to regroup it to avoid it being an <OTHER>
                viable_regroupings = []
                for reading in heteronym_dict[heteronym]:
                    regrouped_tokens = regroup_furigana_tokens(
                        [token], heteronym, reading, verbose=verbose
                    )
                    if regrouped_tokens != [token]:
                        if verbose:
                            print("viable regrouping found")
                        viable_regroupings.append(regrouped_tokens)
                if len(viable_regroupings) == 1:
                    output_tokens += viable_regroupings[0]
                    continue
                else:
                    if verbose:
                        print("multiple viable readings found, cannot regroup")
                    pass
        output_tokens.append(token)

    output_string = RubyToken(groups=output_tokens).to_code()
    assert utils.furigana_to_kana(output_string) == utils.furigana_to_kana(s)
    assert utils.remove_furigana(output_string) == utils.remove_furigana(s)
    return output_string


def regroup_furigana_tokens(ruby_tokens, heteronym, reading, verbose=False):
    if not len(ruby_tokens) == 1:
        raise ValueError("regroup failed, no support yet for token merging")
    ruby_token = ruby_tokens[0]
    text = ruby_token.text
    furi = ruby_token.furi
    try:
        split_text = [
            text[0 : text.index(heteronym)],
            heteronym,
            text[text.index(heteronym) + len(heteronym) :],
        ]
        split_text = [text for text in split_text if text != ""]
    except ValueError:
        if verbose:
            print("regroup failed, heteronym not in token text")
        return ruby_tokens

    try:
        split_furi = [
            furi[0 : furi.index(reading)],
            reading,
            furi[furi.index(reading) + len(reading) :],
        ]
        split_furi = [furi for furi in split_furi if furi != ""]
    except ValueError:
        if verbose:
            print("regroup failed, reading not in token furi")
        return ruby_tokens

    if not len(split_text) == len(split_furi):
        if verbose:
            print(
                "regroup failed, failed to find heteronym and its reading in the same place in the inputs"
            )
        return ruby_tokens

    regrouped_tokens = [
        RubyFrag(text=split_text[i], furi=split_furi[i]) for i in range(len(split_text))
    ]

    if not "".join([token.furi for token in ruby_tokens]) == "".join(
        [token.furi for token in regrouped_tokens]
    ):
        if verbose:
            print(
                "regroup failed, reading of produced result does not agree with reading of input"
            )
        return ruby_tokens
    if not [token.furi for token in regrouped_tokens if token.text == heteronym] == [
        reading
    ]:
        if verbose:
            print("regroup failed, the heteronym did not get assigned the reading")
        return ruby_tokens
    return regrouped_tokens


def optimize_furigana(input_file, output_file, heteronym_dict, dictionary) -> None:
    df = pd.read_csv(input_file)  # load
    logger.info("Optimizing furigana using heteronym list and dictionary")
    for heteronym in heteronym_dict.keys():
        logger.info(f"Heteronym {heteronym} {heteronym_dict[heteronym]}")
        n_with_het = sum(df["sentence"].str.contains(heteronym))
        rows_to_rearrange = df["sentence"].str.contains(heteronym)
        optimized_rows = df.loc[rows_to_rearrange, "furigana"].apply(
            lambda s: regroup_furigana(s, heteronym, heteronym_dict, dictionary)
        )
        n_rearranged = sum(df.loc[rows_to_rearrange, "furigana"] != optimized_rows)
        logger.info(f"{n_rearranged}/{n_with_het} sentences were optimized")
        df.loc[rows_to_rearrange, "furigana"] = optimized_rows
    df.to_csv(output_file, index=False)


def remove_other_readings(input_file, output_file, heteronym_dict):
    df = pd.read_csv(input_file)  # load
    logger.info(f"Prefilter size: {len(df)}")
    df["keep_row"] = False
    for heteronym in heteronym_dict.keys():
        logger.info(heteronym)
        n_with_het = sum(df["sentence"].str.contains(heteronym))
        keep_for_het = df["furigana"].str.contains(
            r"|".join(
                [f"{{{heteronym}/{reading}}}" for reading in heteronym_dict[heteronym]]
            )
        )
        df["keep_row"] = df["keep_row"] | keep_for_het
        logger.info(
            f"Dropped {n_with_het-sum(keep_for_het)}/{n_with_het} sentences which have different readings"
        )  # TODO reword
    df = df.loc[df["keep_row"]]
    df = df.drop("keep_row", axis=1)
    df.to_csv(output_file, index=False)


def check_data(input_file) -> bool:
    df = pd.read_csv(input_file)  # load
    df["furigana-test"] = df["sentence"] == df["furigana"].apply(utils.remove_furigana)
    assert df["furigana-test"].all()
    df["sentence-standardize-test"] = df["sentence"] == df["sentence"].apply(
        utils.standardize_text
    )
    assert df["sentence-standardize-test"].all()

    return True


def split_data(data_file) -> None:
    df = pd.read_csv(data_file)  # load

    X = df["sentence"].values
    y = df["furigana"].values

    (X_train, X_val, X_test, y_train, y_val, y_test) = train_val_test_split(
        X=X,
        y=y,
        train_size=config.TRAIN_SIZE,
        val_size=config.VAL_SIZE,
        test_size=config.TEST_SIZE,
    )

    train_df = pd.DataFrame({"sentence": X_train, "furigana": y_train})
    val_df = pd.DataFrame({"sentence": X_val, "furigana": y_val})
    test_df = pd.DataFrame({"sentence": X_test, "furigana": y_test})

    train_df.to_csv(Path(config.TRAIN_DATA_DIR, "train_" + data_file.name), index=False)
    val_df.to_csv(Path(config.VAL_DATA_DIR, "val_" + data_file.name), index=False)
    test_df.to_csv(Path(config.TEST_DATA_DIR, "test_" + data_file.name), index=False)


if __name__ == "__main__":
    input_files = [
        Path(config.SENTENCE_DATA_DIR, "aozora.csv"),
        Path(config.SENTENCE_DATA_DIR, "kwdlc.csv"),
        Path(config.SENTENCE_DATA_DIR, "bccwj.csv"),
        Path(config.SENTENCE_DATA_DIR, "ndlbib.csv"),
    ]

    logger.info("Merging sentence data")
    utils.merge_csvs(input_files, Path(config.SENTENCE_DATA_DIR, "all.csv"), n_header=1)

    logger.info("Rough filtering for sentences with heteronyms")
    filter_simple(
        Path(config.SENTENCE_DATA_DIR, "all.csv"),
        Path(config.SENTENCE_DATA_DIR, "have_heteronyms_simple.csv"),
        config.HETERONYMS.keys(),
    )

    logger.info("Sudachidict filtering for out heteronyms in known compounds")
    filter_dictionary(
        Path(config.SENTENCE_DATA_DIR, "have_heteronyms_simple.csv"),
        Path(config.SENTENCE_DATA_DIR, "have_heteronyms.csv"),
        config.HETERONYMS.keys(),
        Dictionary("sudachi"),
    )

    logger.info("Optimizing furigana")
    optimize_furigana(
        Path(config.SENTENCE_DATA_DIR, "have_heteronyms.csv"),
        Path(config.SENTENCE_DATA_DIR, "optimized_heteronyms.csv"),
        config.HETERONYMS,
        Dictionary("sudachi"),
    )

    logger.info("Removing heteronyms with unexpected readings")
    remove_other_readings(
        Path(config.SENTENCE_DATA_DIR, "optimized_heteronyms.csv"),
        Path(config.SENTENCE_DATA_DIR, "optimized_strict_heteronyms.csv"),
        config.HETERONYMS,
    )

    logger.info("Running checks on data")
    test_result = check_data(
        Path(config.SENTENCE_DATA_DIR, "optimized_strict_heteronyms.csv")
    )

    logger.info("Performing train/test/split")
    split_data(Path(config.SENTENCE_DATA_DIR, "optimized_strict_heteronyms.csv"))

    logger.info("Data splits successfully generated!")
