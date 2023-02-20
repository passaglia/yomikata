"""ndlbib.py
Data processing script for ndlbib sentence file from https://github.com/ndl-lab/huriganacorpus-ndlbib
"""

import warnings
from pathlib import Path

from pandas.errors import ParserError

from yomikata.config import config, logger
from yomikata.dataset.aozora import read_file

# ndlbib and aozora use same file structure

warnings.filterwarnings("ignore")


def ndlbib_data():
    """Extract, load and transform the ndlbib data"""

    # Extract sentences from the data files
    files = list(Path(config.RAW_DATA_DIR, "shosi").glob("*.txt"))

    with open(Path(config.SENTENCE_DATA_DIR, "ndlbib.csv"), "w") as f:
        f.write("sentence,furigana,sentenceid\n")

    for i, file in enumerate(files):
        logger.info(f"{i+1}/{len(files)} {file.name}")
        try:
            df = read_file(file)
        except ParserError:
            logger.error(f"Parser error on {file}")

        df.to_csv(
            Path(config.SENTENCE_DATA_DIR, "ndlbib.csv"),
            mode="a",
            index=False,
            header=False,
        )

    logger.info("✅ Saved ndlbib data!")


if __name__ == "__main__":
    ndlbib_data()
