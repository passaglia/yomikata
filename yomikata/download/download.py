"""download.py

Modified by Sam Passaglia from https://github.com/polm/unidic-py

MIT License

Copyright (c) 2020 Paul O'Leary McCann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import os
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import requests
from tqdm import tqdm

from yomikata.config import config, logger


# This is used to show progress when downloading.
# see here: https://github.com/tqdm/tqdm#hooks-and-callbacks
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_file(url, fname):
    with requests.get(url, stream=True) as r:
        with open(fname, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    return fname


def download_progress(url, fname):
    """Download a file and show a progress bar."""
    with TqdmUpTo(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:  # all optional kwargs
        urlretrieve(url, filename=fname, reporthook=t.update_to, data=None)
        t.total = t.n
    return fname


def get_json(url, desc):
    r = requests.get(url)
    if r.status_code != 200:
        logger.error(
            "Server error ({})".format(r.status_code),
            "Couldn't fetch {}. If this error persists please open an issue."
            " http://github.com/passaglia/yomikata/issues/".format(desc),
            exits=1,
        )
    return r.json()


def download_and_clean(version, url):
    """Download and position the folder.
    This downloads the tar file from the source, extracts it, renames the
    resulting directory, and removes large files not used at runtime.
    """

    cdir = config.YOMIKATA_DIR
    fname = Path(cdir, "dbert-artifacts.tar.gz")
    print("Downloading dBert v{}...".format(version), file=sys.stderr)
    download_progress(url, fname)
    print("Finished download.")

    dbertdir = os.path.join(cdir, "dbert-artifacts")
    if os.path.isdir(dbertdir):
        shutil.rmtree(dbertdir)

    with tarfile.open(fname, "r") as tf:
        tf.extractall(cdir)
    os.remove(fname)

    print("Downloaded dbert v{} to {}".format(version, dbertdir), file=sys.stderr)


DOWNLOAD_INFO = "https://raw.githubusercontent.com/passaglia/yomikata/master/downloads.json"


def download_version(ver="latest"):
    res = get_json(DOWNLOAD_INFO, "download info")
    try:
        dictinfo = res[ver]
    except KeyError:
        print('Unknown version "{}".'.format(ver))
        print("Known versions:")
        for key, val in res.items():
            print("\t", key, "({})".format(val["version"]))

    print("download url:", dictinfo["url"])
    print("Model version:", dictinfo["version"])
    download_and_clean(dictinfo["version"], dictinfo["url"])
