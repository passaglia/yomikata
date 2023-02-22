"""__main__.py

Modified by Sam Passaglia from https://github.com/polm/unidic-py

This allows yomikata to download model artifacts after install.

python -m yomikata download

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
if __name__ == "__main__":
    import sys

    import plac

    from yomikata.download.download import download_version

    commands = {
        "download": download_version,
    }

    if len(sys.argv) == 1:
        print("Available commands:", ", ".join(commands))
        sys.exit(1)

    command = sys.argv.pop(1)

    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        print("Unknown command:", command)
        print("Available commands:", ", ".join(commands))
        sys.exit(1)
