"""
dictionary.py
Provides the Dictionary class which implements Reader using dictionary lookup.
"""

import jaconv
from speach import ttlig

from yomikata import utils
from yomikata.config.config import ASCII_SPACE_TOKEN
from yomikata.reader import Reader


class Dictionary(Reader):
    def __init__(self, tagger: str = "unidic") -> None:
        """Create a Dictionary object to apply furigana using Dictionary lookup
        Object holds configuration and tokenizer state.

        Typical usage:

        ```python
        reader = Dictionary()
        furi = Dictionary.furigana("お前はもう死んでいる")
        # "お{前/まえ}はもう{死/し}んでいる"
        ```

        Args:
            tagger (str, optional): Tokenizing dictionary to be used。 Defaults to `unidic`. `juman`, `ipadic`, 'sudachi' also possible.
        """

        if tagger == "unidic":
            import fugashi

            self.tagger = fugashi.Tagger()
            self.token_to_surface = lambda word: word.surface
            self.token_to_pos = lambda word: word.feature.pos1
            self.token_to_kana = (
                lambda word: jaconv.kata2hira(str(word))
                if (word.feature.kana == "*" or word.feature.kana is None)
                else jaconv.kata2hira(str(word.feature.kana))
            )
        elif tagger == "ipadic":
            import fugashi
            import ipadic

            self.tagger = fugashi.GenericTagger(ipadic.MECAB_ARGS)
            self.token_to_surface = lambda word: word.surface
            self.token_to_pos = lambda word: word.feature[0]
            self.token_to_kana = (
                lambda word: jaconv.kata2hira(str(word.feature[7]))
                if len(word.feature) >= 8
                else jaconv.kata2hira(str(word.surface))
            )
        elif tagger == "juman":
            import fugashi
            import jumandic

            self.tagger = fugashi.GenericTagger(jumandic.MECAB_ARGS)
            self.token_to_surface = lambda word: word.surface
            self.token_to_pos = lambda word: word.feature[0]
            self.token_to_kana = (
                lambda word: word.feature[5]
                if word.feature[5] != "*"
                else jaconv.kata2hira(str(word))
            )
        elif tagger == "sudachi":
            from sudachipy import dictionary as sudachidict
            from sudachipy import tokenizer as sudachitokenizer

            tokenizer_obj = sudachidict.Dictionary(dict="full").create()
            mode = sudachitokenizer.Tokenizer.SplitMode.C
            self.tagger = lambda s: tokenizer_obj.tokenize(s, mode)
            self.token_to_surface = lambda word: word.surface()
            self.token_to_pos = lambda word: word.part_of_speech()[0]
            self.token_to_kana = lambda word: jaconv.kata2hira(
                utils.standardize_text(str(word.reading_form()))
            )

    def furigana(self, text: str) -> str:
        text = utils.standardize_text(text)
        text = text.replace(" ", ASCII_SPACE_TOKEN)
        rubytoken = utils.parse_furigana(text)
        output = ""

        for group in rubytoken.groups:
            if isinstance(group, ttlig.RubyFrag):
                output += f"{{{group.text}/{group.furi}}}"
            else:
                group = group.replace("{", "").replace("}", "")
                for word in self.tagger(group):
                    kana = self.token_to_kana(word)
                    surface = self.token_to_surface(word)
                    pos = self.token_to_pos(word)
                    if (surface == kana) or pos in ["記号", "補助記号", "特殊"]:
                        output += surface
                    else:
                        output += ttlig.RubyToken.from_furi(surface, kana).to_code()
        output = output.replace(ASCII_SPACE_TOKEN, " ")
        return output
