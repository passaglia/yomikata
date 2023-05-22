"""
dictionary.py
Provides the Dictionary class which implements Reader using dictionary lookup.
"""

from difflib import ndiff

import jaconv
from chirptext import deko
from speach import ttlig
from speach.ttlig import RubyFrag, RubyToken

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
                        output += Dictionary.furi_to_ruby(surface, kana).to_code()
        output = output.replace(ASCII_SPACE_TOKEN, " ")
        return output

    @staticmethod
    def furi_to_ruby(surface, kana):
        """Combine a surface string and a kana string to a RubyToken object with furigana.

        Args:
            surface (str): Surface string
            kana (str): Kana string

        Returns:
            RubyToken: RubyToken object with furigana

        This code is modified from the version in the part of speach library: 
        https://github.com/neocl/speach/
        https://github.com/neocl/speach/blob/main/speach/ttlig.py
        :copyright: (c) 2018 Le Tuan Anh <tuananh.ke@gmail.com>
        :license: MIT
        """

        def common_substring_from_right(string1, string2):
            i = -1  # start from the end of strings
            while -i <= min(len(string1), len(string2)):
                if string1[i] != string2[i]:  # if characters don't match, break
                    break
                i -= 1  # decrement i to move towards start
            return string1[i + 1 :] if i != -1 else ""  # return common substring

        def assert_rubytoken_kana_match(ruby: RubyToken, kana: str) -> None:
            assert (
                "".join(
                    [token.furi if isinstance(token, RubyFrag) else token for token in ruby.groups]
                )
                == kana
            )

        original_kana = kana

        final_text = common_substring_from_right(surface, kana)

        if final_text:
            surface = surface[: -len(final_text)]
            kana = kana[: -len(final_text)]

        ruby = RubyToken(surface=surface)
        if deko.is_kana(surface):
            ruby.append(surface)
            if final_text:
                ruby.append(final_text)
            assert_rubytoken_kana_match(ruby, original_kana)
            return ruby

        edit_seq = ndiff(surface, kana)
        kanji = ""
        text = ""
        furi = ""
        before = ""
        expected = ""
        for item in edit_seq:
            if item.startswith("- "):
                # flush text if needed
                if expected and kanji and furi:
                    ruby.append(RubyFrag(text=kanji, furi=furi))
                    kanji = ""
                    furi = ""
                    print(ruby)
                if text:
                    ruby.append(text)
                    text = ""
                kanji += item[2:]
            elif item.startswith("+ "):
                if expected and item[2:] == expected:
                    if expected and kanji and furi:
                        ruby.append(RubyFrag(text=kanji, furi=furi))
                        kanji = ""
                        furi = ""
                    ruby.append(item[2:])
                    expected = ""
                else:
                    furi += item[2:]
            elif item.startswith("  "):
                if before == "-" and not furi:
                    # shifting happened
                    expected = item[2:]
                    furi += item[2:]
                else:
                    text += item[2:]
                    # flush if possible
                    if kanji and furi:
                        ruby.append(RubyFrag(text=kanji, furi=furi))
                        kanji = ""
                        furi = ""
                    else:
                        # possible error?
                        pass
            before = item[0]  # end for
        if kanji:
            if furi:
                ruby.append(RubyFrag(text=kanji, furi=furi))
            else:
                ruby.append(kanji)
        elif text:
            ruby.append(text)

        if final_text:
            ruby.append(final_text)

        assert_rubytoken_kana_match(ruby, original_kana)
        return ruby
