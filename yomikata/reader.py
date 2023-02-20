""" reader.py
An abstract class for assigning readings to Japanese sentences.
"""
import abc


class Reader(abc.ABC):
    @abc.abstractmethod
    def furigana(self, text: str) -> str:
        """Add furigana to Japanese text

        Args:
            text (str): a sentence in Japanese

        Returns:
            str: sentence annotated with furigana

        """
        pass
