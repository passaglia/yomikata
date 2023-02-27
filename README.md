# yomikata

<img src="https://github.com/passaglia/yomikata/blob/main/robot_reading.png" width=125 height=125 alt="A robot reading a book" />

**Yomikata** uses context to resolve ambiguous words in Japanese. Check out the [**interactive demo**](https://huggingface.co/spaces/passaglia/yomikata-demo)!

**Yomikata** supports 130 ambiguous forms and reaches a global accuracy of 94%. See the demo page for detailed performance information.

**Yomikata** follows the approach of [Sato et al. 2022](https://aclanthology.org/2022.lrec-1.770/) by fine-tuning the Tohoku group's [Japanese BERT transformer](https://github.com/cl-tohoku/bert-japanese) to classify words into different readings based on the sentence context. A similar approach was used in English by [Nicolis et al. 2021](https://www.amazon.science/publications/homograph-disambiguation-with-contextual-word-embeddings-for-tts-systems]).

**Yomikata** recognizes ~50% more heteronyms than Sato et al. by adding support for words which are not in the original BERT vocabulary, and it expands the original [Aozora Bunko](https://github.com/ndl-lab/huriganacorpus-aozora) and [NDL titles](https://github.com/ndl-lab/huriganacorpus-ndlbib) training data to include the [core BCCWJ corpus](https://clrd.ninjal.ac.jp/bccwj/) and the [KWDLC corpus](https://github.com/ku-nlp/KWDLC). 

# Usage

```python
from yomikata.dbert import dBert
reader = dBert()
reader.furigana('そして、畳の表は、すでに幾年前に換えられたのか分らなかった。')
# => そして、畳の{表/おもて}は、すでに幾年前に換えられたのか分らなかった。
```

This example sentence, from the short story *When I Was looking for a Room to Let* (1923) by Mimei Ogawa, contains the very common heteronym 表, which admits the readings *omote* (surface) and *hyō* (table). **Yomikata**'s dBert (disambiguation BERT) correctly determines that in this sentence it refers to the surface of a tatami mat and should be read *omote*.

The furigana function outputs the sentence with the heteronym annotated. Readings for the other words can be obtained with a simple dictionary lookup.

```python
from yomikata.dictionary import Dictionary
dictreader = Dictionary() # defaults to unidic.
dictreader.furigana("そして、畳の{表/おもて}は、すでに幾年前に換えられたのか分らなかった。")
# => そして、{畳/たたみ}の{表/おもて}は、すでに{幾年/いくねん}{前/まえ}に{換/か}えられたのか{分/わ}らなかった。
```

Without **Yomikata**, the dictionary outputs the wrong reading for the heteronym.

# Installation 

```python
pip install yomikata
python -m yomikata download
```
The second command is necessary to download the model weights, which are too large to host PyPI.

Inference should work fine on CPU.

For details on data processing and training, see the [main notebook](https://github.com/passaglia/yomikata/tree/main/notebooks).



