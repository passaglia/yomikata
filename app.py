"""app.py
streamlit demo of yomikata"""
from pathlib import Path

import pandas as pd
import spacy
import streamlit as st
from speach import ttlig

from yomikata import utils
from yomikata.dictionary import Dictionary
from yomikata.utils import parse_furigana


@st.cache
def add_border(html: str):
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1.0rem; display: inline-block">{}</div>"""
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


def get_random_sentence():
    from yomikata.config.config import TEST_DATA_DIR

    df = pd.read_csv(Path(TEST_DATA_DIR, "test_optimized_strict_heteronyms.csv"))
    return df.sample(1).iloc[0].sentence


@st.cache
def get_dbert_prediction_and_heteronym_list(text):
    from yomikata.dbert import dBert

    reader = dBert()
    return reader.furigana(text), reader.heteronyms


@st.cache
def get_stats():
    from yomikata.config import config
    from yomikata.utils import load_dict

    stats = load_dict(Path(config.STORES_DIR, "dbert/training_performance.json"))

    global_accuracy = stats["test"]["accuracy"]

    stats = stats["test"]["heteronym_performance"]
    heteronyms = stats.keys()

    accuracy = [stats[heteronym]["accuracy"] for heteronym in heteronyms]

    readings = [
        "、".join(
            [
                "{reading} ({correct}/{n})".format(
                    reading=reading,
                    correct=stats[heteronym]["readings"][reading]["found"][reading],
                    n=stats[heteronym]["readings"][reading]["n"],
                )
                for reading in stats[heteronym]["readings"].keys()
                if (
                    stats[heteronym]["readings"][reading]["found"][reading] != 0
                    or reading != "<OTHER>"
                )
            ]
        )
        for heteronym in heteronyms
    ]

    # if reading != '<OTHER>'

    df = pd.DataFrame({"heteronym": heteronyms, "accuracy": accuracy, "readings": readings})

    df = df[df["readings"].str.contains("、")]

    df["readings"] = df["readings"].str.replace("<OTHER>", "Other")

    df = df.rename(columns={"readings": "readings (correct/total)"})

    df = df.sort_values("accuracy", ascending=False, ignore_index=True)

    df.index += 1

    return global_accuracy, df


@st.cache
def furigana_to_spacy(text_with_furigana):
    tokens = parse_furigana(text_with_furigana)
    ents = []
    output_text = ""
    heteronym_count = 0
    for token in tokens.groups:
        if isinstance(token, ttlig.RubyFrag):
            if heteronym_count != 0:
                output_text += ", "

            ents.append(
                {
                    "start": len(output_text),
                    "end": len(output_text) + len(token.text),
                    "label": token.furi,
                }
            )

            output_text += token.text
            heteronym_count += 1
        else:
            pass
    return {
        "text": output_text,
        "ents": ents,
        "title": None,
    }


st.title("Yomikata: Disambiguate Japanese Heteronyms")

# Input text box
st.markdown("Input a Japanese sentence:")

if "default_sentence" not in st.session_state:
    st.session_state.default_sentence = "え、{人間/にんげん}というものかい? {人間/にんげん}というものは{角/つの}の{生/は}えない、{生白/なまじろ}い{顔/かお}や{手足/てあし}をした、{何/なん}ともいわれず{気味/きみ}の{悪/わる}いものだよ。"

input_text = st.text_area(
    "Input a Japanese sentence:",
    utils.remove_furigana(st.session_state.default_sentence),
    label_visibility="collapsed",
)

# Yomikata prediction
dbert_prediction, heteronyms = get_dbert_prediction_and_heteronym_list(input_text)

# spacy-style output for the predictions
colors = ["#85DCDF", "#DF85DC", "#DCDF85", "#85ABDF"]
spacy_dict = furigana_to_spacy(dbert_prediction)
label_colors = {
    reading: colors[i % len(colors)]
    for i, reading in enumerate(set([item["label"] for item in spacy_dict["ents"]]))
}
html = spacy.displacy.render(spacy_dict, style="ent", manual=True, options={"colors": label_colors})

if len(spacy_dict["ents"]) > 0:
    st.markdown("**Yomikata** disambiguated the following words with multiple readings:")
    st.write(
        f"{add_border(html)}",
        unsafe_allow_html=True,
    )
else:
    st.markdown("**Yomikata** found no heteronyms in the input text.")

# Dictionary + Yomikata prediction
st.markdown("**Yomikata** can be coupled with a dictionary to get full furigana:")
dictionary = st.radio(
    "It can be coupled with a dictionary",
    ("sudachi", "unidic", "ipadic", "juman"),
    horizontal=True,
    label_visibility="collapsed",
)

dictreader = Dictionary(dictionary)
dictionary_prediction = dictreader.furigana(dbert_prediction)
html = parse_furigana(dictionary_prediction).to_html()
st.write(
    f"{add_border(html)}",
    unsafe_allow_html=True,
)

# Dictionary alone prediction
if len(spacy_dict["ents"]) > 0:
    dictionary_prediction = dictreader.furigana(utils.remove_furigana(input_text))
    html = parse_furigana(dictionary_prediction).to_html()
    st.markdown("Without **Yomikata** disambiguation, the dictionary would yield:")
    st.write(
        f"{add_border(html)}",
        unsafe_allow_html=True,
    )

# Randomize button
if st.button("🎲 Randomize the input sentence"):
    st.session_state.default_sentence = get_random_sentence()
    st.experimental_rerun()

# Stats section
global_accuracy, stats_df = get_stats()

st.subheader(
    f"**Yomikata** supports {len(stats_df)} heteronyms, with a global accuracy of {global_accuracy:.0%}!"
)

st.dataframe(stats_df)

st.subheader("Check out **Yomikata** on [GitHub](https://github.com/passaglia/yomikata) today!")

# Hide the footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
