""" utils.py
A collection of utility functions used throughout the project.
"""
import json
import random
import re
import unicodedata

import numpy as np
import pynvml
from speach.ttlig import RubyFrag, RubyToken

"""
Loading and Saving Utilities
"""


def load_dict(filepath: str) -> dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(d: dict, filepath: str, cls: json.JSONEncoder = None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specified location.

    Args:
        d (Dict): data to save.
        filepath (str): location of where to save the data.
        cls (JSONEncoder, optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.

    MIT License
    Copyright (c) 2020 Made With ML

    """
    with open(filepath, "w", encoding="utf8") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys, ensure_ascii=False)
        fp.write("\n")


def merge_csvs(input_files, output_file, n_header=1) -> None:
    """Merge multiple CSVs into one. They must have the same headers.

    Args:
        input_files (list of Paths): location of csv files to merge.
        output_file (Path): location of where to save the data.
        n_header (int, optional): number of header lines to skip. Defaults to 1.
    """

    with open(output_file, "w") as f_out:
        for i, input_file in enumerate(input_files):
            with open(input_file, "r") as f_in:
                ith_header = ""
                for j in range(n_header):
                    ith_header += f_in.readline()
                if i == 0:
                    header = ith_header
                    f_out.write(header)
                else:
                    assert ith_header == header
                f_out.writelines(f_in.readlines())


"""
Seeds and GPU Utilities
"""


def set_seeds(seed: int) -> None:
    """Set seed for reproducibility.

    Args:
        seed (int, optional): number to be used as the seed. Defaults to 42.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)


def print_gpu_utilization(gpu_index: int) -> None:
    """Print gpu utilization stats

    Args:
        gpu_index (int): The PCI index of the GPU
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


"""
ML Utilities
"""


class LabelEncoder(object):
    """Label encoder for tag labels.

    MIT License
    Copyright (c) 2020 Made With ML"""

    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def get_max_token_size(dataset, tokenizer, input_feature, output_feature):
    """Get the max token size for a dataset's input and output feature given a specific tokenizer."""

    def count_tokens(entry):
        model_inputs = tokenizer(
            entry[input_feature], text_target=entry[output_feature], return_tensors="np"
        )
        return {
            input_feature + "_length": len(model_inputs["input_ids"][0]),
            output_feature + "_length": len(model_inputs["labels"][0]),
        }

    counting_dataset = dataset.map(count_tokens)

    for key in list(counting_dataset):
        print(key)
        print(input_feature + "_length")
        max_input = max(counting_dataset[key][input_feature + "_length"])
        print(max_input)
        print(output_feature + "_length")
        max_output = max(counting_dataset[key][output_feature + "_length"])
        print(max_output)

    return max_input, max_output


"""
Text and Furigana utilities
"""

UNICODE_KANJI_START = 0x4E00
UNICODE_KANJI_END = 0x9FFF

# 旧字体漢字一覧
old_kanji = "亞惡壓圍爲醫壹稻飮隱營榮衞驛悅閱圓緣艷鹽奧應橫歐毆黃溫穩假價畫會囘壞懷繪槪擴殼覺學嶽樂渴鐮勸卷寬歡罐觀閒關陷巖顏歸氣龜僞戲犧卻糺舊據擧虛峽挾敎强狹鄕堯曉區驅勳薰羣徑惠揭攜溪經繼莖螢輕鷄藝擊缺儉劍圈檢權獻縣硏險顯驗嚴吳娛效廣恆鑛號國黑歲濟碎齋劑冱櫻册雜產參慘棧蠶贊殘絲姊齒兒辭濕實舍寫釋壽收從澁獸縱肅處緖敍尙奬將牀涉燒稱證乘剩壤孃條淨狀疊穰讓釀囑觸寢愼晉眞刄盡圖粹醉隨髓數樞瀨淸靑聲靜齊稅蹟說攝竊絕專戰淺潛纖踐錢禪曾瘦雙遲壯搜插巢爭窗總聰莊裝騷增臟藏卽屬續墮體對帶滯臺瀧擇澤單擔膽團彈斷癡晝蟲鑄廳徵聽敕鎭脫遞鐵轉點傳黨盜燈當鬭德獨讀屆繩貳姙黏惱腦霸廢拜賣麥發髮拔晚蠻祕彥姬濱甁拂佛倂竝變邊辨瓣辯舖步穗寶萠襃豐沒飜槇每萬滿麵默餠歷戀戾彌藥譯豫餘與譽搖樣謠遙瑤慾來賴亂覽畧龍兩獵綠鄰凜壘淚勵禮隸靈齡曆鍊爐勞樓郞祿錄亙灣"

# 新字体漢字一覧
new_kanji = "亜悪圧囲為医壱稲飲隠営栄衛駅悦閲円縁艶塩奥応横欧殴黄温穏仮価画会回壊懐絵概拡殻覚学岳楽渇鎌勧巻寛歓缶観間関陥巌顔帰気亀偽戯犠却糾旧拠挙虚峡挟教強狭郷尭暁区駆勲薫群径恵掲携渓経継茎蛍軽鶏芸撃欠倹剣圏検権献県研険顕験厳呉娯効広恒鉱号国黒歳済砕斎剤冴桜冊雑産参惨桟蚕賛残糸姉歯児辞湿実舎写釈寿収従渋獣縦粛処緒叙尚奨将床渉焼称証乗剰壌嬢条浄状畳穣譲醸嘱触寝慎晋真刃尽図粋酔随髄数枢瀬清青声静斉税跡説摂窃絶専戦浅潜繊践銭禅曽双痩遅壮捜挿巣争窓総聡荘装騒増臓蔵即属続堕体対帯滞台滝択沢単担胆団弾断痴昼虫鋳庁徴聴勅鎮脱逓鉄転点伝党盗灯当闘徳独読届縄弐妊粘悩脳覇廃拝売麦発髪抜晩蛮秘彦姫浜瓶払仏併並変辺弁弁弁舗歩穂宝萌褒豊没翻槙毎万満麺黙餅歴恋戻弥薬訳予余与誉揺様謡遥瑶欲来頼乱覧略竜両猟緑隣凛塁涙励礼隷霊齢暦錬炉労楼郎禄録亘湾"

tr_table = str.maketrans(old_kanji, new_kanji)


def convert_old_kanji(s: str) -> str:
    """Convert kyujitai to shinjitai

    Args:
        s (str): string containing kyutijai

    Returns:
        str: string with shinjitai
    """

    return s.translate(tr_table)


def standardize_text(s: str) -> str:
    """Clean and normalize text

    Args:
        s (str): input string

    Returns:
        str: a cleaned string
    """

    # perform unicode normalization
    s = unicodedata.normalize("NFKC", s)

    # convert old kanji to new
    s = convert_old_kanji(s)

    return s.strip()


FURIMAP = re.compile(
    r"\{(?P<text>[^{}]+?)/(?P<furi>[\w%％]+?)\}"
)  # pattern prevents text from including curly braces


def parse_furigana(text: str) -> RubyToken:
    """Parse TTLRuby token (returns a RubyToken)

    Args:
        text (str): string with furigana in {<text>/<furi>} form

    Returns:
        RubyToken: RubyToken object containing parsed furigana


    MIT License

    Copyright (c) 2018 Le Tuan Anh <tuananh.ke@gmail.com>
    """
    if text is None:
        raise ValueError
    start = 0
    ruby = RubyToken(surface=text)
    ms = [(m.groupdict(), m.span()) for m in FURIMAP.finditer(text)]
    # frag: ruby fragment
    for frag, (cfrom, cto) in ms:
        if start < cfrom:
            ruby.append(text[start:cfrom])
        ruby.append(RubyFrag(text=frag["text"], furi=frag["furi"]))
        start = cto
    if start < len(text):
        ruby.append(text[start : len(text)])
    return ruby


def remove_furigana(s: str) -> str:
    """Remove furigana from a string

    Args:
        s (str): string with furigana in {<text>/<furi>} form

    Returns:
        str: string without furigana

    """
    rubytoken = parse_furigana(s)
    return "".join(
        [token.text if isinstance(token, RubyFrag) else token for token in rubytoken.groups]
    )


def furigana_to_kana(s: str) -> str:
    """Take string with furigana in {<text>/<furi>} form and replace text with furigana

    Args:
        s (str): string with {<text>/<furi>}

    Returns:
        str: string with <furi>

    """
    rubytoken = parse_furigana(s)
    return "".join(
        [token.furi if isinstance(token, RubyFrag) else token for token in rubytoken.groups]
    )


def has_kanji(s: str) -> bool:
    """Check if a string contains any kanji
    Args:
        s (str): input string

    Returns:
        bool: True if any kanji found, False otherwise
    """
    # iterate through all character codes in string
    for code in [ord(char) for char in s]:
        if code >= UNICODE_KANJI_START and code <= UNICODE_KANJI_END:
            return True
    return False
