from pyknp import Juman, KNP
from typing import List, TypedDict
import os
import pandas as pd
from pprint import pprint
import re
import demoji
import spacy
import ginza


class Mrph(TypedDict):
    genkei: str
    hinsi: str


def clean_text(text: str) -> str:
    # 改行コード除去
    text = text.replace("\n", "").replace("\r", "")

    # URL除去
    text = re.sub(r"http?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", text)
    text = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", text)

    # 絵文字除去
    text = demoji.replace(string=text, repl="")

    # 半角記号除去
    text = re.compile(
        '[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕""〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]'
    ).sub("", text)

    # 全角記号除去
    text = re.sub(
        "[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", "", text
    )

    # スペース除去
    text = text.replace(" ", "").replace("　", "")

    return text


def morphological_analysis(text: str) -> List[Mrph]:
    jumanpp = Juman()
    result = jumanpp.analysis(clean_text(text=text))

    mrph_list = [
        {"genkei": mrph.genkei, "hinsi": mrph.hinsi} for mrph in result.mrph_list()
    ]

    return mrph_list


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))

    # description_df = pd.read_csv(
    #     f"{current_path}/csv/01H15YBTM7E7FMQ8377JF9TTQA/01H15YBTM7E7FMQ8377JF9TTQA_description.csv",
    #     sep=",",
    #     index_col=0,
    # )
    # description = description_df.loc[0, "description"]

    review_df = pd.read_csv(
        f"{current_path}/csv/01H1DSDE65SJA2T0D2BWWBQSP1/01H1DSDE65SJA2T0D2BWWBQSP1_review.csv",
        sep=",",
        index_col=0,
    )

    # description_mrphs = morphological_analysis(text=description)

    # review_mrphs_list = []
    # for index in range(len(review_df)):
    #     review_text = review_df.loc[index, "title"] + review_df.loc[index, "content"]
    #     review_mrphs = morphological_analysis(review_text)
    #     review_mrphs_list.append(review_mrphs)
    # pprint(review_mrphs_list[:5])

    review_0 = review_df.loc[0, "title"] + review_df.loc[0, "content"]

    # mrphs = morphological_analysis(text=clean_text(review_0))
    # pprint(mrphs)

    nlp = spacy.load("ja_ginza")
    doc = nlp(clean_text(review_0))

    for bunsetu in ginza.bunsetu_spans(doc):
        print("文節：", bunsetu.text)
        for token in bunsetu.lefts:
            print(token)


if __name__ == "__main__":
    main()
