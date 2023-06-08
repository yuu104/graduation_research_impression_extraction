import CaboCha
import re
import demoji
import unicodedata
from pprint import pprint
from typing import TypedDict, List


class ImpressionWord(TypedDict):
    chunk_id: int  # 文節のインデックス
    base: int  # 形態素の原型 or 表層型
    pos: str  # 品詞
    dependent_chunk_id: int  # 係先文節のインデックス


def zenkaku_to_hankaku(text: str):
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    # 改行コード除去
    text = text.replace("\n", "").replace("\r", "")

    text = text.replace("!", "。").replace("?", "。")

    # URL除去
    text = re.sub(r"http?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", text)
    text = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", text)

    # 絵文字除去
    text = demoji.replace(string=text, repl="")

    # 半角記号除去
    text = re.compile(
        '["#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕""〈〉『』【】＆＊・（）＄＃＠。、？｀＋￥％]'
    ).sub("", text)

    # 全角記号除去
    text = re.sub(
        "[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", "", text
    )

    # スペース除去
    text = text.replace(" ", "").replace("　", "")

    return text


def main():
    cabocha = CaboCha.Parser("-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")

    split_pattern = r"[。!?]"

    text = input()

    sentence_list = re.split(split_pattern, zenkaku_to_hankaku(text=text))

    token_sentence_list: List[List[ImpressionWord]] = []
    pos_list = ["名詞", "形容詞", "動詞", "副詞"]

    for sentence in sentence_list:
        if sentence == "":
            continue

        tree = cabocha.parse(clean_text(sentence))
        token_list: List[ImpressionWord] = []
        chunk_id = -1
        chunk_link = -1

        for token_index in range(tree.size()):
            token = tree.token(token_index)
            token_feature = token.feature.split(",")
            pos = token_feature[0]

            if token.chunk is not None:
                chunk_id += 1
                chunk_link = token.chunk.link

            if pos in pos_list and not token.surface.isdigit():
                base = token_feature[6] if token_feature[6] != "*" else token.surface
                impression_word: ImpressionWord = {
                    "chunk_id": chunk_id,
                    "base": base,
                    "pos": pos,
                    "dependent_chunk_id": chunk_link,
                }
                token_list.append(impression_word)
        token_sentence_list.append(token_list)

    pprint(token_sentence_list)


if __name__ == "__main__":
    main()
