import CaboCha
import re
import demoji
import unicodedata
from pprint import pprint


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

    token_sentence_list = []
    path_of_speech_list = ["名詞", "形容詞", "動詞", "副詞"]

    for sentence in sentence_list:
        if sentence == "":
            continue
        parsed = cabocha.parse(clean_text(sentence))
        token_list = []
        for i in range(parsed.size()):
            token = parsed.token(i)
            token_feature = token.feature.split(",")
            path_of_speech = token_feature[0]
            if path_of_speech in path_of_speech_list and not token.surface.isdigit():
                word = token_feature[6] if token_feature[6] != "*" else token.surface
                token_list.append(f"{word} {path_of_speech} {token_feature[1]}")
        token_sentence_list.append(token_list)

    pprint(token_sentence_list)


if __name__ == "__main__":
    main()
