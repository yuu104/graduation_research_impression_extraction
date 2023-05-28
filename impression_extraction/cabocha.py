import CaboCha
import re
import demoji


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

    text = "1本でも十分満足感があります。"

    parsed = cabocha.parse(text)

    print(parsed.toString(CaboCha.FORMAT_LATTICE))


if __name__ == "__main__":
    main()
