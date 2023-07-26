from typing import List, TypedDict, Union
import os

current_path = os.path.dirname(os.path.abspath(__file__))


class EvaluationExpressionsDic(TypedDict):
    """
    「評価値表現辞書」で取得した評価表現の形態素

    Attributes
    ----------
    word: str
        単語
    pos: Union[str, None]
        品詞
    pos_detail: Union[str, None]
        品詞詳細
    """

    word: str
    pos: Union[str, None]
    pos_detail: Union[str, None]


def get_evaluation_expressions() -> List[EvaluationExpressionsDic]:
    """
    「評価値表現辞書」から評価表現を取得し、配列を返す
    - 複数の形態素を持つ評価表現の場合、最初の形態素を抽出する
    - 評価表現の重複は排除
    """

    evaluation_expressions: List[EvaluationExpressionsDic] = []
    with open(f"{current_path}/dic/EVALDIC_ver1.01+POS", "r") as f:
        for index, line in enumerate(f):
            tokens = line.split(" ")
            if index <= 8 or len(tokens) > 1:
                continue
            splitted = tokens[0].split("+")
            splitted_2 = splitted[1].rstrip().split("-")
            splitted_2_len = len(splitted_2)
            evaluation_expressions.append(
                {
                    "word": splitted[0],
                    "pos": splitted_2[0] if splitted_2_len >= 1 else None,
                    "pos_detail": splitted_2[1] if splitted_2_len >= 2 else None,
                }
            )
    return list({item["word"]: item for item in evaluation_expressions}.values())
