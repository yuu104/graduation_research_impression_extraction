from typing import List
import os

current_path = os.path.dirname(os.path.abspath(__file__))


def get_evaluation_expressions() -> List[str]:
    """
    「評価値表現辞書」から評価表現を取得し、配列を返す
    - 評価表現の重複は排除

    Returns
    -------
    tokens: List[str]
        形態素の原型 or 表層型となる文字列
    """

    tokens: List[str] = []
    with open(f"{current_path}/dic/EVALDIC_ver1.01", "r") as f:
        for line in f:
            token = line.strip()
            tokens.append(token)
    return tokens


def get_stopwords() -> List[str]:
    """
    ストップワード辞書からストップワードを取得し、配列を返す
    - 重複は除去

    Returns
    -------
    _: List[str]
    ストップワードのリスト
    """

    stopwords: List[str] = []
    with open(f"{current_path}/dic/stopword.txt", "r") as f:
        for line in f:
            stopwords.append(line.rstrip())
    return list(set(stopwords))
