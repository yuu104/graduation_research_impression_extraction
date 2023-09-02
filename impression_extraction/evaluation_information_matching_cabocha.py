import os
import re
from typing import List, TypedDict, Union
from enum import Enum
from pprint import pprint
import copy
import CaboCha
import pandas as pd
from dotenv import load_dotenv
from dictionary import get_evaluation_expressions, get_stopwords
from utils.folder_file import get_all_folder_names
from utils.array_dic import count_elements_in_range

load_dotenv()

current_path = os.path.dirname(os.path.abspath(__file__))

stopwords = get_stopwords()


class TokenType(Enum):
    """
    <対象, 属性, 評価表現>を表すEnum型

    Attributes
    ----------
    Subject: str
        対象
    Attribute: str
        属性
    Evaluation: str
        評価表現
    """

    Subject = "subject"
    Attribute = "attribute"
    Evaluation = "evaluation"


class Token(TypedDict):
    """
    形態素を表す型

    Attributes
    ----------
    base: str
        形態素の原型 or 表層型
    surface: str
        形態素の表層型
    pos: str
        品詞
    pos_detail: str
        品詞の詳細
    token_type: Union[TokenType, None]
        形態素が<対象, 属性, 評価表現>の中のどれに概要するか又はどれにも該当しないかを示す
    """

    base: str
    surface: str
    pos: str
    pos_detail: str
    token_type: Union[TokenType, None]


class Chunk(TypedDict):
    """
    文節を表す型

    Attributes
    ----------
    chunk_id: int
        文節のインデックス
    dependent_chunk_id: int
        係先文節のインデックス
    head_pos: int
        主辞
    func_pos: int
        機能辞
    dependent_score: float
        係り受け関係のスコア
    tokens: List[Token]
        形態素`Token`の配列
    """

    chunk_id: int
    dependent_chunk_id: int
    head_pos: int
    func_pos: int
    dependent_score: float
    tokens: List[Token]


class EvaluationInformation(TypedDict):
    """
    <対象, 属性, 評価表現>の組み合わせを表す型

    Attributes
    ----------
    sentence: str
        一文
    subject: List[Token]
        対象
    attribute: List[Token]
        属性
    evaluation: List[Token]
        評価表現
    """

    sentence: str
    subject: List[Token]
    attribute: List[Token]
    evaluation: List[Token]


class CorrelationPair(TypedDict):
    """
    相関関係を分析する2変数を表す型

    Attributes
    ----------
    match_count: int
        説明文中のキーワードとレビューの<対象, 属性>のマッチ数
    useful_count: int
        役立ち数
    match_tokens: List[str]
        マッチしたキーワード
    evaluation: List[str]
    review_text: str
        レビューテキスト
    """

    match_count: int
    useful_count: int
    match_tokens: List[str]
    evaluation: List[str]
    review_text: str


def get_token_word(token: Token) -> str:
    """
    形態素の原型 or 表層型を返す
    - 原型を優先して返す
    - 原型が存在しなければ、表層型を返す

    Parameters
    ----------
    token: Token
        形態素

    Returns
    -------
    形態素の原型 or 表層型となる文字列
    """

    return token["base"] if token["base"] else token["surface"]


def conect_compound_words(chunk: Chunk) -> Chunk:
    """
    複合語を接続する関数
    - 文節内で複合語となる形態素を結合する

    Parameters
    ----------
    chunk: Chunk
        文節

    Returns
    -------
    _: Chunk
        文節
    """

    tokens = copy.deepcopy(chunk["tokens"])
    new_tokens: List[Token] = []

    index = 0
    while index < len(tokens):
        if index == len(tokens) - 1:
            new_tokens.append(tokens[index])
            index += 1
            continue
        if (
            tokens[index + 1]["pos"] == "形容詞"
            and tokens[index + 1]["pos_detail"] == "接尾"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": None,
                    "pos": "形容詞",
                    "pos_detail": "自立",
                    "token_type": None,
                }
            )
            index += 2
        elif tokens[index]["pos"] == "動詞" and tokens[index + 1]["pos"] == "形容詞":
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": None,
                    "pos": "形容詞",
                    "pos_detail": "自立",
                    "token_type": None,
                }
            )
            index += 2
        elif (
            tokens[index + 1]["pos"] == "名詞" and tokens[index + 1]["pos_detail"] == "接尾"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "pos": "名詞",
                    "pos_detail": "一般",
                    "token_type": None,
                }
            )
            index += 2
        elif (
            tokens[index + 1]["pos"] == "動詞" and tokens[index + 1]["pos_detail"] == "接尾"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": None,
                    "pos": "動詞",
                    "pos_detail": "自立",
                    "token_type": None,
                }
            )
            index += 2
        elif (
            tokens[index]["pos"] == "接頭詞"
            and tokens[index]["pos_detail"] == "名詞接続"
            and tokens[index + 1]["pos"] == "名詞"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "pos": "名詞",
                    "pos_detail": "一般",
                    "token_type": None,
                }
            )
            index += 2
        elif (
            tokens[index]["pos"] == "名詞"
            and tokens[index + 1]["pos_detail"] == "ナイ形容詞語幹"
            and tokens[index + 1]["surface"] == "ない"
        ):
            new_tokens.append(
                {
                    "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "base": tokens[index]["surface"] + tokens[index + 1]["surface"],
                    "pos": "名詞",
                    "pos_detail": "一般",
                    "token_type": None,
                }
            )
            index += 2
        elif tokens[index + 1]["pos"] == "助動詞" and tokens[index + 1]["surface"] == "た":
            new_tokens.append(tokens[index])
            index += 2
        else:
            new_tokens.append(tokens[index])
            index += 1

    return {
        "chunk_id": chunk["chunk_id"],
        "dependent_chunk_id": chunk["dependent_chunk_id"],
        "head_pos": chunk["head_pos"],
        "func_pos": chunk["func_pos"],
        "dependent_score": chunk["dependent_score"],
        "tokens": new_tokens,
    }


def get_chunk_list(sentence: str) -> Union[List[Chunk], None]:
    """
    一文から文節のリストを生成する関数

    Parameters
    ----------
    sentence: str
        1センテンスの文字列

    Returns
    -------
    _: Union[List[Chunk], None]
        `Chunk`型の文節リスト
    """

    if sentence == "":
        return None

    cabocha = CaboCha.Parser(os.getenv("NEOLOGD_PATH"))
    tree = cabocha.parse(sentence)

    chunk_id = 0
    chunk: Chunk = {}
    chunk_list: List[Chunk] = []
    for token_index in range(tree.size()):
        token = tree.token(token_index)
        if token.chunk is not None:
            if chunk_id != 0:
                chunk_list.append(conect_compound_words(chunk=chunk))
                chunk = {}
            chunk["chunk_id"] = chunk_id
            chunk["dependent_chunk_id"] = token.chunk.link
            chunk["head_pos"] = token.chunk.head_pos
            chunk["func_pos"] = token.chunk.func_pos
            chunk["dependent_score"] = token.chunk.score
            chunk["tokens"] = []
            chunk_id += 1
        token_feature = token.feature.split(",")
        surface = token.surface
        base = token_feature[6] if token_feature[6] != "*" else None
        pos = token_feature[0]
        pos_detail = token_feature[1]
        chunk["tokens"].append(
            {
                "surface": surface,
                "base": base,
                "pos": pos,
                "pos_detail": pos_detail,
                "token_type": None,
            }
        )
    chunk_list.append(conect_compound_words(chunk=chunk))

    return chunk_list


def is_evaluation_expressions(token: Token, description_keywords: List[str]) -> bool:
    """
    引数から受け取ったトークンが評価表現であるかを判定する

    Parameters
    ----------
    token: Token
        トークン情報
    description_keywords: List[str]
        説明文から抽出したキーワード

    Returns
    -------
    _: bool
        `True`であれば評価表現
    """

    evaluation_value_expressions = get_evaluation_expressions()
    surface = token["surface"]
    base = token["base"]
    pos = token["pos"]
    pos_detail = token["pos_detail"]
    match_tokens = list(  # 評価値表見辞書に含まれる単語かどうかを調べる
        filter(
            lambda item: item in [surface, base],
            evaluation_value_expressions,
        )
    )

    if (
        pos == "形容詞"
        or (pos == "名詞" and pos_detail == "ナイ形容動詞語幹")
        or (pos == "名詞" and pos_detail == "形容動詞語幹")
    ) and not base in stopwords:
        return True
    elif len(match_tokens) and not any(
        item in match_tokens for item in description_keywords
    ):
        return True
    else:
        return False


def find_evaluation_expressions(chunk: Chunk, description_keywords: List[str]) -> None:
    """
    文節から<評価表現>を見つける関数

    Parameters
    ----------
    chunk: Chunk
        文節
    description_keywords: List[str]
        説明文から抽出したキーワード
    """

    tokens = chunk["tokens"]
    for token in tokens:
        if is_evaluation_expressions(
            token=token, description_keywords=description_keywords
        ):
            token["token_type"] = TokenType.Evaluation.value


def find_subject_attribute(chunk_list: List[Chunk]) -> None:
    """
    一文から<対象, 属性>を見つける関数

    Parameters
    ----------
    chunk_list: List[Chunk]
        文節リスト
    """

    # <評価表現> → <対象>, <評価表現> → <属性> の共起パタンに該当する<対象, 属性>　（名詞の場合は<対象>として割り当てる)
    for chunk in chunk_list:
        tokens = chunk["tokens"]
        if any(token["token_type"] == TokenType.Evaluation.value for token in tokens):
            dependent_chunk = chunk_list[chunk["dependent_chunk_id"]]
            dependent_tokens = dependent_chunk["tokens"]
            for dependent_token in dependent_tokens:
                dependent_token_word = get_token_word(token=dependent_token)
                if (
                    dependent_token["token_type"] == TokenType.Evaluation.value
                    or dependent_token_word in stopwords
                ):
                    continue
                if (
                    dependent_token["pos"] == "名詞"
                    and dependent_token["pos_detail"] != "形容動詞語幹"
                ):
                    dependent_token["token_type"] = TokenType.Subject.value
                elif dependent_token["pos"] == "動詞":
                    dependent_token["token_type"] = TokenType.Attribute.value

    # <属性> → <評価表現>, <対象> → <評価表現> の共起パタンに該当する<対象, 属性> （名詞の場合は<対象>として割り当てる)
    for chunk in chunk_list:
        tokens = chunk["tokens"]
        for token in tokens:
            token_word = get_token_word(token=token)
            if (
                token["token_type"] == TokenType.Evaluation.value
                or token_word in stopwords
            ):
                continue
            dependent_chunk = chunk_list[chunk["dependent_chunk_id"]]
            dependent_tokens = dependent_chunk["tokens"]
            if any(
                dependent_token["token_type"] == TokenType.Evaluation.value
                for dependent_token in dependent_tokens
            ):
                if token["pos"] == "名詞" and token["pos_detail"] != "形容動詞語幹":
                    token["token_type"] = TokenType.Subject.value
                elif token["pos"] == "動詞":
                    token["token_type"] = TokenType.Attribute.value

    # <対象>{の} → <属性> の共起パタンに該当する<対象, 属性>
    for chunk in chunk_list:
        tokens = chunk["tokens"]
        for index, token in enumerate(tokens):
            if (
                index != len(tokens) - 1
                and token["pos"] == "名詞"
                and tokens[index + 1]["surface"] == "の"
                and token["token_type"] != TokenType.Evaluation.value
            ):
                dependent_chunk = chunk_list[chunk["dependent_chunk_id"]]
                dependent_tokens = dependent_chunk["tokens"]
                for dependent_token in dependent_tokens:
                    if (
                        dependent_token["token_type"]
                        and dependent_token["token_type"] != TokenType.Evaluation.value
                    ):
                        token["token_type"] = TokenType.Subject.value
                        dependent_token["token_type"] = TokenType.Attribute.value

    # <対象, 属性>１ {と} <対象, 属性>2 → <評価表現> の共起パタンに該当する <対象, 属性>1
    all_tokens = [token for chunk in chunk_list for token in chunk["tokens"]]
    for index, token in enumerate(all_tokens):
        if index + 2 > len(all_tokens) - 1:
            break
        if (
            token["pos"] == "名詞"
            and all_tokens[index + 1]["surface"] == "と"
            and all_tokens[index + 2]["token_type"] == TokenType.Subject.value
        ):
            token["token_type"] = TokenType.Subject.value


def get_evaluation_information(
    chunk_list: List[Chunk], sentence: str
) -> Union[EvaluationInformation, None]:
    """
    評価情報を取得する
    - <対象, 属性>が存在しない or <評価表現>が存在しない場合は取得対象外

    Parameters
    ----------
    chunk_list: List[Chunk]
        文節`Chunk`型のリスト
    sentence: str
        1センテンスの文字列

    Returns
    -------
    _: Union[EvaluationInformation, None]
        評価情報
    """

    evaluation_information: EvaluationInformation = {
        "sentence": sentence,
        "subject": [],
        "attribute": [],
        "evaluation": [],
    }
    for chunk in chunk_list:
        for token in chunk["tokens"]:
            if token["token_type"] == TokenType.Subject.value:
                evaluation_information["subject"].append(token)
            elif token["token_type"] == TokenType.Attribute.value:
                evaluation_information["attribute"].append(token)
            elif token["token_type"] == TokenType.Evaluation.value:
                evaluation_information["evaluation"].append(token)
    return (
        evaluation_information
        if (
            len(evaluation_information["subject"])
            or len(evaluation_information["attribute"])
        )
        and len(evaluation_information["evaluation"])
        else None
    )


def get_description_keywords(chunk_list: List[Chunk]) -> List[str]:
    """
    説明文のセンテンスからキーワード（名詞・動詞）を抽出し、リストとして返す

    Parameters
    ----------
    chunk_list: List[Chunk]
        説明文の1センテンス

    Returns
    -------
    _: List[str]
        説明文から抽出したキーワード
    """

    keywords: List[str] = []
    for chunk in chunk_list:
        for token in chunk["tokens"]:
            token_word = get_token_word(token=token)
            if (
                token["pos"] in ["名詞", "動詞"]
                and token["pos_detail"] != "数"
                and not re.compile(r"^[\u3040-\u309F]$").match(token["surface"])
                and not token_word in stopwords
            ):
                keywords.append(token_word)
    return keywords


def main():
    category_name = "soup"
    item_folder_names = get_all_folder_names(
        f"{current_path}/csv/{category_name}/items"
    )

    correlation_pair: List[CorrelationPair] = []
    for item_folder_name in item_folder_names:
        # 説明文からキーワードを抽出し、データフレームを作成
        description_df = pd.read_csv(
            f"{current_path}/csv/{category_name}/items/{item_folder_name}/{item_folder_name}_description.csv",
            sep=",",
            index_col=0,
        )
        description = description_df.loc[0, "description"]
        description_sentence_list = description.split("\n")
        description_keywords: List[str] = []
        for sentence in description_sentence_list:
            if len(sentence) > 100:  # センテンスの文字数が長すぎるものは対象外
                continue
            chunk_list = get_chunk_list(sentence=sentence)
            if not chunk_list:
                continue
            description_keywords.extend(get_description_keywords(chunk_list=chunk_list))
        description_keywords = set(description_keywords)
        description_keywords_df = pd.DataFrame(
            data=list(description_keywords), columns=["keyword"]
        )
        os.makedirs(
            f"{current_path}/csv/{category_name}/evaluation_information_matching/cabocha/{item_folder_name}",
            exist_ok=True,
        )
        description_keywords_df.to_csv(
            f"{current_path}/csv/{category_name}/evaluation_information_matching/cabocha/{item_folder_name}/{item_folder_name}_description.csv"
        )

        ## レビュー文から印象情報<対象, 属性, 評価表現>を抽出し、データフレームを作成
        ## 同時に、説明文中のキーワードとレビューの<対象, 属性>のマッチ数をカウントし、相関関係を分析する2変数の辞書配列を作成
        review_df = pd.read_csv(
            f"{current_path}/csv/{category_name}/items/{item_folder_name}/{item_folder_name}_review.csv",
            sep=",",
            index_col=0,
        )
        reviews_evaluation_informations: List[EvaluationInformation] = []
        for i in range(len(review_df)):
            match_count = 0
            match_tokens: List[str] = []
            review_title = str(review_df.loc[i, "title"]).strip("\n")
            review_content = str(review_df.loc[i, "content"])
            review_sentence_list = review_content.split("\n")
            review_sentence_list.append(review_title)
            evaluation_expressions: List[str] = []
            for sentence in review_sentence_list:
                chunk_list = get_chunk_list(sentence=sentence)
                if not chunk_list:
                    continue
                for chunk in chunk_list:
                    find_evaluation_expressions(
                        chunk=chunk, description_keywords=description_keywords
                    )
                find_subject_attribute(chunk_list=chunk_list)
                evaluation_information = get_evaluation_information(
                    chunk_list=chunk_list, sentence=sentence
                )
                if evaluation_information:
                    subject_words = [
                        get_token_word(token=token)
                        for token in evaluation_information["subject"]
                    ]
                    attribute_words = [
                        get_token_word(token=token)
                        for token in evaluation_information["attribute"]
                    ]
                    review_keywords: List[str] = list(
                        set(subject_words + attribute_words)
                    )
                    for review_keyword in review_keywords:
                        if review_keyword in description_keywords:
                            match_count += 1
                            match_tokens.append(review_keyword)

                    evaluation_expressions.extend(
                        list(
                            map(
                                lambda eva_item: get_token_word(token=eva_item),
                                evaluation_information["evaluation"],
                            )
                        )
                    )
                    reviews_evaluation_informations.append(evaluation_information)
            correlation_pair.append(
                {
                    "useful_count": int(review_df.loc[i, "useful_count"]),
                    "match_count": match_count,
                    "match_tokens": match_tokens,
                    "evaluation": evaluation_expressions,
                    "review_text": review_content,
                }
            )
        reviews_evaluation_informations_token = list(
            map(
                lambda item: {
                    "sentence": item["sentence"],
                    "subject": list(
                        map(
                            lambda sub_item: get_token_word(token=sub_item),
                            item["subject"],
                        )
                    ),
                    "attribute": list(
                        map(
                            lambda att_item: get_token_word(token=att_item),
                            item["attribute"],
                        )
                    ),
                    "evaluation": list(
                        map(
                            lambda eva_item: get_token_word(token=eva_item),
                            item["evaluation"],
                        )
                    ),
                },
                reviews_evaluation_informations,
            )
        )
        reviews_evaluation_informations_token_df = pd.DataFrame(
            reviews_evaluation_informations_token
        )
        reviews_evaluation_informations_token_df.to_csv(
            f"{current_path}/csv/{category_name}/evaluation_information_matching/cabocha/{item_folder_name}/{item_folder_name}_review.csv"
        )

    ## 役立ち数ごとのレビュー数を表したデータフレームと相関関係の統計処理結果を表したデータフレームを作成
    correlation_pair_df = pd.DataFrame(data=correlation_pair)
    correlation_pair_df = correlation_pair_df.sort_values("useful_count").reset_index(
        drop=True
    )
    useful_count_values: List[int] = correlation_pair_df["useful_count"].values
    useful_count_count = [
        {
            "0": count_elements_in_range(useful_count_values, 0, 0),
            "1": count_elements_in_range(useful_count_values, 1, 1),
            "2": count_elements_in_range(useful_count_values, 2, 2),
            "3": count_elements_in_range(useful_count_values, 3, 3),
            "4": count_elements_in_range(useful_count_values, 4, 4),
            "5": count_elements_in_range(useful_count_values, 5, 5),
            "6-10": count_elements_in_range(useful_count_values, 6, 10),
            "11-20": count_elements_in_range(useful_count_values, 11, 20),
            "21-30": count_elements_in_range(useful_count_values, 21, 30),
            "31-40": count_elements_in_range(useful_count_values, 31, 40),
            "41-50": count_elements_in_range(useful_count_values, 41, 50),
            "51-": count_elements_in_range(useful_count_values, 51, None),
        }
    ]
    useful_count_count_df = pd.DataFrame(useful_count_count)
    match_token_rate = [
        {
            "0人": correlation_pair_df[correlation_pair_df["useful_count"] == 0][
                "match_count"
            ].mean(),
            "1〜2人": correlation_pair_df[
                (correlation_pair_df["useful_count"] >= 1)
                & (correlation_pair_df["useful_count"] <= 2)
            ]["match_count"].mean(),
            "3〜4人": correlation_pair_df[
                (correlation_pair_df["useful_count"] >= 3)
                & (correlation_pair_df["useful_count"] <= 4)
            ]["match_count"].mean(),
            "4〜6人": correlation_pair_df[
                (correlation_pair_df["useful_count"] >= 5)
                & (correlation_pair_df["useful_count"] <= 6)
            ]["match_count"].mean(),
            "7〜9人": correlation_pair_df[
                (correlation_pair_df["useful_count"] >= 7)
                & (correlation_pair_df["useful_count"] <= 9)
            ]["match_count"].mean(),
            "10人以上": correlation_pair_df[correlation_pair_df["useful_count"] >= 10][
                "match_count"
            ].mean(),
        }
    ]
    match_token_rate_df = pd.DataFrame(match_token_rate)
    useful_count_count_df.to_csv(
        f"{current_path}/csv/{category_name}/evaluation_information_matching/cabocha/useful_count_count.csv"
    )
    match_token_rate_df.to_csv(
        f"{current_path}/csv/{category_name}/evaluation_information_matching/cabocha/match_token_rate.csv"
    )
    correlation_pair_df.to_csv(
        f"{current_path}/csv/{category_name}/evaluation_information_matching/cabocha/correlation_pair.csv"
    )


if __name__ == "__main__":
    main()
