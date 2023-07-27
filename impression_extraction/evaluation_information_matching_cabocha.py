import os
from typing import List, TypedDict, Union
from enum import Enum
from pprint import pprint
import copy
import CaboCha
import pandas as pd
from dotenv import load_dotenv
from evaluation_expressions_dic import get_evaluation_expressions
from utils.folder_file import get_all_folder_names

load_dotenv()

current_path = os.path.dirname(os.path.abspath(__file__))

evaluation_expressions = get_evaluation_expressions()


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
    <対象, 属性, 評価表現>の組み合わせを表すクラス

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
        # elif (
        #     tokens[index]["pos"] == "名詞"
        #     and tokens[index]["pos_detail"] == "接尾"
        #     and tokens[index + 1]["pos"] == "名詞"
        # ):
        #     new_tokens.append(
        #         {
        #             "surface": tokens[index]["surface"] + tokens[index + 1]["surface"],
        #             "base": tokens[index]["surface"] + tokens[index + 1]["surface"],
        #             "pos": "名詞",
        #             "pos_detail": "一般",
        #             "token_type": None,
        #         }
        #     )
        #     index += 2
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


def is_evaluation_expressions(token: Token) -> bool:
    surface = token["surface"]
    base = token["base"]
    pos = token["pos"]
    pos_detail = token["pos_detail"]
    match_tokens = list(
        filter(
            lambda item: item.get("word") in [surface, base]
            and item["pos"] == pos
            and item["pos_detail"] == pos_detail,
            evaluation_expressions,
        )
    )

    if (
        pos == "形容詞"
        or (pos == "名詞" and pos_detail == "ナイ形容動詞語幹")
        or (pos == "名詞" and pos_detail == "形容動詞語幹")
    ):
        return True
    elif len(match_tokens):
        return True
    else:
        return False


def find_evaluation_expressions(chunk: Chunk) -> None:
    """
    文節から<評価表現>を見つける関数

    Parameters
    ----------
    chunk: Chunk
        文節
    """

    tokens = chunk["tokens"]
    for token in tokens:
        if is_evaluation_expressions(token=token):
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
                if dependent_token["token_type"] == TokenType.Evaluation.value:
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
            if token["token_type"] == TokenType.Evaluation.value:
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


def main():
    category_name = "emulsion_cleam"
    item_folder_names = get_all_folder_names(
        f"{current_path}/csv/{category_name}/items"
    )

    for item_folder_name in item_folder_names:
        # 説明文
        description_df = pd.read_csv(
            f"{current_path}/csv/{category_name}/items/{item_folder_name}/{item_folder_name}_description.csv",
            sep=",",
            index_col=0,
        )
        description = description_df.loc[0, "description"]
        description_sentence_list = description.split("\n")
        description_evaluation_informations: List[EvaluationInformation] = []
        for sentence in description_sentence_list:
            if len(sentence) > 100:  # センテンスの文字数が長すぎるものは対象外
                continue
            chunk_list = get_chunk_list(sentence=sentence)
            if not chunk_list:
                continue
            for chunk in chunk_list:
                find_evaluation_expressions(chunk=chunk)
            find_subject_attribute(chunk_list=chunk_list)
            evaluation_information = get_evaluation_information(
                chunk_list=chunk_list, sentence=sentence
            )
            if evaluation_information:
                description_evaluation_informations.append(evaluation_information)
        # description_evaluation_informations_df = pd.DataFrame(
        #     description_evaluation_informations
        # )

        # description_evaluation_informations_df.to_csv(
        #     f"{current_path}/csv/{category_name}/evaluation_information_matching/cabocha/{item_folder_name}/{item_folder_name}_description.csv"
        # )

        # # レビュー文
        review_df = pd.read_csv(
            f"{current_path}/csv/{category_name}/items/{item_folder_name}/{item_folder_name}_review.csv",
            sep=",",
            index_col=0,
        )
        reviews_evaluation_informations: List[EvaluationInformation] = []
        for i in range(len(review_df)):
            review = review_df.loc[i, "content"]
            review_sentence_list = review.split("\n")
            for sentence in review_sentence_list:
                chunk_list = get_chunk_list(sentence=sentence)
                if not chunk_list:
                    continue
                for chunk in chunk_list:
                    find_evaluation_expressions(chunk=chunk)
                find_subject_attribute(chunk_list=chunk_list)
                evaluation_information = get_evaluation_information(
                    chunk_list=chunk_list, sentence=sentence
                )
                if evaluation_information:
                    reviews_evaluation_informations.append(evaluation_information)

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

        os.makedirs(
            f"{current_path}/csv/{category_name}/evaluation_information_matching/cabocha/{item_folder_name}"
        )
        reviews_evaluation_informations_token_df.to_csv(
            f"{current_path}/csv/{category_name}/evaluation_information_matching/cabocha/{item_folder_name}/{item_folder_name}_review.csv"
        )


if __name__ == "__main__":
    main()
