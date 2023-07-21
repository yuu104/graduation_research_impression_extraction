import os
from typing import List, TypedDict, Union
from enum import Enum
from pprint import pprint
import copy
import CaboCha
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

current_path = os.path.dirname(os.path.abspath(__file__))


# 対象・属性・評価表現を表すEnum型
class TokenType(Enum):
    Subject = "subject"  # 対象
    Attribute = "attribute"  # 属性
    Evaluation = "evaluation"  # 評価表現


# 形態素クラス
class Token(TypedDict):
    base: str  # 形態素の原型 or 表層型
    surface: str
    pos: str  # 品詞
    pos_detail: str  # 品詞の詳細
    token_type: Union[TokenType, None]


# 文節クラス
class Chunk(TypedDict):
    chunk_id: int  # 文節のインデックス
    dependent_chunk_id: int  # 係先文節のインデックス
    head_pos: int  # 主辞
    func_pos: int  # 機能辞
    dependent_score: float  # 係り受け関係のスコア
    tokens: List[Token]


# <対象, 属性, 評価表現>の組み合わせを表すクラス
class EvaluationInformation(TypedDict):
    sentence: str  # 一文
    subject: List[Token]  # 対象
    attribute: List[Token]  # 属性
    evaluation: List[Token]  # 評価表現


def get_all_folder_names(root_folder_path: str) -> List[str]:
    folder_names = []
    for root, folders, _ in os.walk(root_folder_path):
        for folder in folders:
            folder_path = os.path.join(root, folder)
            folder_names.append(folder_path.split("/")[-1])
    return folder_names


# 複合語を接続する関数
def conect_compound_words(chunk: Chunk) -> Chunk:
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
        #     pprint(tokens[index])
        #     pprint(tokens[index + 1])
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


# 一文から`Chunk`型のリストを生成する関数
def get_chunk_list(sentence: str) -> Union[List[Chunk], None]:
    cabocha = CaboCha.Parser(os.getenv("NEOLOGD_PATH"))

    if sentence == "":
        return None

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
    pos = token["pos"]
    pos_detail = token["pos_detail"]

    if (
        pos == "形容詞"
        or (pos == "名詞" and pos_detail == "ナイ形容動詞語幹")
        or (pos == "名詞" and pos_detail == "形容動詞語幹")
    ):
        return True
    else:
        return False


# 文節から<評価表現>を見つける関数
def find_evaluation_expressions(chunk: Chunk) -> None:
    tokens = chunk["tokens"]
    for token in tokens:
        if is_evaluation_expressions(token=token):
            token["token_type"] = TokenType.Evaluation.value


# 一文から<対象, 属性>を見つける関数
def find_subject_attribute(chunk_list: List[Chunk]) -> None:
    # <評価表現> → <対象>, <評価表現> → <属性> の共起パタンに該当する<対象, 属性>　（名詞の場合は<対象>として割り当てる)
    for chunk in chunk_list:
        tokens = chunk["tokens"]
        if any(token["token_type"] == TokenType.Evaluation.value for token in tokens):
            dependent_chunk = chunk_list[chunk["dependent_chunk_id"]]
            dependent_tokens = dependent_chunk["tokens"]
            for dependent_token in dependent_tokens:
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
            ):
                token["token_type"] = TokenType.Subject.value
                dependent_chunk = chunk_list[chunk["dependent_chunk_id"]]
                dependent_tokens = dependent_chunk["tokens"]
                for dependent_token in dependent_tokens:
                    if dependent_token["pos"] == "名詞" or dependent_token["pos"] == "動詞":
                        dependent_token["token_type"] = TokenType.Attribute.value


def get_evaluation_information(
    chunk_list: List[Chunk], sentence: str
) -> Union[EvaluationInformation, None]:
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
        if len(evaluation_information["subject"])
        or len(evaluation_information["attribute"])
        or len(evaluation_information["evaluation"])
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
                            lambda sub_item: sub_item["base"]
                            if sub_item["base"]
                            else sub_item["surface"],
                            item["subject"],
                        )
                    ),
                    "attribute": list(
                        map(
                            lambda att_item: att_item["base"]
                            if att_item["base"]
                            else att_item["surface"],
                            item["attribute"],
                        )
                    ),
                    "evaluation": list(
                        map(
                            lambda eva_item: eva_item["base"]
                            if eva_item["base"]
                            else eva_item["surface"],
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
