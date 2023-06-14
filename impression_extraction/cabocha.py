import CaboCha
from pprint import pprint
from typing import TypedDict, List
import pandas as pd
import os
from dotenv import load_dotenv
import re
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()


class ImpressionWord(TypedDict):
    chunk_id: int  # 文節のインデックス
    base: int  # 形態素の原型 or 表層型
    pos: str  # 品詞
    pos_detail: str  # 品詞の詳細
    dependent_chunk_id: int  # 係先文節のインデックス


def is_unwanted_token(token: any) -> bool:
    token_feature = token.feature.split(",")
    pos = token_feature[0]
    pos_detail = token_feature[1]
    pos_list = ["名詞", "形容詞", "動詞", "副詞"]

    if token.surface.isdigit():  # 数字のみ
        return True

    if not pos in pos_list:
        return True

    if pos == "形容詞" and (pos_detail == "非自立" or pos_detail == "接尾"):
        return True

    if re.compile(r"^[\u3040-\u309F]$").match(token.surface):
        return True


def get_tokens(text: str) -> List[List[ImpressionWord]]:
    cabocha = CaboCha.Parser(os.getenv("NEOLOGD_PATH"))
    sentence_list = text.split("\n")

    token_sentence_list: List[List[ImpressionWord]] = []

    for sentence in sentence_list:
        if sentence == "":
            continue

        tree = cabocha.parse(sentence)
        token_list: List[ImpressionWord] = []
        chunk_id = -1
        chunk_link = -1

        for token_index in range(tree.size()):
            token = tree.token(token_index)
            token_feature = token.feature.split(",")
            pos = token_feature[0]
            pos_detail = token_feature[1]

            if token.chunk is not None:
                chunk_id += 1
                chunk_link = token.chunk.link

            if not is_unwanted_token(token=token):
                base = token_feature[6] if token_feature[6] != "*" else token.surface
                impression_word: ImpressionWord = {
                    "chunk_id": chunk_id,
                    "base": base,
                    "pos": pos,
                    "pos_detail": pos_detail,
                    "dependent_chunk_id": chunk_link,
                }
                token_list.append(impression_word)
        token_sentence_list.append(token_list)

    return token_sentence_list


# 重複しないようにしている
def get_matching_tokens(
    description: List[List[ImpressionWord]], review: List[List[ImpressionWord]]
) -> List[ImpressionWord]:
    flat_description = [elem for sublist in description for elem in sublist]
    flat_review = [elem for sublist in review for elem in sublist]

    matching_tokens: List[ImpressionWord] = []
    base_values = set()

    for description_elem in flat_description:
        for review_elem in flat_review:
            if (
                description_elem["base"] == review_elem["base"]
                and description_elem["base"] not in base_values
            ):
                matching_tokens.append(description_elem)
                base_values.add(description_elem["base"])

    return matching_tokens


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))

    # 説明文
    description_df = pd.read_csv(
        f"{current_path}/csv/01H2D1FEAJY2NTHM09WE1CH092/01H2D1FEAJY2NTHM09WE1CH092_description.csv",
        sep=",",
        index_col=0,
    )
    description = description_df.loc[0, "description"]
    description_tokens = get_tokens(text=description)

    # レビュー文
    review_df = pd.read_csv(
        f"{current_path}/csv/01H2D1FEAJY2NTHM09WE1CH092/01H2D1FEAJY2NTHM09WE1CH092_review.csv",
        sep=",",
        index_col=0,
    )
    review_df_sorted = review_df.sort_values(
        "useful_count", ascending=False
    ).reset_index(drop=True)
    index = 144
    # print("役立ち数", review_df_sorted.loc[index, "useful_count"])
    # print("星の数", review_df_sorted.loc[index, "rating"])
    review = review_df_sorted.loc[index, "content"]
    review_tokens = get_tokens(text=review)

    # マッチした形態素
    match_tokens = get_matching_tokens(
        description=description_tokens, review=review_tokens
    )

    data = []
    for i in range(len(review_df)):
        review = review_df_sorted.loc[i, "content"]
        review_tokens = get_tokens(text=review)
        match_tokens = get_matching_tokens(
            description=description_tokens, review=review_tokens
        )
        data.append(
            {
                "useful_count": int(review_df_sorted.loc[i, "useful_count"]),
                "count": len(match_tokens),
            }
        )
    data_df = pd.DataFrame(data)
    correlation_matrix = data_df.corr()
    print(correlation_matrix)

    # data_df.plot.scatter(x="useful_count", y="count")
    sns.regplot(x=data_df["useful_count"], y=data_df["count"])

    plt.show()


if __name__ == "__main__":
    main()
