import CaboCha
from pprint import pprint
from typing import TypedDict, List, Union
import pandas as pd
from pandas import DataFrame
import os
from dotenv import load_dotenv
import re
import matplotlib.pyplot as plt
import seaborn as sns
from utils import remove_duplicate

load_dotenv()


class ImpressionWord(TypedDict):
    chunk_id: int  # 文節のインデックス
    base: int  # 形態素の原型 or 表層型
    pos: str  # 品詞
    pos_detail: str  # 品詞の詳細
    dependent_chunk_id: int  # 係先文節のインデックス


class CorrelationPair(TypedDict):
    useful_count: int
    count: int
    tokens: List[str]


current_path = os.path.dirname(os.path.abspath(__file__))


def get_all_folder_names(root_folder_path: str) -> List[str]:
    folder_names = []
    for root, folders, _ in os.walk(root_folder_path):
        for folder in folders:
            folder_path = os.path.join(root, folder)
            folder_names.append(folder_path.split("/")[-1])
    return folder_names


def is_stopword(token: any) -> bool:
    token_feature = token.feature.split(",")
    token_base = token_feature[6] if token_feature[6] != "*" else token.surface
    pos = token_feature[0]
    pos_detail = token_feature[1]
    pos_list = ["名詞", "形容詞", "動詞", "副詞"]
    stopword = [
        "する",
        "ある",
        "てる",
        "思う",
        "なる",
        "商品",
        "よく",
        "まず",
        "それぞれ",
        "ヶ月",
        "私",
        "そのもの",
        "次",
        "よう",
        "ない",
        "こと",
        "ため",
        "できる",
        "いる",
        "おる",
        "こちら",
        "これ",
        "くださる",
    ]

    if token.surface.isdigit():  # 数字のみ
        return True

    if not pos in pos_list:
        return True

    if pos == "名詞" and (pos_detail == "数"):
        return True

    if pos == "形容詞" and (pos_detail == "非自立" or pos_detail == "接尾"):
        return True

    if re.compile(r"^[\u3040-\u309F]$").match(token.surface):
        return True

    if token_base in stopword:
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

            if not is_stopword(token=token):
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


def get_correation_pair(
    description_tokens: List[List[ImpressionWord]],
    review_df: DataFrame,
    item_folder_name: str,
) -> List[CorrelationPair]:
    data: List[CorrelationPair] = []
    # total_match_tokens: List[ImpressionWord] = []
    for i in range(len(review_df)):
        review = review_df.loc[i, "content"]
        review_tokens = get_tokens(text=review)
        match_tokens = get_matching_tokens(
            description=description_tokens, review=review_tokens
        )
        # total_match_tokens.extend(match_tokens)
        data.append(
            {
                "useful_count": int(review_df.loc[i, "useful_count"]),
                "count": len(match_tokens),
                "tokens": list(map(lambda item: item["base"], match_tokens)),
            }
        )
    # total_match_tokens = remove_duplicate.remove_duplicate_dict_array_items(
    #     items=total_match_tokens, key_name="base"
    # )
    # total_match_tokens_df = pd.DataFrame(total_match_tokens)
    # total_match_tokens_df.to_csv(
    #     f"{current_path}/csv/{item_folder_name}/{item_folder_name}_match_tokens.csv"
    # )
    return data


def count_elements_in_range(
    array: List[int], lower_limit: int, upper_limit: Union[int, None]
) -> int:
    count = 0
    for element in array:
        if upper_limit == None:
            if element >= lower_limit:
                count += 1
            continue
        if lower_limit <= element <= upper_limit:
            count += 1
    return count


def main():
    category_name = "emulsion_cleam"
    item_folder_names = get_all_folder_names(
        f"{current_path}/csv/{category_name}/items"
    )

    correlation_pair: List[CorrelationPair] = []
    for item_folder_name in item_folder_names:
        # 説明文
        description_df = pd.read_csv(
            f"{current_path}/csv/{category_name}/items/{item_folder_name}/{item_folder_name}_description.csv",
            sep=",",
            index_col=0,
        )
        description = description_df.loc[0, "description"]
        description_tokens = get_tokens(text=description)

        # レビュー文
        review_df = pd.read_csv(
            f"{current_path}/csv/{category_name}/items/{item_folder_name}/{item_folder_name}_review.csv",
            sep=",",
            index_col=0,
        )

        correlation_pair.extend(
            get_correation_pair(
                description_tokens=description_tokens,
                review_df=review_df,
                item_folder_name=item_folder_name,
            )
        )
    correlation_pair_df = pd.DataFrame(correlation_pair)
    correlation_pair_df = correlation_pair_df.sort_values("useful_count").reset_index(
        drop=True
    )

    useful_count_values = correlation_pair_df["useful_count"].values
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
                "count"
            ].mean(),
            "1〜2人": correlation_pair_df[
                (correlation_pair_df["useful_count"] >= 1)
                & (correlation_pair_df["useful_count"] <= 2)
            ]["count"].mean(),
            "3〜4人": correlation_pair_df[
                (correlation_pair_df["useful_count"] >= 3)
                & (correlation_pair_df["useful_count"] <= 4)
            ]["count"].mean(),
            "4〜6人": correlation_pair_df[
                (correlation_pair_df["useful_count"] >= 5)
                & (correlation_pair_df["useful_count"] <= 6)
            ]["count"].mean(),
            "7〜9人": correlation_pair_df[
                (correlation_pair_df["useful_count"] >= 7)
                & (correlation_pair_df["useful_count"] <= 9)
            ]["count"].mean(),
            "10人以上": correlation_pair_df[correlation_pair_df["useful_count"] >= 10][
                "count"
            ].mean(),
        }
    ]
    match_token_rate_df = pd.DataFrame(match_token_rate)

    # data_0 = correlation_pair_df[correlation_pair_df["useful_count"] == 0]["count"]
    # print(data_0.sum(), len(data_0))
    # data_3 = correlation_pair_df[1 <= correlation_pair_df["useful_count"] <= 2]["count"]
    # print(data_3.sum(), len(data_3))
    # data_5 = correlation_pair_df[3 <= correlation_pair_df["useful_count"] <= 4]["count"]
    # print(data_5.sum(), len(data_5))
    # data_7 = correlation_pair_df[5 <= correlation_pair_df["useful_count"] <= 6]["count"]
    # print(data_7.sum(), len(data_7))
    # data_8 = data_7 = correlation_pair_df[
    #     7 <= correlation_pair_df["useful_count"] <= 9
    # ]["count"]
    # print(data_8.sum(), len(data_8))
    # data_10 = correlation_pair_df[correlation_pair_df["useful_count"] >= 10]["count"]
    # print(data_10.sum(), len(data_10))

    correlation_pair_df.to_csv(
        f"{current_path}/csv/{category_name}/correlation_pair.csv", sep=","
    )
    useful_count_count_df.to_csv(
        f"{current_path}/csv/{category_name}/useful_count_count.csv", sep=","
    )
    match_token_rate_df = match_token_rate_df.to_csv(
        f"{current_path}/csv/{category_name}/match_token_rate.csv", sep=","
    )

    # correlation_matrix = correlation_pair_df.corr()
    # print(correlation_matrix)
    # print(correlation_pair_df)

    # # correlation_pair_df.plot.scatter(x="useful_count", y="count")
    # sns.regplot(x=correlation_pair_df["useful_count"], y=correlation_pair_df["count"])

    # plt.show()


if __name__ == "__main__":
    main()
