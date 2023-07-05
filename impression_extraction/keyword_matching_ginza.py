import spacy
from pprint import pprint
from typing import TypedDict, List, Union
import pandas as pd
from pandas import DataFrame
import os
from dotenv import load_dotenv
import re

load_dotenv()

nlp = spacy.load("ja_ginza_electra")


class ImpressionWord(TypedDict):
    base: int
    pos: str
    pos_detail: Union[str, None]


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
    token_tags = token.tag_.split("-")
    token_base = token.lemma_ if token.lemma_ else token.orth_
    pos = token_tags[0]
    pos_detail = token_tags[1] if len(token_tags) > 1 else None
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
        "られる",
        "ここ",
    ]

    if token.orth_.isdigit():  # 数字のみ
        return True

    if not pos in pos_list:
        return True

    if pos == "名詞" and (pos_detail == "数詞"):
        return True

    if re.compile(r"^[\u3040-\u309F]$").match(token.orth_):
        return True

    if token_base in stopword:
        return True


def get_tokens(text: str) -> List[ImpressionWord]:
    sentence_list = text.split("\n")

    token_list: List[ImpressionWord] = []

    for sentence in sentence_list:
        if sentence == "":
            continue

        doc = nlp(sentence)
        for token in doc:
            if is_stopword(token=token):
                continue
            token_tags = token.tag_.split("-")
            token_list.append(
                {
                    "base": token.lemma_ if token.lemma_ else token.orth_,
                    "pos": token_tags[0],
                    "pos_detail": token_tags[1] if len(token_tags) > 1 else None,
                }
            )

    return token_list


# 重複しないようにしている
def get_matching_tokens(
    description: List[ImpressionWord], review: List[ImpressionWord]
) -> List[ImpressionWord]:
    matching_tokens: List[ImpressionWord] = []
    base_values = set()

    for description_elem in description:
        for review_elem in review:
            if (
                description_elem["base"] == review_elem["base"]
                and description_elem["base"] not in base_values
            ):
                matching_tokens.append(description_elem)
                base_values.add(description_elem["base"])

    return matching_tokens


def get_correation_pair(
    description_tokens: List[ImpressionWord],
    review_df: DataFrame,
) -> List[CorrelationPair]:
    data: List[CorrelationPair] = []
    for i in range(len(review_df)):
        review = review_df.loc[i, "content"]
        review_tokens = get_tokens(text=review)
        match_tokens = get_matching_tokens(
            description=description_tokens, review=review_tokens
        )
        data.append(
            {
                "useful_count": int(review_df.loc[i, "useful_count"]),
                "count": len(match_tokens),
                "tokens": list(map(lambda item: item["base"], match_tokens)),
            }
        )
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

    correlation_pair_df.to_csv(
        f"{current_path}/csv/{category_name}/ginza/correlation_pair.csv", sep=","
    )
    useful_count_count_df.to_csv(
        f"{current_path}/csv/{category_name}/ginza/useful_count_count.csv", sep=","
    )
    match_token_rate_df = match_token_rate_df.to_csv(
        f"{current_path}/csv/{category_name}/ginza/match_token_rate.csv", sep=","
    )


if __name__ == "__main__":
    main()
