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


class Tokens(TypedDict):
    text: str
    tokens: List[str]


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
        "今",
    ]

    if token.is_stop and token.orth_ != "いい":
        return True

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
    item_folder_name = item_folder_names[1]

    # 説明文からキーワードを抽出
    description_df = pd.read_csv(
        f"{current_path}/csv/{category_name}/items/{item_folder_name}/{item_folder_name}_description.csv",
        sep=",",
        index_col=0,
    )
    description = description_df.loc[0, "description"]
    description_tokens = get_tokens(text=description)
    description_keywords: Tokens = [
        {
            "text": description,
            "tokens": list(map(lambda x: x["base"], description_tokens)),
        }
    ]
    # description_noun_tokens = list(
    #     map(lambda y: y["base"], filter(lambda x: x["pos"] == "名詞", description_tokens))
    # )
    # # pprint(description_noun_tokens)

    # レビューからキーワードを抽出
    review_df = pd.read_csv(
        f"{current_path}/csv/{category_name}/items/{item_folder_name}/{item_folder_name}_review.csv",
        sep=",",
        index_col=0,
    )
    review_keywords_list: List[Tokens] = []
    for i in range(len(review_df)):
        review = review_df.loc[i, "content"]
        review_tokens = get_tokens(text=review)
        review_keywords: Tokens = {
            "text": review,
            "tokens": list(map(lambda x: x["base"], review_tokens)),
        }
        review_keywords_list.append(review_keywords)

    description_keywords_df = pd.DataFrame(description_keywords)
    description_keywords_df.to_csv(
        f"{current_path}/csv/{category_name}/ginza/item_tokens/description.csv",
        sep=",",
    )
    review_keywords_df = pd.DataFrame(review_keywords_list)
    review_keywords_df.to_csv(
        f"{current_path}/csv/{category_name}/ginza/item_tokens/review.csv",
        sep=",",
    )


if __name__ == "__main__":
    main()
