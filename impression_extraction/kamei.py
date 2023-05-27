import spacy
import os
import csv

current_path = os.path.dirname(os.path.abspath(__file__))

nlp = spacy.load("ja_ginza")


def parse_dependency(text):
    doc = nlp(text)
    dependencies = []
    for sent in doc.sents:
        for token in sent:
            head_index = token.head.i - token.sent.start
            head_text = token.head.text
            dep_index = token.i - token.sent.start
            dep_text = token.text
            dependency = {
                "head_index": head_index,
                "head_text": head_text,
                "dep_index": dep_index,
                "dep_text": dep_text,
            }
            dependencies.append(dependency)
    return dependencies


text_column_index = 3  # テキストが含まれている列のインデックス
csv_filename = f"{current_path}/csv/01H1DSDE65SJA2T0D2BWWBQSP1/01H1DSDE65SJA2T0D2BWWBQSP1_review.csv"
with open(csv_filename, "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        text = row[text_column_index]
        dependencies = parse_dependency(text)
        print("文:", text)
        for dep in dependencies:
            if dep["head_index"] != dep["dep_index"]:
                print(
                    f"{dep['head_text']} ({dep['head_index']}) -> {dep['dep_text']} ({dep['dep_index']})"
                )
        print()
