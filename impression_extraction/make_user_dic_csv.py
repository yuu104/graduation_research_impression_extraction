import os
from typing import List
import csv


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    input_file_path = f"{current_path}/dic/EVALDIC_ver1.01"
    output_file_path = f"{current_path}/dic/evaluation_value_expression_dic.csv"

    dic_form_list: List[str] = []
    with open(input_file_path, "r") as f:
        for line in f:
            keyword = line.strip()
            dic_form = f"{keyword},,,1,名詞,一般,*,*,*,*,*,*,*,評価値表現辞書"
            dic_form_list.append(dic_form)

    with open(output_file_path, "w") as f:
        writer = csv.writer(f)
        for dic_form in dic_form_list:
            writer.writerow(dic_form.split(","))


if __name__ == "__main__":
    main()
