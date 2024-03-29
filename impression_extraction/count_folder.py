import os


def count_subfolders(folder_path):
    folder_count = 0

    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            folder_count += 1

    return folder_count


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 対象のフォルダのパスを指定
    folder_path = f"{current_path}/csv/soup/evaluation_information_matching"
    subfolder_count = count_subfolders(folder_path)
    print(f"Subfolder count: {subfolder_count}")
