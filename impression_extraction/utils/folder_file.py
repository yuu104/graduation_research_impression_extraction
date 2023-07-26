from typing import List
import os


def get_all_folder_names(root_folder_path: str) -> List[str]:
    """
    指定したパス直下にあるフォルダ名を全て返す

    Parameters
    ----------
    root_folder_path: str
      取得したいフォルダ名までのパス

    Returns
    -------
    _: List[str]
    引数から受け取ったパス直下にあるフォルダ名のリスト
    """

    folder_names = []
    for root, folders, _ in os.walk(root_folder_path):
        for folder in folders:
            folder_path = os.path.join(root, folder)
            folder_names.append(folder_path.split("/")[-1])
    return folder_names
