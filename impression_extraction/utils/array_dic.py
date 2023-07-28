from typing import List, Dict, Any, Optional


def remove_duplicate_dict_array_items(
    items: List[Dict[str, Any]], key_name: str
) -> List[Dict[str, Any]]:
    """
    辞書リストから指定したキーで重複を排除する
    - 重複した場合、先の要素が優先される

    Parameters
    ----------
    items: List[Dict[str, Any]]
        辞書リスト
    key_name: str
        重複を排除したいキー

    Returns
    -------
    new_items: List[Dict[str, Any]]
        指定したキーの重複を排除した辞書リスト
    """

    seen = set()
    new_items = []
    for item in items:
        key = item[key_name]
        if key not in seen:
            new_items.append(item)
            seen.add(key)
    return new_items


def count_elements_in_range(
    array: List[int], lower_limit: int, upper_limit: Optional[int]
) -> int:
    """
    数値リストにおいて、指定された範囲内に含まれる要素の数をカウントする

    Parameters
    ----------
    array: List[int]
        カウント対象の数値リスト
    lower_limit: int
        下限値
    upper_limit: Optional[int]
        上限値。指定がない場合は、下限値のみでカウントする。

    Returns
    -------
    count: int
        指定された範囲内に含まれる要素の数
    """

    count = 0
    for element in array:
        if not upper_limit:
            if element >= lower_limit:
                count += 1
        elif lower_limit <= element <= upper_limit:
            count += 1

    return count
