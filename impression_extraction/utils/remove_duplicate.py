from typing import List


def remove_duplicate_dict_array_items(items: List, key_name: str):
    seen = set()
    new_items = []
    for item in items:
        key = item[key_name]
        if key not in seen:
            new_items.append(item)
            seen.add(key)
    return new_items
