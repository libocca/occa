'''
Miscellaneous utility methods
'''
import re
from typing import List


def split_by_whitespace(text: str) -> List[str]:
    '''
    Split by any amount of whitespace
    '''
    if not text:
        return []

    return [
        word
        for word in re.split(r'\s+', text)
        if word
    ]


def nested_get(obj: dict, path: List[str]) -> dict:
    '''
    Traverse a nested path in `obj` and create dict()
    for any keys in the path that don't exist

    Return the very last object fetched or created
    '''
    ptr = obj
    for key in path:
        if key not in ptr:
            ptr[key] = {}
        ptr = ptr[key]
    return ptr
