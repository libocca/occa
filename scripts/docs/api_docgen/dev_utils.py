"""
Helper methods for developing and debugging
"""
import json
import dataclasses
from lxml import etree as et
from typing import Any, Optional


# Print @dataclass as json
class JSONEncoder(json.JSONEncoder):
    def default(self, data):
        if not dataclasses.is_dataclass(data):
            return super().default(data)

        return dataclasses.asdict(data)


def pp_json(d: Any):
    '''
    Pretty-print JSON objects
    '''
    print(json.dumps(d, indent=4, cls=JSONEncoder))


def elem_to_json(elem: Any):
    return {
        'tag': elem.tag,
        'attributes': dict(elem.attrib),
        'children': [
            elem_to_json(child)
            for child in elem
        ]
    }


def pp_elem(elem: Optional[Any], in_json=True):
    '''
    Pretty-print lxml elements
    '''
    if elem is None:
        print('None')
        return

    if in_json:
        pp_json(
            elem_to_json(elem)
        )
    else:
       print(
           et
           .tostring(elem, pretty_print=True)
           .decode('utf-8')
       )
