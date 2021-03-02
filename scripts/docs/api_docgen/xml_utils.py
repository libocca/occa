'''
Helper methods for using lxml objects
'''
import os
import re
from lxml import etree as et
from typing import Any, Dict, Optional, Tuple, Union

from .constants import *


def parse_xml_file(filename: str) -> Tuple[Any, Any]:
    '''
    Return the top-level objects from an lxml-parsed file
    '''
    tree = et.parse(
        os.path.expanduser(filename)
    )
    root = tree.getroot()

    return (tree, root)

def get_node_text(node: Any, query: str) -> str:
    '''
    Get the node text or default to ''
    Note: This does not fetch the full text when there is a mixed content, for example:
       <tag>
         a <b></b> c <d></d>
       </tag>
    '''
    found_node = node.find(query)
    if found_node is not None:
        return found_node.text.strip()

    return ''


def get_node_attributes(node: Any, query: Optional[str]=None):
    '''
    Get the attributes as a dict
    '''
    found_node = node
    if query is not None:
        found_node = node.find(query)

    if found_node is not None:
        return dict(found_node.attrib)

    return {}


def get_bool_attr(attrs: Any, attr: str, default_value :Optional[str]='no'):
    '''
    Doxygen booleans are stored 'yes' and 'no' strings
    '''
    return attrs.get(attr, default_value) == 'yes'


def get_documented_definition_nodes(root: Any):
    '''
    Find all nodes that have an OCCA_DOC_TAG child at a specific depth
    This helps us only look at the documented classes/methods instead of everything
    '''
    return [*root.findall(f'.//{OCCA_DOC_TAG}/../../..')]


def get_documentation_node(node: Any):
    return node.find(f'./*/*/{OCCA_DOC_TAG}')
