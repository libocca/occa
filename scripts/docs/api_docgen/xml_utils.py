'''
Helper methods for using lxml objects
'''
import os
import re
from lxml import etree as et

from .constants import *

def parse_xml_file(filename):
    '''
    Return the top-level objects from an lxml-parsed file
    '''
    tree = et.parse(
        os.path.expanduser(filename)
    )
    root = tree.getroot()

    return [tree, root]

def get_node_text(node, query):
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


def get_node_attributes(node, query=None):
    '''
    Get the attributes as a dict
    '''
    found_node = node
    if query is not None:
        found_node = node.find(query)

    if found_node is not None:
        return dict(found_node.attrib)

    return {}


def get_bool_attr(attrs, attr, default_value='no'):
    '''
    Doxygen booleans are stored 'yes' and 'no' strings
    '''
    return attrs.get(attr, default_value) == 'yes'


def get_documented_definition_nodes(root):
    '''
    Find all nodes that have an ID_TAG child at a specific depth
    This helps us only look at the documented classes/methods instead of everything
    '''
    return [*root.findall(f'.//{ID_TAG}/../../..')]


def parse_description(markdown):
    '''
    Separate the hyperlink and markdown content
    '''
    from .types import Markdown, Hyperlink

    if not markdown:
        return []

    # Searching for content inside [[...]]
    left = '[['.replace('[', r'\[')
    right = ']]'.replace(']', r'\]')

    groups = re.split(f'{left}(.*?){right}', markdown)

    return [
        Hyperlink.parse(text) if index % 2 else Markdown(text = text)
        for index, text in enumerate(groups)
    ]


def get_node_description(node):
    '''
    Find the description tag and extract its contents
    '''
    return parse_description(
        get_node_text(node, f'./*/*/{DESCRIPTION_TAG}')
    )


def get_node_instance_description(node):
    '''
    Find the instance description tag and extract its contents
    '''
    return parse_description(
        get_node_text(node, f'./*/*/{INSTANCE_DESCRIPTION_TAG}')
    )
