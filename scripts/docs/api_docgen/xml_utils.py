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
    Find all nodes that have an OCCA_DOC_TAG child at a specific depth
    This helps us only look at the documented classes/methods instead of everything
    '''
    return [*root.findall(f'.//{OCCA_DOC_TAG}/../../..')]


def get_documentation_node(node):
    return node.find(f'./*/*/{OCCA_DOC_TAG}')


def expand_hyperlinks(markdown):
    '''
    Separate the hyperlink and markdown content
    '''
    from .types import Markdown, Hyperlink

    markdown = markdown.strip()

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


def get_documentation_sections(content):
    '''
    Find the sections given by header and and indentation
    For example:

    "
    Section Header 1:
      line1

      line3

    Section Header 2:
      line1

    Section Header 3:
      line1
    "

    ->

    {
      "Section Header 1": "line1\n\nline3",
      "Section Header 2": "line1",
      "Section Header 3": "line1",
    }
    '''
    content = content.strip()

    parts = re.split(f'(?:^|\n+)([^\s].*?[^\s]):\n', content)

    # Content starts with ^ so we want to ignore parts[0]
    headers = parts[1::2]
    contents = parts[2::2]

    return {
        header: expand_hyperlinks(contents[index])
        for index, header in enumerate(headers)
    }
