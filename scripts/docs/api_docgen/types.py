'''
Basic types
'''
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .constants import *
from .utils import *
from .xml_utils import *


Content = Union['Markdown', 'Hyperlink']

@dataclass
class Documentation:
    # Automatic from Doxygen
    filepath: str
    line_number: str

    # Manually provided
    id_: str
    id_index: Optional[int]
    sections: Dict[str, List[Content]]

    @staticmethod
    def parse(node):
        location = get_node_attributes(node, f'./location')
        filepath = location['file']
        line_number = location['line']

        doc_node = get_documentation_node(node)
        [id_, id_index] = Documentation.parse_id(doc_node)


        return Documentation(
            filepath=filepath,
            line_number=line_number,
            id_=id_,
            id_index=id_index,
            sections=get_documentation_sections(doc_node.text)
        )

    @staticmethod
    def parse_id(doc_node):
        full_id = doc_node.attrib['id']
        if '[' not in full_id:
            # If there is only 1, default to 0 index
            return [full_id, 0]

        # Searching for something like "constructor[0]" (<id>[<id_index>])
        id_pattern = r'(?P<id>[a-zA-Z0-9_]+)'
        id_index_pattern = r'\[(?P<id_index>[0-9]+)\]'

        m = re.match(id_pattern + id_index_pattern, full_id)
        return [m['id'], int(m['id_index'])]

    def get_id(self):
        return self.id_ or self.alias_id

    def is_alias(self):
        return self.id_index > 0

@dataclass
class Markdown:
    text: str

    def to_string(self):
        return text

@dataclass
class Hyperlink:
    node_id: str
    text: str

    @staticmethod
    def parse(text):
        # Format:
        #   [[device.malloc]]
        #   [[malloc|device.malloc]]
        text = text.strip()
        parts = text.split('|')

        if len(parts) == 1:
            return Hyperlink(
                node_id=text,
                text=text,
            )

        return Hyperlink(
            node_id=parts[1].strip(),
            text=parts[0].strip(),
        )

    def to_string(self):
        return 'hi'

@dataclass
class Description:
    entries: List[Content]

    def to_string(self):
        return [
            entry.to_string()
            for entry in entries
        ].join(' ')

@dataclass
class Type:
    qualifiers: List[str]
    post_qualifiers: List[str]
    type_: 'Type'
    ref_id: Optional[str]

    @classmethod
    def parse(cls, node):
        words = split_by_whitespace(node.text)

        children = list(node.getchildren())
        if len(children):
            return Type(
                qualifiers=words,
                post_qualifiers=split_by_whitespace(children[0].tail),
                type_=cls.parse(children[0]),
                ref_id=children[0].attrib.get('refid')
            )

        qualifiers = []
        post_qualifiers = []
        type_ = ''

        for word in words:
            if not type_:
                if word in QUALIFIERS:
                    qualifiers.append(word)
                else:
                    type_ = word
            else:
                post_qualifiers.append(word)

        return Type(
            qualifiers=qualifiers,
            post_qualifiers=post_qualifiers,
            type_=type_,
            ref_id=None,
        )

@dataclass
class Argument:
    type_: Type
    name: str

@dataclass
class BaseNodeInfo:
    ref_id: str
    type_: str

@dataclass
class Function(BaseNodeInfo):
    is_static: bool
    is_const: bool
    template: List[Argument]
    arguments: List[Argument]
    return_type: Type
    name: str

    @classmethod
    def parse(cls, node, base_info):
        attrs = get_node_attributes(node)

        return Function(
            is_static=get_bool_attr(attrs, 'static'),
            is_const=get_bool_attr(attrs, 'const'),
            template=cls.get_function_arguments(node.find('./templateparamlist')),
            arguments=cls.get_function_arguments(node),
            return_type=cls.get_function_return_type(node),
            name=get_node_text(node, './name'),
            **dataclasses.asdict(base_info)
        )

    def get_function_arguments(node):
        if node is None:
            return []
        return [
            Argument(
                type_=Type.parse(param.find('./type')),
                name= get_node_text(param, './declname'),
            )
            for param in node.findall('./param')
        ]

    def get_function_return_type(node):
        return Type.parse(
            node.find('./type')
        )

@dataclass
class Class(BaseNodeInfo):
    name: str

    @staticmethod
    def parse(node, base_info):
      return Class(
          name=get_node_text(node, './compoundname'),
          **dataclasses.asdict(base_info),
      )

@dataclass
class Definition:
    doc: Documentation
    code: Union[Function, Class]

    @staticmethod
    def parse(node):
        attrs = get_node_attributes(node)

        base_info = BaseNodeInfo(
            ref_id=attrs.get('id'),
            type_=attrs.get('kind'),
        )

        if base_info.type_ == 'function':
            code = Function.parse(node, base_info)
        elif base_info.type_ == 'class':
            code = Class.parse(node, base_info)
        else:
            code = base_info

        return Definition(
            doc=Documentation.parse(node),
            code=code,
        )

    @property
    def id_(self):
        return self.doc.id_

    def get_priority(self):
        # Order methods based on the following types
        if self.id_.startswith('constructor'):
            return 0

        if self.id_.startswith('operator'):
            return 1

        return 2

    @staticmethod
    def sort_key(definition):
        # Sort by id and then by it's index
        return [
            definition.get_priority(),
            definition.doc.id_,
            definition.doc.id_index
        ]

    def is_alias(self):
        return self.doc.is_alias()

@dataclass
class DocNode:
    definitions: List[Definition]

@dataclass(init=False)
class DocTreeNode:
    definitions: List[Definition]
    children: List['DocTreeNode']

    def __init__(self,
                 definitions: List[Definition],
                 children: List['DocTreeNode']):
        self.definitions = sorted(
            definitions,
            key=Definition.sort_key
        )
        self.children = children

    def build_api_path(self, *path):
        if self.api_dir:
            return '/'.join([self.api_dir, *path])

        return '/'.join(path)

    @property
    def id_(self):
        return self.definitions[0].id_

    @property
    def name(self):
        name = self.definitions[0].code.name

        if name.startswith('operator'):
            return re.sub(r'^operator', 'operator &nbsp; ', name)

        if self.id_ == 'constructor':
            return '(constructor)'

        return name

@dataclass
class DocTree:
    roots: List['DocTreeNode']
