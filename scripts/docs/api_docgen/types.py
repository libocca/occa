'''
Basic types
'''
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union

from .constants import *
from .utils import *
from .xml_utils import *


@dataclass
class NodeInfo:
    # Manual
    id_: str
    is_alias: bool
    # Automatic
    filepath: str
    line_number: str

    @staticmethod
    def parse(node):
        id_node = node.find(f'./*/*/{ID_TAG}')
        id_node_attrs = get_node_attributes(id_node)

        location = get_node_attributes(node, f'./location')
        return NodeInfo(
            id_ = id_node.text.strip(),
            is_alias = 'true' == id_node_attrs.get('is_alias', 'false'),
            filepath = location['file'],
            line_number = location['line'],
        )

    def get_id(self):
        return self.id_ or self.alias_id

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
                node_id = text,
                text = text,
            )

        return Hyperlink(
            node_id = parts[1].strip(),
            text = parts[0].strip(),
        )

    def to_string(self):
        return 'hi'

@dataclass
class Description:
    entries: List[Union[Markdown, Hyperlink]]

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
                qualifiers = words,
                post_qualifiers = split_by_whitespace(children[0].tail),
                type_ = cls.parse(children[0]),
                ref_id = children[0].attrib.get('refid')
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
            qualifiers = qualifiers,
            post_qualifiers = post_qualifiers,
            type_ = type_,
            ref_id = None,
        )

@dataclass
class Argument:
    type_: Type
    name: str

@dataclass
class BaseNodeInfo:
    ref_id: str
    type_: str
    description: Description
    instance_description: Description

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
            is_static = get_bool_attr(attrs, 'static'),
            is_const = get_bool_attr(attrs, 'const'),
            template = cls.get_function_arguments(node.find('./templateparamlist')),
            arguments = cls.get_function_arguments(node),
            return_type = cls.get_function_return_type(node),
            name = get_node_text(node, './name'),
            **dataclasses.asdict(base_info)
        )

    def get_function_arguments(node):
        if node is None:
            return []
        return [
            Argument(
                type_ = Type.parse(param.find('./type')),
                name =  get_node_text(param, './declname'),
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
          name = get_node_text(node, './compoundname'),
          **dataclasses.asdict(base_info),
      )

@dataclass
class Definition:
    node_info: NodeInfo
    definition: Union[Function, Class]

    @staticmethod
    def parse(node):
        attrs = get_node_attributes(node)

        base_info = BaseNodeInfo(
            ref_id = attrs.get('id'),
            type_ = attrs.get('kind'),
            description = get_node_description(node),
            instance_description = get_node_instance_description(node),
        )

        if base_info.type_ == 'function':
            definition = Function.parse(node, base_info)
        elif base_info.type_ == 'class':
            definition = Class.parse(node, base_info)
        else:
            definition = base_info

        return Definition(
            node_info = NodeInfo.parse(node),
            definition = definition,
        )

    def is_alias(self):
        return self.node_info.is_alias