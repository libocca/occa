'''
Basic types
'''
import dataclasses
from dataclasses import dataclass
from typing import cast, Any, Dict, List, Match, Optional, Tuple, Union

from .constants import *
from .utils import *
from .xml_utils import *
from .dev_utils import *


Content = Union['Markdown', 'Hyperlink']
Code = Union['DefinitionInfo', 'Function', 'Class']


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
    def parse(node: Any) -> 'Documentation':
        location = get_node_attributes(node, f'./location')
        filepath = location['file']
        line_number = location['line']

        doc_node = get_documentation_node(node)
        (id_, id_index) = Documentation.parse_id(doc_node)


        return Documentation(
            filepath=filepath,
            line_number=line_number,
            id_=id_,
            id_index=id_index,
            sections=Markdown.get_sections(doc_node.text),
        )

    @staticmethod
    def parse_id(doc_node: Any) -> Tuple[str, int]:
        full_id = doc_node.attrib['id']
        if '[' not in full_id:
            # If there is only 1, default to 0 index
            return (full_id, 0)

        # Searching for something like "constructor[0]" (<id>[<id_index>])
        id_pattern = r'(?P<id>[a-zA-Z0-9_]+)'
        id_index_pattern = r'\[(?P<id_index>[0-9]+)\]'

        m = cast(
            Match[Any],
            re.match(id_pattern + id_index_pattern, full_id)
        )
        return (m['id'], int(m['id_index']))


@dataclass
class Markdown:
    text: str

    @staticmethod
    def expand_hyperlinks(markdown: str) -> List[Content]:
        '''
        Separate the hyperlink and markdown content
        '''
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

    @staticmethod
    def get_sections(content: str) -> Dict[str, List[Content]]:
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
            header: Markdown.expand_hyperlinks(contents[index])
            for index, header in enumerate(headers)
        }


    def to_string(self) -> str:
        return self.text


@dataclass
class Hyperlink:
    node_id: str
    text: str

    @staticmethod
    def parse(text: str) -> 'Hyperlink':
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

    def to_string(self) -> str:
        # TODO
        return 'hi'


@dataclass
class Description:
    entries: List[Content]

    def to_string(self) -> str:
        return ' '.join([
            entry.to_string()
            for entry in self.entries
        ])


@dataclass
class Type:
    qualifiers: List[str]
    post_qualifiers: List[str]
    type_: Union['Type', str]
    ref_id: Optional[str]

    @classmethod
    def parse(cls, node) -> 'Type':
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
class DefinitionInfo:
    ref_id: str
    type_: str
    name: str


@dataclass
class Function(DefinitionInfo):
    is_static: bool
    is_const: bool
    template: List[Argument]
    arguments: List[Argument]
    return_type: Type

    @classmethod
    def parse(cls, node: Any, def_info: DefinitionInfo) -> 'Function':
        attrs = get_node_attributes(node)

        return Function(**{
            **dataclasses.asdict(def_info),
            'is_static': get_bool_attr(attrs, 'static'),
            'is_const': get_bool_attr(attrs, 'const'),
            'template': cls.get_function_arguments(node.find('./templateparamlist')),
            'arguments': cls.get_function_arguments(node),
            'return_type': cls.get_function_return_type(node),
            'name': get_node_text(node, './name'),
        })

    def get_function_arguments(node: Any) -> List[Argument]:
        if node is None:
            return []
        return [
            Argument(
                type_=Type.parse(param.find('./type')),
                name= get_node_text(param, './declname'),
            )
            for param in node.findall('./param')
        ]

    def get_function_return_type(node: Any) -> Type:
        return Type.parse(
            node.find('./type')
        )

    def get_markdown_content(self,
                             doc: Documentation,
                             overrides: List['Definition']) -> str:
        return 'hi'


@dataclass
class Class(DefinitionInfo):
    name: str

    @staticmethod
    def parse(node: Any, def_info: DefinitionInfo):
      return Class(**{
          **dataclasses.asdict(def_info),
          'name': get_node_text(node, './compoundname'),
      })

    def get_markdown_content(self, doc: Documentation) -> str:
        if doc.id_ == 'device':
            pp_json(self)
            pp_json(doc)
        return 'hi'


@dataclass
class Definition:
    doc: Documentation
    code: Code

    @staticmethod
    def parse(node) -> 'Definition':
        attrs = get_node_attributes(node)

        def_info = DefinitionInfo(
            ref_id=attrs.get('id'),
            type_=attrs.get('kind'),
            name='',
        )

        code: Code
        if def_info.type_ == 'function':
            code = Function.parse(node, def_info)
        elif def_info.type_ == 'class':
            code = Class.parse(node, def_info)
        else:
            code = def_info

        return Definition(
            doc=Documentation.parse(node),
            code=code,
        )

    @property
    def id_(self) -> str:
        return self.doc.id_

    def get_priority(self) -> int:
        # Order methods based on the following types
        if self.id_.startswith('constructor'):
            return 0

        if self.id_.startswith('operator'):
            return 1

        return 2

    @staticmethod
    def sort_key(definition: 'Definition') -> Any:
        # Sort by id and then by it's index
        return (
            definition.get_priority(),
            definition.doc.id_,
            definition.doc.id_index
        )


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
        self.children = sorted(
            children,
            key=DocTreeNode.sort_key
        )

    @property
    def root_definition(self):
        return self.definitions[0]

    @property
    def id_(self) -> str:
        return self.root_definition.id_

    @property
    def name(self) -> str:
        name = self.root_definition.code.name

        if name.startswith('operator'):
            return re.sub(r'^operator', 'operator &nbsp; ', name)

        if self.id_ == 'constructor':
            return '(constructor)'

        return name

    @staticmethod
    def sort_key(node: 'DocTreeNode') -> Any:
        return Definition.sort_key(node.root_definition)

    def get_markdown_content(self) -> str:
        def_ = self.root_definition

        if isinstance(def_.code, Class):
            return def_.code.get_markdown_content(def_.doc)

        if isinstance(def_.code, Function):
            return def_.code.get_markdown_content(def_.doc, self.definitions[1:])

        return r'¯\\_(ツ)_/¯'

@dataclass
class DocTree:
    roots: List['DocTreeNode']
