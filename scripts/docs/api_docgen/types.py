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


Code = Union['DefinitionInfo', 'Function', 'Class']
HyperlinkMapping = Dict[str, 'HyperlinkNodeInfo']

@dataclass
class HyperlinkNodeInfo:
    name: str
    link: str

@dataclass
class HyperlinkMapping:
    mapping: Dict[str, HyperlinkNodeInfo]

    def get(self, node_id: str) -> HyperlinkNodeInfo:
        try:
            return self.mapping[node_id]
        except KeyError as e:
            raise KeyError(f"Missing documentation for: [{node_id}]") from e

@dataclass
class Documentation:
    # Automatic from Doxygen
    filepath: str
    line_number: str

    # Manually provided
    id_: str
    id_index: Optional[int]
    description: str

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
            description=doc_node.text,
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
    @staticmethod
    def parse_sections(content: str,
                       node_link: str,
                       hyperlink_mapping: HyperlinkMapping) -> Dict[str, str]:
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

        sections = {}
        # Content starts with ^ so we want to ignore parts[0]
        for section_header, section_content in zip(parts[1::2], parts[2::2]):
            section_content = Markdown.remove_section_padding(section_content)

            section_content = Markdown.replace_headers(section_header,
                                                       section_content,
                                                       node_link)

            section_content = Markdown.replace_hyperlinks(section_content,
                                                          hyperlink_mapping)

            sections.update({
                section_header: section_content,
            })

        return sections

    @staticmethod
    def remove_section_padding(content: str) -> str:
        return '\n'.join(
            re.sub('^' + DEFINITION_SECTION_INDENT, '', line)
            for line in content.splitlines()
        )

    @staticmethod
    def replace_headers(header: str, content: str, node_link: str) -> str:
        # Downgrade headers and use HTML to avoid creating sidebar items for them
        for header_index in range(4, 0, -1):
            markdown_header_prefix = '#' * header_index
            parts = re.split(f'\n{markdown_header_prefix} ([^\n]*)\n', content)

            # Nothing to do
            if len(parts) == 1:
                continue

            # We could start a section without a header
            if len(parts) % 2:
                section_headers = ['', *parts[1::2]]
                section_contents = parts[0::2]
            else:
                section_headers = parts[0::2]
                section_contents = parts[1::2]

            content = ''
            for section_header, section_content in zip(section_headers, section_contents):
                if not section_header:
                    content += section_content
                    continue

                header_html = Markdown.build_header(
                    header=section_header,
                    header_index=(header_index + 1),
                    node_link=node_link,
                )
                content += f'\n{header_html}\n{section_content}'

        return content

    @staticmethod
    def build_header(header: str,
                     header_index: int,
                     node_link: str,
                     include_id: bool = True) -> str:
        # Camel/Pascal case -> Kebab case
        header_id = re.sub(r'([^A-Z])([A-Z])', r'\1-\2', header).lower()

        href = f'#{node_link}'
        if include_id:
            href += f'?id={header_id}'

        return f'''
<h{header_index} id="{header_id}">
 <a href="{href}" class="anchor">
   <span>{header}</span>
  </a>
</h{header_index}>
'''.strip()

    @staticmethod
    def replace_hyperlinks(content: str,
                           hyperlink_mapping: HyperlinkMapping) -> str:
        '''
        Separate the hyperlink and markdown content
        '''
        # Searching for content inside [[...]]
        left = '[['.replace('[', r'\[')
        right = ']]'.replace(']', r'\]')

        groups = re.split(f'{left}(.*?){right}', content)

        return ''.join([
            (
                Markdown.replace_hyperlink(text, hyperlink_mapping)
                if index % 2 else
                text
            )
            for index, text in enumerate(groups)
        ])

    @staticmethod
    def replace_hyperlink(content: str,
                          hyperlink_mapping: HyperlinkMapping) -> str:
        # Format:
        #   [[device.malloc]]
        #   [[malloc|device.malloc]]
        parts = content.split('|')

        if len(parts) == 1:
            # [[device.malloc]]
            node_id = parts[0].strip()
            link_text = None
        else:
            # [[malloc|device.malloc]]
            link_text = parts[0].strip()
            node_id = parts[1].strip()

        info = hyperlink_mapping.get(node_id)

        return f'[{link_text or info.name}]({info.link})'


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
                             overrides: List['Definition'],
                             hyperlink_mapping: HyperlinkMapping) -> str:
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

    def get_markdown_content(self,
                             doc: Documentation,
                             hyperlink_mapping: HyperlinkMapping) -> str:
        if doc.id_ == 'device':
            pp_json(self)
            pp_json(doc)

        info = hyperlink_mapping.get(self.ref_id)

        sections = Markdown.parse_sections(doc.description,
                                           info.link,
                                           hyperlink_mapping)

        description = sections.get('Description')
        if not description:
            raise NotImplementedError('Classes need to have a [Description] section')

        # Add the class header and missing description header
        class_header = Markdown.build_header(self.name, 1, info.link, include_id=False)
        description_header = Markdown.build_header('Description', 2, info.link)
        return f'''
{class_header}

{description_header}

{description}
'''.strip()


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

        return 1

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
    def ref_id(self) -> str:
        return self.root_definition.code.ref_id

    @property
    def name(self) -> str:
        name = self.root_definition.code.name

        if name.startswith('operator'):
            return re.sub(r'^operator', 'operator ', name)

        if self.id_ == 'constructor':
            return '(constructor)'

        return name

    @staticmethod
    def sort_key(node: 'DocTreeNode') -> Any:
        return Definition.sort_key(node.root_definition)

    @property
    def link_name(self) -> str:
        if self.children:
            return f'{self.id_}/'
        else:
            return self.id_;

    def get_markdown_content(self, hyperlink_mapping: HyperlinkMapping) -> str:
        def_ = self.root_definition

        if isinstance(def_.code, Class):
            return def_.code.get_markdown_content(def_.doc,
                                                  hyperlink_mapping)

        if isinstance(def_.code, Function):
            return def_.code.get_markdown_content(def_.doc,
                                                  self.definitions[1:],
                                                  hyperlink_mapping)

        raise NotImplementedError("Code type undefined")

@dataclass
class DocTree:
    roots: List[DocTreeNode]

    @staticmethod
    def get_hyperlink_mapping(base_link: str,
                              namespace: str,
                              node: DocTreeNode) -> Dict[str, HyperlinkNodeInfo]:
        global_id = (
            f'{namespace}.{node.id_}'
            if namespace else
            node.id_
        )

        node_link = f'{base_link}{node.link_name}'

        info = HyperlinkNodeInfo(
            name=node.name,
            link=node_link,
        )

        # Also store the ref_id so we can map from a node to the hyperlink info
        hyperlink_mapping = {
            global_id: info,
            node.ref_id: info,
        }

        for child in node.children:
            hyperlink_mapping.update(
                DocTree.get_hyperlink_mapping(node_link, global_id, child)
            )

        return hyperlink_mapping

    def build_hyperlink_mapping(self) -> HyperlinkMapping:
        root_link = f'/{API_RELATIVE_DIR}/'
        namespace = ''

        hyperlink_mapping = {}
        for child in self.roots:
            hyperlink_mapping.update(
                DocTree.get_hyperlink_mapping(root_link, namespace, child)
            )

        return HyperlinkMapping(mapping=hyperlink_mapping)
