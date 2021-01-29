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

    def optional_get(self, node_id: str) -> Optional[HyperlinkNodeInfo]:
        return self.mapping.get(node_id)

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
        filepath = location['file'].replace(f'{OCCA_DIR}/', '')
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

    def to_string(self,
                  hyperlink_mapping: HyperlinkMapping,
                  var_name: Optional[str] = None) -> Tuple[str, int]:
        # Keep track of the character count for padding purposes
        content = ''
        char_count = 0
        if self.qualifiers:
            content += f'''
<span class="token keyword">{' '.join(self.qualifiers)}</span>
'''.strip()
            # Spaces
            char_count += len(self.qualifiers) - 1
            # Words
            char_count += sum(
                len(qualifier)
                for qualifier in self.qualifiers
            )

        info = None
        if self.ref_id:
            info = hyperlink_mapping.optional_get(self.ref_id)

        if self.qualifiers:
            content += ' '
            char_count += 1

        if info:
            content += f'''<a href="#{info.link}">{info.name}</a>'''.strip()
            char_count += len(info.name)
        elif self.type_:
            content += f'<span class="token keyword">{self.type_}</span>'
            char_count += len(cast(str, self.type_))

        needs_space_before_name = True
        if self.post_qualifiers:
            # Format:
            # - void*
            # - void *ptr
            if var_name:
                needs_space_before_name = False
                content += ' '
                char_count += 1
            content += ''.join(self.post_qualifiers)
            char_count += sum(
                len(qualifier)
                for qualifier in self.post_qualifiers
            )

        if var_name:
            if needs_space_before_name:
                content += ' '
                char_count += 1

            content += var_name
            char_count += len(var_name)

        return (content, char_count)

@dataclass
class Argument:
    type_: Type
    name: str

    def to_string(self, hyperlink_mapping: HyperlinkMapping):
        return self.type_.to_string(hyperlink_mapping, self.name)


@dataclass
class DefinitionInfo:
    ref_id: str
    type_: str
    name: str

    @property
    def short_name(self):
        # Remove the templates
        return self.name.split('<')[0]


@dataclass
class Function(DefinitionInfo):
    full_name: str
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
            'full_name': ''.join([
                get_node_text(node, './definition'),
                get_node_text(node, './argsstring'),
            ]),
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
                             name: str,
                             overrides: List['Definition'],
                             git_hash: str,
                             hyperlink_mapping: HyperlinkMapping) -> str:
        from . import markdown

        info = hyperlink_mapping.get(self.ref_id)
        sections = markdown.parse_sections(doc.description,
                                           info.link,
                                           hyperlink_mapping,
                                           header_index_offset=2)

        description = sections.get('Description')
        examples = sections.get('Examples')

        method_header = markdown.build_header(name, 1, info.link, include_id=False)

        function_definitions = '\n  <hr>\n'.join(
            f'''
  <div class="definition-container">
    <div class="definition">
      <code>{func.get_function_signature(hyperlink_mapping)}</code>
      <div class="flex-spacing"></div>
      <a href="{func.get_source_link(override.doc, git_hash)}" target="_blank">Source</a>
    </div>
    {self.get_description_markdown(override, hyperlink_mapping)}
  </div>
'''
            for override in overrides
            for func in [cast('Function', override.code)]
        )

        content = f'''
{method_header}

<div class="signature">
  <hr>

  {function_definitions}

  <hr>
</div>
'''

        for (section_header, section_content) in [('Description', description), ('Examples', examples)]:
            if section_content:
                section_header = markdown.build_header(section_header, 2, info.link)
                content += f'''

{section_header}

{section_content}
'''

        return content

    def get_function_signature(self, hyperlink_mapping: HyperlinkMapping):
        # Example:
        #
        #   template <class TM>
        #   occa::memory malloc(const int *arg1,
        #                       const int *arg2,
        #                       const int *arg3)
        content = ''
        char_count = 0

        # template <class TM>
        if self.template:
            content += '<span class="token keyword">template</span> <'

            for (index, arg) in enumerate(self.template):
                if index:
                    content += ', '

                (arg_content, arg_char_count) = arg.type_.to_string(hyperlink_mapping)
                content += arg_content

            content += '>\n'

        # occa::memory
        (return_type, return_char_count) = self.return_type.to_string(hyperlink_mapping)
        if return_type:
            content += f'{return_type} '
            char_count += return_char_count + 1

        # malloc(
        name_str = f'{self.name}'
        if name_str.startswith('operator'):
            name_str = re.sub(r'^operator', 'operator ', name_str)
            name_str += ' '
        name_str += '('

        content += name_str
        char_count += len(name_str)

        left_padding = ' ' * char_count

        # . . . . . . . . const int *arg1,
        #                 const int *arg2
        #                 const int *arg3
        for index, arg in enumerate(self.arguments):
            if index:
                content += f',\n{left_padding}'
            (arg_content, arg_char_count) = arg.to_string(hyperlink_mapping)
            content += arg_content
            char_count += arg_char_count

        # )
        content += ')'

        return content

    def get_source_link(self,
                        doc: Documentation,
                        git_hash: str) -> str:
        return f'{OCCA_GITHUB_URL}/blob/{git_hash}/{doc.filepath}#L{doc.line_number}'

    def get_description_markdown(self,
                                 def_info: 'Definition',
                                 hyperlink_mapping: HyperlinkMapping):
        from . import markdown

        info = hyperlink_mapping.get(self.ref_id)
        sections = markdown.parse_sections(def_info.doc.description,
                                           info.link,
                                           hyperlink_mapping)

        description = sections.get('Overloaded Description')
        arguments = sections.get('Arguments')
        returns = sections.get('Returns')

        argument_descriptions = []
        if arguments:
            argument_sections = markdown.parse_sections(arguments,
                                                        info.link,
                                                        hyperlink_mapping)

            func = cast('Function', def_info.code)
            argument_descriptions = [
                (arg.name, argument_sections.get(arg.name))
                for arg in func.arguments
                if argument_sections.get(arg.name)
            ]

        if (not description and
            not argument_descriptions and
            not returns):
            return ''

        content = '<div class="description">\n'

        if description:
            content += f'''
      <div>
        ::: markdown
        {description}
        :::
      </div>
'''

        if argument_descriptions:
            argument_content = '\n'.join(
                f'''
        <li>
          <strong>{arg_name}</strong>: ::: markdown
            {arg_description}
            :::
        </li>
'''
                for (arg_name, arg_description) in argument_descriptions
            )
            content += f'''
      <div class="section-header">Arguments</div>
      <ul class="section-list">
          {argument_content}
      </ul>
'''

        if returns:
            content += f'''
      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li>
          ::: markdown
          {returns}
          :::
        </li>
      </ul>
    </div>
'''

        return content


@dataclass
class Class(DefinitionInfo):
    @staticmethod
    def parse(node: Any, def_info: DefinitionInfo):
      return Class(**{
          **dataclasses.asdict(def_info),
          'name': get_node_text(node, './compoundname'),
      })

    def get_markdown_content(self,
                             doc: Documentation,
                             hyperlink_mapping: HyperlinkMapping) -> str:
        from . import markdown

        info = hyperlink_mapping.get(self.ref_id)

        sections = markdown.parse_sections(doc.description,
                                           info.link,
                                           hyperlink_mapping)

        description = sections.get('Description')
        if not description:
            raise NotImplementedError('Classes need to have a [Description] section')

        # Add the class header and missing description header
        class_header = markdown.build_header(self.short_name, 1, info.link, include_id=False)
        description_header = markdown.build_header('Description', 2, info.link)
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
        name = self.root_definition.code.short_name

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

    def get_markdown_content(self,
                             git_hash: str,
                             hyperlink_mapping: HyperlinkMapping) -> str:
        def_ = self.root_definition

        if isinstance(def_.code, Class):
            return def_.code.get_markdown_content(def_.doc,
                                                  hyperlink_mapping)

        if isinstance(def_.code, Function):
            return def_.code.get_markdown_content(def_.doc,
                                                  self.name,
                                                  self.definitions,
                                                  git_hash,
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
