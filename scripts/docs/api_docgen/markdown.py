'''
Markdown generation
'''
from .types import *

def parse_sections(content: str,
                   node_link: str,
                   hyperlink_mapping: HyperlinkMapping,
                   header_index_offset: int = 1) -> Dict[str, str]:
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
    content = (content or '').strip()

    parts = re.split(f'(?:^|\n+)([^\s].*?[^\s]):\n', content)

    sections = {}
    # Content starts with ^ so we want to ignore parts[0]
    for section_header, section_content in zip(parts[1::2], parts[2::2]):
        section_content = remove_section_padding(section_content)

        section_content = replace_headers(section_header,
                                          section_content,
                                          node_link,
                                          header_index_offset)

        section_content = replace_hyperlinks(section_content,
                                             hyperlink_mapping)

        sections.update({
            section_header: section_content,
        })

    return sections


def remove_section_padding(content: str) -> str:
    return '\n'.join(
        re.sub('^' + DEFINITION_SECTION_INDENT, '', line)
        for line in content.splitlines()
    )


def replace_headers(header: str,
                    content: str,
                    node_link: str,
                    header_index_offset: int) -> str:
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

            header_html = build_header(
                header=section_header,
                header_index=(header_index + header_index_offset),
                node_link=node_link,
            )
            content += f'\n{header_html}\n{section_content}'

    return content


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
            replace_hyperlink(text, hyperlink_mapping)
            if index % 2 else
            text
        )
        for index, text in enumerate(groups)
    ])


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
