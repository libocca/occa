'''
Generates Markdown from extracted metadata
'''
import os

from .constants import *
from .types import *
from .system_commands import *


def create_directory_for(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def clear_api(api_dir: str):
    # We don't want to remove the non-auto-generated file inside api/README.md
    api_readme = f'{api_dir}/{README_FILENAME}'

    with open(api_readme, 'r') as fd:
        api_readme_contents = fd.read()

    safe_rmrf(api_dir)

    create_directory_for(api_readme)
    with open(api_readme, 'w') as fd:
        fd.write(api_readme_contents)


def make_sidebar(root_dir: str,
                 relative_filepath: str,
                 children: List[DocTreeNode],
                 before_content: str,
                 after_content: str,
                 indent: str):
    indent += '  '
    # Nested directories need the `/` at the end
    children_headers = [
        f'{indent}- [{child.name}](/{relative_filepath}/{child.link_name})\n'
        for child in children
    ]

    sidebar_filepath = f'{root_dir}/{relative_filepath}/{SIDEBAR_FILENAME}'

    create_directory_for(sidebar_filepath)
    with open(sidebar_filepath, 'w') as fd:
        fd.write(before_content)

        for header in children_headers:
            fd.write(header)

        fd.write(after_content)

    for index, child in enumerate(children):
        child_before_content = before_content
        child_after_content = after_content

        for header in children_headers[:index + 1]:
            child_before_content += header

        for header in children_headers[index + 1:]:
            child_after_content += header

        make_sidebar(
            root_dir,
            relative_filepath=f'{relative_filepath}/{child.id_}',
            children=child.children,
            before_content=child_before_content,
            after_content=child_after_content,
            indent=indent,
        )


def generate_sidebars(root_dir: str, tree: DocTree):
    before_content = f'''
<div class="api-version-container">
  <select onchange="vm.onLanguageChange(this)">
    <option value="cpp">C++</option>
  </select>
  <select onchange="vm.onVersionChange(this)">
    <option value="nightly">Nightly</option>
  </select>
</div>

- [**{API_SIDEBAR_NAME}**](/{API_RELATIVE_DIR}/)
'''.strip() + '\n'

    make_sidebar(
        root_dir,
        relative_filepath=API_RELATIVE_DIR,
        children=tree.roots,
        before_content=before_content,
        after_content='',
        indent='',
    )


def generate_node_markdown(node: DocTreeNode,
                           filepath: str,
                           git_hash: str,
                           hyperlink_mapping: HyperlinkMapping):
    node_filepath = f'{filepath}/{node.id_}'
    markdown_filepath = (
        f'{node_filepath}/{README_FILENAME}'
        if node.children else
        f'{node_filepath}.md'
    )

    create_directory_for(markdown_filepath)
    with open(markdown_filepath, 'w') as fd:
        fd.write(node.get_markdown_content(git_hash, hyperlink_mapping))

    for child in node.children:
        generate_node_markdown(child, node_filepath, git_hash, hyperlink_mapping)


def generate_markdown(root_dir: str, tree: DocTree):
    git_hash = get_git_hash()
    hyperlink_mapping = tree.build_hyperlink_mapping()

    for child in tree.roots:
        generate_node_markdown(child, root_dir, git_hash, hyperlink_mapping)


def generate_api(root_dir: str, tree: DocTree):
    api_dir = f'{root_dir}/{API_RELATIVE_DIR}'

    clear_api(api_dir)
    generate_sidebars(root_dir, tree)
    generate_markdown(api_dir, tree)
