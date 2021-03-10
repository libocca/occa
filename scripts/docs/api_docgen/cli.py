'''
Wrapper to the CLI interface
'''
from .api_docgen import generate_api
from .file_parser import load_documentation
from .system_commands import run_doxygen
from .constants import OCCA_DIR

def main():
    run_doxygen()
    doc_tree = load_documentation()
    generate_api(
        f'{OCCA_DIR}/docs',
        doc_tree
    )
