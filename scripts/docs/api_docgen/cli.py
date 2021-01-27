'''
Wrapper to the CLI interface
'''
import argparse


from .api_docgen import generate_api
from .file_parser import load_documentation
from .system_commands import run_doxygen


def main():
    args = parse_args()

    run_doxygen()
    doc_tree = load_documentation()
    generate_api(
        args.output,
        doc_tree
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate OCCA API documentation'
    )
    parser.add_argument(
        '-o', '--output',
        help='Directory to the libocca.org git repo'
    )

    return parser.parse_args()
