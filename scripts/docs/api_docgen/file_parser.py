'''
Parse doxygen output files
'''
from typing import Any, Dict, List, Tuple

from .types import *
from .system_commands import *


def alias_path(path: str,
               node_id: str,
               ordered_path_aliases: List[Tuple[str, str]]) -> Tuple[str, str]:
    for (alias_path, alias_id) in ordered_path_aliases:
        if path.startswith(alias_path):
            return (
                path.replace(path, alias_id),
                f'{alias_id}.{node_id}',
            )

    return (path, node_id)


def get_global_id_map(tree: Any, nodes: List[Any]) -> Dict[str, str]:
    paths = sorted([
        [tree.getpath(node), Documentation.parse(node)]
        for node in nodes
    ])

    global_id_map = {}
    ordered_path_ids: List[Tuple[str, str]] = []
    for [path, node_info] in paths:
        # Build aliases from short names -> long names
        (aliased_path, aliased_id) = alias_path(
            path,
            node_info.id_,
            ordered_path_ids
        )

        # Build the ids backwards to match longer -> shorter to find
        # the longest alias possible
        ordered_path_ids.insert(0, (aliased_path, aliased_id))

        global_id_map[path] = aliased_id

    return global_id_map

def get_doc_tree(filename: str) -> Dict[str, Any]:
    (tree, root) = parse_xml_file(filename)

    nodes = get_documented_definition_nodes(root)
    global_id_map = get_global_id_map(tree, nodes)

    doc_tree: Dict[str, Any] = {}
    for node in nodes:
        # Split by . and append a CHILDREN_FIELD in between
        # Example:
        #   device.malloc
        #   ->
        #  ['device', CHILDREN_FIELD, 'malloc]
        path = []
        for entry in global_id_map[tree.getpath(node)].split('.'):
            path.append(entry)
            path.append(CHILDREN_FIELD)
        path.pop()

        doc_node = nested_get(doc_tree, path)

        if DOC_FIELD not in doc_node:
            doc_node[DOC_FIELD] = DocNode(
                definitions=[],
            )

        doc_node[DOC_FIELD].definitions.append(
            Definition.parse(node)
        )

    return doc_tree


def finalize_doc_tree(doc_tree: Dict[str, Any]) -> List[DocTreeNode]:
    return [
        DocTreeNode(
            definitions=doc_info[DOC_FIELD].definitions,
            children=finalize_doc_tree(
                doc_info.get(CHILDREN_FIELD, {})
            )
        )
        for doc_info in doc_tree.values()
    ]


def load_documentation() -> DocTree:
    # Merge all documented files into one tree
    doc_tree = {}
    for documented_file in find_documented_files():
        doc_tree.update(
            get_doc_tree(documented_file)
        )

    return DocTree(
        roots=finalize_doc_tree(doc_tree)
    )
