import os

OCCA_DIR = os.environ.get(
    'OCCA_DIR',
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../..')
    )
)

# Where Doxygen should output the XML
DOXYGEN_OUTPUT = f'{OCCA_DIR}/.doxygen_output'

# Each unique class/method should have a @id{id-name}
OCCA_DOC_TAG = 'occa-doc'

# Where the class/method information is stored inside the doc tree
INFO_FIELD = '__info__'

# Where aliases are store in the doc tree
ALIASES_FIELD = '__aliases__'

# C++ Qualifiers to find the type
#
# Note that `long` is not in the list since it's more likely to be used as a
# type than a qualifier in the library (crossing fingers)
QUALIFIERS = [
    'class',
    'const',
    'enum',
    'explicit',
    'extern',
    'friend',
    'inline',
    'mutable',
    'register',
    'signed',
    'static',
    'struct',
    'typedef',
    'union',
    'unsigned',
    'virtual',
    'volatile',
]
