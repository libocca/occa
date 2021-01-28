'''
System commands
'''
import subprocess

# Local modules
from .constants import *


def is_safe_to_rmrf(path, verbose=True):
    '''
    Check whether it's safe to 'rm -rf' the given path
    '''
    path = os.path.abspath(
        os.path.expanduser(path)
    )

    path_directories = []
    for directory in path.split('/'):
        if directory in ['', '.']:
            continue

        if directory == '..':
            path_directories.pop();
        else:
            path_directories.append(directory)

    # Make sure there are at least 3 directories between the root directory
    if len(path_directories) < 4:
        if verbose:
            print(f'WARNING: Skipping \'rm -rf {path}\' since it seems unsafe')
        return False

    return True


# Tests just to make sure it's safe to run
# We don't want to accidentally remove wrong directories!!!
assert not is_safe_to_rmrf('/', verbose=False)
assert not is_safe_to_rmrf('/a', verbose=False)
assert not is_safe_to_rmrf('/a/b', verbose=False)
assert not is_safe_to_rmrf('/a/b/c', verbose=False)
assert is_safe_to_rmrf('/a/b/c/d', verbose=False)

assert not is_safe_to_rmrf('/a/b/c/d/..', verbose=False)
assert not is_safe_to_rmrf('/a/b/c/d/../e/..', verbose=False)
assert not is_safe_to_rmrf('/a/./b/./.', verbose=False)
assert is_safe_to_rmrf('/a/./b/././c/d', verbose=False)


def safe_rmrf(path: str):
    '''
    Safely 'rm -rf' path or exit out
    '''
    if is_safe_to_rmrf(path):
        subprocess.check_output(['rm', '-rf', path])


def run_doxygen():
    '''
    Run Doxygen to generate documentation metadata
    '''
    # Remove cached output
    safe_rmrf(DOXYGEN_OUTPUT)

    # Run doxygen
    subprocess.check_output(
        ['doxygen', f'{OCCA_DIR}/.doxygen'],
        env=dict(os.environ,
                 OCCA_DIR=OCCA_DIR,
                 DOXYGEN_OUTPUT=DOXYGEN_OUTPUT),
    )


def find_documented_files():
    '''
    Grep for our custom id tags to avoid reading
    all Doxygen-output files
    '''
    output = subprocess.check_output(
        ['grep', '-Rl', OCCA_DOC_TAG, DOXYGEN_OUTPUT]
    ).decode('utf-8')

    return [
        path.strip()
        for path in output.split('\n')
        if path.strip()
    ]


def get_git_hash():
    '''
    Find the current git hash to link the documentation
    with the proper Github commit hash
    '''
    return subprocess.check_output(
        'git rev-parse --short HEAD'.split(' ')
    ).decode('utf-8').strip()
