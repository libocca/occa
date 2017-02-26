# The MIT License (MIT)
#
# Copyright (c) 2014-2017 David Medina and Tim Warburton
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys, os, imp, re
import numpy as np
from distutils import sysconfig
import os.path as osp

occadir = osp.abspath(osp.join(osp.abspath(osp.dirname(__file__)), '..'))

def license(argv):
    with open(osp.join(occadir, 'LICENSE')) as h:
        header       = [l.strip() for l in h.readlines()]
        header_lines = len(header)

    def has_license(lines):
        if len(lines) < header_lines:
            return False

        for i in range(header_lines):
            if lines[i].strip() != header[i]:
                return False

        return True

    def cpp_find_license(lines):
        linec = len(lines)
        i = 0
        while i < linec and not lines[i].strip():
            i += 1
        if i < linec and not lines[i].strip().startswith('/*'):
            return 0
        if 1 >= linec or 2 <= lines[i].strip().find('*/'):
            return 0
        i += 1
        while i < linec and lines[i].find('*/') < 0:
            i += 1
        return i + 1

    def py_find_license(lines):
        linec = len(lines)
        i = 0
        while i < linec and not lines[i].strip():
            i += 1
        while i < linec and lines[i].startswith('#'):
            i += 1
        return i

    cpp_license = (['/* ' + header[0]] +
                   [' * ' + h for h in header[1:-1]] +
                   [' */'])
    py_license  = ['# ' + h for h in header]

    cpp = {'find'   : cpp_find_license,
           'license': cpp_license}

    py = {'find'   : py_find_license,
          'license': py_license}

    ext_ops = {'cpp'      : cpp,
               'hpp'      : cpp,
               'tpp'      : cpp,
               'py'       : py,
               'sh'       : py,
               'Makefile' : py}

    def add_license(file_):
        ext = osp.basename(file_).split('.')[-1]
        if ext not in ext_ops:
            return

        ext_op = ext_ops[ext]

        lines = []
        with open(file_) as f:
            lines = f.readlines()

        if has_license(lines):
            return

        first_line = ext_op['find'](lines)
        if first_line == len(ext_op['license']):
            return

        with open(file_, 'w') as f:
            f.writelines(l + '\n' for l in ext_op['license'])
            f.write('\n')
            f.writelines(lines[first_line:])

    def add_license_to_dir(dir_):
        for path_, dirs_, files_ in os.walk(dir_):
            for file_ in files_:
                add_license(osp.join(path_, file_))

    add_license_to_dir(osp.join(occadir, 'include'))
    add_license_to_dir(osp.join(occadir, 'src'))
    add_license_to_dir(osp.join(occadir, 'scripts'))
    add_license(osp.join(occadir, 'Makefile'))

def compile(argv):
    py_major = str(sys.version_info.major)
    py_minor = str(sys.version_info.minor)
    python_name = 'python' + py_major + '.' + py_minor
    if py_major == '3':
        python_name += 'm'

    py_header_dir = sys.prefix + '/include/' + python_name + '/'

    numpy_header_dir = np.get_include() + '/'

    libpython_dir = sysconfig.get_config_var("LIBDIR") + '/'
    libpython     = python_name

    while occadir[-1] == '/':
        occadir = occadir[:-1]

    cmd_args = ' '.join(argv[1:])

    cmd = ('make'                                    +\
           ' OCCA_COMPILE_PYTHON=1'                  +\
           ' OCCA_LIBPYTHON='     + libpython        +\
           ' OCCA_LIBPYTHON_DIR=' + libpython_dir    +\
           ' OCCA_PYTHON_DIR='    + py_header_dir    +\
           ' OCCA_NUMPY_DIR='     + numpy_header_dir +\
           ' ' + cmd_args                            +\
           ' -f ' + occadir + '/Makefile')

    os.system(cmd)

    try:
        imp.find_module('occa')
    except ImportError:
        print("""
    ---[ Note ]-----------------------------
     Remember to:
       export PYTHONPATH=$PYTHONPATH:{}/lib
    ========================================
    """.format(occadir))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(0)

    cmd  = sys.argv[1]
    argv = sys.argv[2:]

    if cmd == 'build'  : build(argv)
    if cmd == 'license': license(argv)