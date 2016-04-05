import sys, os, imp
import numpy as np
from distutils import sysconfig
import os.path as osp

occadir = osp.abspath(osp.dirname(__file__))

py_major = str(sys.version_info.major)
py_minor = str(sys.version_info.minor)
pythonName = 'python' + py_major + '.' + py_minor
if py_major == '3':
    pythonName += 'm'

py_header_dir = sys.prefix + '/include/' + pythonName + '/'

numpy_header_dir = np.get_include() + '/'

libpython_dir = sysconfig.get_config_var("LIBDIR") + '/'
libpython     = pythonName

while occadir[-1] == '/':
    occadir = occadir[:-1]

commandLineArgs = ' '.join(sys.argv[1:])

cmd = ('make'                                    +\
       ' OCCA_LIBPYTHON='     + libpython        +\
       ' OCCA_LIBPYTHON_DIR=' + libpython_dir    +\
       ' OCCA_PYTHON_DIR='    + py_header_dir    +\
       ' OCCA_NUMPY_DIR='     + numpy_header_dir +\
       ' ' + commandLineArgs                     +\
       ' -f ' + occadir + '/makefile')

print(cmd)
os.system(cmd)

try:
    imp.find_module('occa')
except ImportError:
    print("Remember to:")
    print("  export PYTHONPATH=$PYTHONPATH:{}/lib".format(occadir))
