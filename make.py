import sys, os
import numpy as np
from distutils import sysconfig

if not os.environ.has_key('OCCA_DIR'):
    print "Error: You need to set the environment variable [OCCA_DIR]"
    print "For example:"
    print "  export OCCA_DIR='$(shell pwd)'"
    sys.exit()

pythonMajorVersion = str(sys.version_info.major)
pythonMinorVersion = str(sys.version_info.minor)
pythonName         = 'python' + pythonMajorVersion + '.' + pythonMinorVersion

pythonHeaderDir = sys.prefix + '/include/' + pythonName + '/'

numpyHeaderDir = np.get_include() + '/'

libpythonDir  = sysconfig.get_config_var("LIBDIR") + '/'
libpythonFlag = '-l' + pythonName

OCCA_DIR = os.environ['OCCA_DIR']

commandLineArgs = ' '.join(sys.argv[1:])

os.system('make'                                    +\
          ' OCCA_LIBPYTHON_FLAG=' + libpythonFlag   +\
          ' OCCA_LIBPYTHON_DIR='  + libpythonDir    +\
          ' OCCA_PYTHON_DIR='     + pythonHeaderDir +\
          ' OCCA_NUMPY_DIR='      + numpyHeaderDir  +\
          ' ' + commandLineArgs                     +\
          ' -f ' + OCCA_DIR + '/makefile')