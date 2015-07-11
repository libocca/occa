import sys, os
import numpy as np

pythonHeaderDir = sys.prefix                  +\
                  '/include/python'           +\
                  str(sys.version_info.major) +\
                  '.'                         +\
                  str(sys.version_info.minor) +\
                  '/'

numpyHeaderDir = np.get_include() + '/'

if not os.environ.has_key('OCCA_DIR'):
    print "Error: You need to set the environment variable [OCCA_DIR]"
    print "For example:"
    print "  export OCCA_DIR='$(shell pwd)'"
    sys.exit()

OCCA_DIR = os.environ['OCCA_DIR']

commandLineArgs = ' '.join(sys.argv[1:])

os.system('make'                                +\
          ' OCCA_PYTHON_DIR=' + pythonHeaderDir +\
          ' OCCA_NUMPY_DIR='  + numpyHeaderDir  +\
          ' ' + commandLineArgs                 +\
          ' -f ' + OCCA_DIR + '/makefile')