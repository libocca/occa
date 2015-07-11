import sys
import numpy as np

pythonHeaderDir = sys.prefix                  +\
                  '/include/python'           +\
                  str(sys.version_info.major) +\
                  '.'                         +\
                  str(sys.version_info.major) +\
                  '/'

numpyHeaderDir = np.get_include() + '/'

print pythonHeaderDir
print numpyHeaderDir