from os import environ as ENV
import binascii

modes = ['Serial',
         'OpenMP',
         'OpenCL',
         'CUDA',
         'Pthreads',
         'COI',

         'vector']

occaDir = ENV['OCCA_DIR']

def defineVar(varName, filename):
    contents  = open(filename).read()
    contents += '\0'

    chars = len(contents)

    define = '  char ' + varName + '[' + str(len(contents)) + '] = {'
    indent = ' ' * len(define)

    for i in xrange(chars):
        define += '0x' + binascii.hexlify(contents[i])

        if i < (chars - 1):
            define += ', '

        if ((i % 8) == 7):
            define += '\n' + indent

    define += '};\n'

    return [chars, define]

decs = 'namespace occa {\n'
defs = 'namespace occa {\n'

for mode in modes:
    varName  = 'occa' + mode[0].capitalize() + mode[1:] + 'Defines'
    filename = occaDir + '/include/occa/defines/' + mode + '.hpp'

    info = defineVar(varName, filename)

    decs += '  extern char ' + varName + '[' + str(info[0]) + '];\n'
    defs += info[1]

varName  = 'occaShellTools'
filename = occaDir + '/scripts/shellTools.sh'

info = defineVar(varName, filename)

decs += '  extern char ' + varName + '[' + str(info[0]) + '];\n'
defs += info[1]

decs += '}'
defs += '}'

hpp = open(occaDir + '/include/occa/varFiles.hpp', 'w')
hpp.write(decs)
hpp.write('\n')  # Make sure there is a newline at the end of the file
hpp.close()

cpp = open(occaDir + '/src/varFiles.cpp', 'w')
cpp.write(defs)
cpp.write('\n')
cpp.close()