from os import environ as ENV
import binascii

modes = ['Serial',
         'OpenMP',
         'OpenCL',
         'CUDA',
         'Pthreads',
         'COI',

         'Vector']

occaDir = ENV['OCCA_DIR']

def defineVar(filename):
    contents  = open(occaDir + '/include/defines/' + filename + '.hpp').read()
    contents += '\0'

    chars = len(contents)

    define = '  char ' + filename + '[' + str(len(contents)) + '] = {'
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
    filename = 'occa' + mode + 'Defines'

    info = defineVar(filename)

    decs += '  extern char ' + filename + '[' + str(info[0]) + '];\n'
    defs += info[1]

decs += '}'
defs += '}'

hpp = open(occaDir + '/include/occaKernelDefines.hpp', 'w')
hpp.write(decs)
hpp.close()

cpp = open(occaDir + '/src/occaKernelDefines.cpp', 'w')
cpp.write(defs)
cpp.close()