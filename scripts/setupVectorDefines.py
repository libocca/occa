from os import environ as ENV

types = ['bool',
         'char',
         'short',
         'int',
         'long',
         'float',
         'double']

Ns = [2, 4, 3, 8, 16]

cudaDefined = { 'char2'   : True,
                'char3'   : True,
                'char4'   : True,
                'short2'  : True,
                'short3'  : True,
                'short4'  : True,
                'int2'    : True,
                'int3'    : True,
                'int4'    : True,
                'long2'   : True,
                'long3'   : True,
                'long4'   : True,
                'float2'  : True,
                'float3'  : True,
                'float4'  : True,
                'double2' : True,
                'double3' : True,
                'double4' : True }

unaryOps  = ['+', '-']
binaryOps = ['+', '-', '*', '/']

def varL(n):
    if n < 4:
        return chr(ord('w') + ((n + 1) % 4))
    else:
        return varN(n)

def varN(n):
    return 's' + str(n)

def defineTypeN(type_, n):
    typeN = type_ + str(n)
    TYPEN = (type_ + str(n)).upper();

    define = ''

    if n == 3:
        if typeN in cudaDefined:
            define += '#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))\n'

        define += 'typedef ' + type_ + '4 ' + type_ + '3;\n'

        if typeN in cudaDefined:
            define += '#endif\n'

        return define

    if typeN in cudaDefined:
        define += '#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))\n'
        define += '#  define OCCA_' + TYPEN + '_CONSTRUCTOR '      + typeN + '\n'
        define += '#else\n'
        define += '#  define OCCA_' + TYPEN + '_CONSTRUCTOR make_' + typeN + '\n'
        define += '#endif\n'

        define += '#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))\n'
    else:
        define += '#  define OCCA_' + TYPEN + '_CONSTRUCTOR ' + typeN + '\n'

    define += 'class ' + type_ + str(n) + '{\n' + \
             'public:\n'

    for i in xrange(n):
        if(i < 4):
            define += '  union { ' + type_ + ' ' + varN(i) + ', ' + varL(i) + '; };\n'
        else:
            define += '  ' + type_ + ' ' + varN(i) + ';\n'

    define += '\n'

    for i in xrange(n + 1):
        start = '  occaFunction inline ' + typeN + '('

        args = (',\n' + (' ' * len(start))).join('const ' + type_ + ' &' + varL(j) + '_' for j in xrange(i))

        define += '  inline occaFunction ' + typeN + '(' + args + ') : \n' + \
                  '    ' + (',\n    '.join(varL(j) +'(' + ('0' if (i <= j) else varL(j) + '_') + ')' for j in xrange(n))) + \
                  ' {}\n'

        if i < n:
            define += '\n'

    define += '};\n'

    if typeN in cudaDefined:
        define += '#endif\n'

    define += '\n'

    for op in unaryOps:
        define += unaryOpDef(type_, n, op)

    for op in binaryOps:
        define += binaryOpDef(type_, n, op)

    define += '\n'
    define += '#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))\n'
    define += 'inline std::ostream& operator << (std::ostream &out, const ' + typeN + '& a){\n'

    l = ''; r = '';

    if type_ == 'bool':
        l = '('; r = ' ? "true" : "false")'

    define += '  out << "[" << ' + (' << ", "\n             << '.join((l + 'a.' + varL(i) + r) for i in xrange(n))) + '\n'

    define += '      << "]\\n";\n\n'
    define += '  return out;\n'
    define += '}\n'
    define += '#endif\n\n'

    return define

def unaryOpDef(type_, n, op):
    typeN = type_ + str(n)
    TYPEN = (type_ + str(n)).upper();

    isBool  =  (type_ == 'bool')
    isFloat = ((type_ == 'float') or (type_ == 'double'))

    if isBool:
        return ''

    op2   = op + op
    ops = [op, op2, op2]

    defines = ['occaFunction inline ' + typeN + ' operator '  + ops[0] + ' (const ' + typeN + ' &a){\n',
               'occaFunction inline ' + typeN + ' operator '  + ops[1] + ' (' + typeN + ' &a, int){\n',
               'occaFunction inline ' + typeN + '& operator ' + ops[2] + ' (' + typeN + ' &a){\n']

    maxDefs = (1 if isFloat else 3)

    indent = ['  ', '  ', '  ']

    for d in xrange(1 if isFloat else 2):
        ret         = '  return OCCA_' + TYPEN + '_CONSTRUCTOR(';
        indent[d]   = ' ' * len(ret)
        defines[d] += ret;

    for d in xrange(maxDefs):
        dm = (',' if (d < 2) else ';')

        if d == 2:
            defines[d] += indent[d]

        for i in xrange(n):
            if d != 1:
                defines[d] += ops[d]

            defines[d] += 'a.' + varL(i)

            if d == 1:
                defines[d] += ops[d]

            if i < (n - 1):
                defines[d] += dm + '\n' + indent[d]

        if d != 2:
            defines[d] += ');\n'
        else:
            defines[d] += ';\n'

    for d in xrange(maxDefs):
        if 0 < d:
            defines[0] += '\n' + defines[d]

        if d == 2:
            defines[0] += indent[2] + 'return a;\n'

        defines[0] += '}\n'

    return defines[0]

def binaryOpDef(type_, n, op):
    typeN = type_ + str(n)
    TYPEN = (type_ + str(n)).upper();

    a = 'a.'
    b = 'b.'

    defines = ['occaFunction inline ' + typeN + ' operator ' + op + ' (const ' + typeN + ' &a, const ' + typeN + ' &b){\n',
               'occaFunction inline ' + typeN + ' operator ' + op + ' (const ' + type_ + ' &a, const ' + typeN + ' &b){\n',
               'occaFunction inline ' + typeN + ' operator ' + op + ' (const ' + typeN + ' &a, const ' + type_ + ' &b){\n',
               'occaFunction inline ' + typeN + '& operator ' + op + '= (' + typeN + ' &a, const ' + typeN + ' &b){\n',
               'occaFunction inline ' + typeN + '& operator ' + op + '= (' + typeN + ' &a, const ' + type_ + ' &b){\n']

    aIsTypeN = [True, False, True, True, True]
    bIsTypeN = [True, True, False, True, False]

    a = [[('a.' + varL(i)) if aIsTypeN[define] else 'a' for i in xrange(n)] for define in xrange(5)]
    b = [[('b.' + varL(i)) if bIsTypeN[define] else 'b' for i in xrange(n)] for define in xrange(5)]

    retDef = '  return OCCA_' + TYPEN + '_CONSTRUCTOR('

    for d in xrange(5):
        for i in xrange(n):
            if d < 3:
                if i == 0:
                    defines[d] += retDef

                defines[d] += a[d][i] + ' ' + op + ' ' + b[d][i]

                if i < (n - 1):
                    defines[d] += ',\n' + (' ' * len(retDef))
            else:
                defines[d] += '  a.' + varL(i) + ' ' + op + '= ' + b[d][i] + ';\n'

    for d in xrange(5):
        if 0 < d:
            defines[0] += '\n' + defines[d]

        if d < 3:
            defines[0] += ');\n'
        else:
            defines[0] += '  return a;\n'

        defines[0] += '}\n'

    return defines[0]

def defineType(type_):
    define = ''

    for n in Ns:
        typeN = type_ + str(n)

        comment = '//---[ ' + typeN + ' ]'
        define += comment + ('-' * (40 - len(comment))) + '\n'

        define += defineTypeN(type_, n)

        define += '//' + ('=' * 38) + '\n\n\n'

    return define

def defineAllTypes():
    define  = '#if (!defined(OCCA_IN_KERNEL) || (!OCCA_USING_OPENCL))\n'
    define += '#  if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))\n'
    define += '#    include <iostream>\n'
    define += '#  endif\n\n'

    define += '#  ifndef OCCA_IN_KERNEL\n'
    define += '#    define occaFunction\n'
    define += 'namespace occa {\n'
    define += '#  endif\n\n'

    for type_ in types:
        define += defineType(type_)

    define += '#  ifndef OCCA_IN_KERNEL\n'
    define += '}\n'
    define += '#  endif\n\n'

    define += '#endif\n'

    return define

occaDir = ENV['OCCA_DIR']

hpp = open(occaDir + '/include/occa/defines/vector.hpp', 'w')
hpp.write(defineAllTypes())
hpp.close()
