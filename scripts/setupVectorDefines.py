#!/usr/bin/env python3
from os import environ as ENV

types = ['bool' ,
         'char' ,
         'short',
         'int'  ,
         'long' ,
         'float',
         'double']

Ns = [2, 4, 3, 8, 16]

defined_in_cuda = [ 'char2'  ,
                    'char3'  ,
                    'char4'  ,
                    'short2' ,
                    'short3' ,
                    'short4' ,
                    'int2'   ,
                    'int3'   ,
                    'int4'   ,
                    'long2'  ,
                    'long3'  ,
                    'long4'  ,
                    'float2' ,
                    'float3' ,
                    'float4' ,
                    'double2',
                    'double3',
                    'double4' ]

unary_ops  = ['+', '-']
binary_ops = ['+', '-', '*', '/']

def varL(n):
    if n < 4:
        return chr(ord('w') + ((n + 1) % 4))
    else:
        return varN(n)

def varN(n):
    return 's' + str(n)

def define_typeN(type_, n):
    typeN = type_ + str(n)
    TYPEN = (type_ + str(n)).upper();

    define = ''

    if n == 3:
        if typeN in defined_in_cuda:
            define += '#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))\n'

        define += 'typedef ' + type_ + '4 ' + type_ + '3;\n'

        if typeN in defined_in_cuda:
            define += '#endif\n'

        return define

    if typeN in defined_in_cuda:
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

    for i in range(n):
        if(i < 4):
            define += '  union { ' + type_ + ' ' + varN(i) + ', ' + varL(i) + '; };\n'
        else:
            define += '  ' + type_ + ' ' + varN(i) + ';\n'

    define += '\n'

    for i in range(n + 1):
        start = '  occaFunction inline ' + typeN + '('

        args = (',\n' + (' ' * len(start))).join('const ' + type_ + ' &' + varL(j) + '_' for j in range(i))

        define += '  inline occaFunction ' + typeN + '(' + args + ') : \n' + \
                  '    ' + (',\n    '.join(varL(j) +'(' + ('0' if (i <= j) else varL(j) + '_') + ')' for j in range(n))) + \
                  ' {}\n'

        if i < n:
            define += '\n'

    define += '};\n'

    if typeN in defined_in_cuda:
        define += '#endif\n'

    define += '\n'

    for op in unary_ops:
        define += unary_op_def(type_, n, op)

    for op in binary_ops:
        define += binary_op_def(type_, n, op)

    define += '\n'
    define += '#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))\n'
    define += 'inline std::ostream& operator << (std::ostream &out, const ' + typeN + '& a){\n'

    l = ''; r = '';

    if type_ == 'bool':
        l = '('; r = ' ? "true" : "false")'

    define += '  out << "[" << ' + (' << ", "\n             << '.join((l + 'a.' + varL(i) + r) for i in range(n))) + '\n'

    define += '      << "]\\n";\n\n'
    define += '  return out;\n'
    define += '}\n'
    define += '#endif\n\n'

    return define

def unary_op_def(type_, n, op):
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

    for d in range(1 if isFloat else 2):
        ret         = '  return OCCA_' + TYPEN + '_CONSTRUCTOR(';
        indent[d]   = ' ' * len(ret)
        defines[d] += ret;

    for d in range(maxDefs):
        dm = (',' if (d < 2) else ';')

        if d == 2:
            defines[d] += indent[d]

        for i in range(n):
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

    for d in range(maxDefs):
        if 0 < d:
            defines[0] += '\n' + defines[d]

        if d == 2:
            defines[0] += indent[2] + 'return a;\n'

        defines[0] += '}\n'

    return defines[0]

def binary_op_def(type_, n, op):
    typeN = type_ + str(n)
    TYPEN = (type_ + str(n)).upper();

    a = 'a.'
    b = 'b.'

    defines = ['occaFunction inline ' + typeN + '  operator ' + op + '  (const ' + typeN + ' &a, const ' + typeN + ' &b){\n',
               'occaFunction inline ' + typeN + '  operator ' + op + '  (const ' + type_ + ' &a, const ' + typeN + ' &b){\n',
               'occaFunction inline ' + typeN + '  operator ' + op + '  (const ' + typeN + ' &a, const ' + type_ + ' &b){\n',
               'occaFunction inline ' + typeN + '& operator ' + op + '= (      ' + typeN + ' &a, const ' + typeN + ' &b){\n',
               'occaFunction inline ' + typeN + '& operator ' + op + '= (      ' + typeN + ' &a, const ' + type_ + ' &b){\n']

    aIsTypeN = [True, False, True , True, True ]
    bIsTypeN = [True, True , False, True, False]

    a = [[('a.' + varL(i)) if aIsTypeN[define] else 'a' for i in range(n)] for define in range(5)]
    b = [[('b.' + varL(i)) if bIsTypeN[define] else 'b' for i in range(n)] for define in range(5)]

    retDef = '  return OCCA_' + TYPEN + '_CONSTRUCTOR('

    for d in range(5):
        for i in range(n):
            if d < 3:
                if i == 0:
                    defines[d] += retDef

                defines[d] += a[d][i] + ' ' + op + ' ' + b[d][i]

                if i < (n - 1):
                    defines[d] += ',\n' + (' ' * len(retDef))
            else:
                defines[d] += '  a.' + varL(i) + ' ' + op + '= ' + b[d][i] + ';\n'

    for d in range(5):
        if 0 < d:
            defines[0] += '\n' + defines[d]

        if d < 3:
            defines[0] += ');\n'
        else:
            defines[0] += '  return a;\n'

        defines[0] += '}\n'

    return defines[0]

def define_type(type_):
    define = ''

    for n in Ns:
        typeN = type_ + str(n)

        comment = '//---[ ' + typeN + ' ]'
        define += comment + ('-' * (40 - len(comment))) + '\n'

        define += define_typeN(type_, n)

        define += '//' + ('=' * 38) + '\n\n\n'

    return define

def define_all_types():
    define  = '#if (!defined(OCCA_IN_KERNEL) || (!OCCA_USING_OPENCL))\n'
    define += '#  if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))\n'
    define += '#    include <iostream>\n'
    define += '#  endif\n\n'

    define += '#  ifndef OCCA_IN_KERNEL\n'
    define += '#    define occaFunction\n'
    define += 'namespace occa {\n'
    define += '#  endif\n\n'

    for type_ in types:
        define += define_type(type_)

    define += '#  ifndef OCCA_IN_KERNEL\n'
    define += '}\n'
    define += '#  endif\n\n'

    define += '#endif\n'

    return define

def intrinsic_headers():
    return """
#if OCCA_MMX
#  include <mmintrin.h>
#endif

#if OCCA_SSE
#  include <xmmintrin.h>
#endif

#if OCCA_SSE2
#  include <emmintrin.h>
#endif

#if OCCA_SSE3
#  include <pmmintrin.h>
#endif

#if OCCA_SSSE3
#  include <tmmintrin.h>
#endif

#if OCCA_SSE4_1
#  include <smmintrin.h>
#endif

#if OCCA_SSE4_2
#  include <nmmintrin.h>
#endif

#if OCCA_AVX
#  include <immintrin.h>
#endif
"""

def intrinsic_macros():
    return """
#if OCCA_USING_CPU && (OCCA_COMPILED_WITH & OCCA_INTEL_COMPILER)
#  define OCCA_CPU_SIMD_WIDTH OCCA_SIMD_WIDTH
#else
#  define OCCA_CPU_SIMD_WIDTH 0
#endif

#if 4 <= OCCA_CPU_SIMD_WIDTH
#  define occaLoadF4(DEST, SRC)   *((_m128*)&DEST) = __mm_load_ps((float*)&SRC)
#  define occaStoreF4(DEST, SRC)  _mm_store_ps((float*)&DEST, *((_m128*)&SRC)
#  define occaAddF4(V12, V1, V2)  *((_m128*)&V12) = __mm_add_ps(*((_m128*)&V1), *((_m128*)&V2))
#  define occaMultF4(V12, V1, V2) *((_m128*)&V12) = __mm_mul_ps(*((_m128*)&V1), *((_m128*)&V2))
#else
#  define occaLoadF4(DEST, SRC)   DEST = SRC
#  define occaStoreF4(DEST, SRC)  DEST = SRC
#  define occaAddF4(V12, V1, V2)  V12 = (V1 + V2)
#  define occaMultF4(V12, V1, V2) V12 = (V1 * V2)
#endif

#if 8 <= OCCA_CPU_SIMD_WIDTH
#  define occaLoadF8(DEST, SRC)   *((_m256*)&DEST) = __mm256_load_ps((float*)&SRC)
#  define occaStoreF8(DEST, SRC)  _mm256_store_ps((float*)&DEST, *((_m256*)&SRC)
#  define occaAddF8(V12, V1, V2)  *((_m256*)&V12) = __mm256_add_ps(*((_m256*)&V1), *((_m256*)&V2))
#  define occaMultF8(V12, V1, V2) *((_m256*)&V12) = __mm256_mul_ps(*((_m256*)&V1), *((_m256*)&V2))
#else
#  define occaLoadF8(DEST, SRC)   DEST = SRC
#  define occaStoreF8(DEST, SRC)  DEST = SRC
#  define occaAddF8(V12, V1, V2)  V12 = (V1 + V2)
#  define occaMultF8(V12, V1, V2) V12 = (V1 * V2)
#endif
"""

def vfloat_defines():
    return """
struct vfloat2 {
#if OCCA_MMX
  union {
    __m64 reg;
    float vec[2];
  };
#else
  float vec[2];
#endif
};

struct vfloat4 {
#if OCCA_SSE
  union {
    __m128 reg;
    float vec[4];
  };
#else
  float vec[4];
#endif
};

struct vfloat8 {
#if OCCA_AVX
  union {
    __m256 reg;
    float vec[4];
  };
#else
  float vec[4];
#endif
};

struct vdouble2 {
#if OCCA_SSE2
  union {
    __m128d reg;
    double vec[2];
  };
#else
  double vec[2];
#endif
};

struct vdouble4 {
#if OCCA_AVX
  union {
    __m256d reg;
    double vec[4];
  };
#else
  double vec[4];
#endif
};
"""

def intrinsic_functions():
    import vfloatOperators as vo

    contents = ''

    for function in vo.get_functions():
        contents += vo.make_function(function)

    return contents

def intrinsic_contents():

    return (intrinsic_headers() +
            '\n'                +
            intrinsic_macros()  +
            '\n'                +
            vfloat_defines()    +
            '\n'                +
            intrinsic_functions())

def gen_file_contents():
    return (define_all_types() +
            '\n'             +
            intrinsic_contents())

occa_dir = ENV['OCCA_DIR']

with open(occa_dir + '/include/occa/defines/vector.hpp', 'w') as f:
    f.write(gen_file_contents())