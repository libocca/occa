#---[ Example Function ]----------------
#  Instructions to make your own function:

# Make a function without arguments with good naming convention
def vfloat4_plus_vfloat4():
    # Start with setup_function,
    #    - Binary      (a + b)
    #        left_type : 'vfloat4'
    #        right_type: 'vfloat4'
    #    - Left unary  (++a)
    #        left_type : 'vfloat4'
    #    - Right unary (a++)
    #        right_type: 'vfloat4'
    setup_function(vfloat4_plus_vfloat4        , # First argument is itself
                   ret_type        = 'vfloat4'  , # Must have return type (can be void)
                   left_type       = 'vfloat4 &', # Must have left and/or right type
                   operator        = '+'        , # Operator (+, -, +=, -=, =, etc)
                   right_type      = 'vfloat4 &',
                   instruction_set = 'SSE')       # Instruction set dependency


    # Return the source-code ONLY when using intrinsics
    # The argument names to the operator are as follows:
    #    - Binary: (a + b)
    #    - Unary : ++a or a++

    return """
    vfloat4 ret;
    ret.reg = _mm_add_ps(a.reg, b.reg);
    return ret;
"""
#=======================================


#---[ vfloat ]--------------------------
def vfloat4_plusEquals_vfloat4():
    setup_function(vfloat4_plusEquals_vfloat4  ,
                   ret_type        = 'vfloat4 &',
                   left_type       = 'vfloat4 &',
                   operator        = '+='       ,
                   right_type      = 'vfloat4 &',
                   instruction_set = 'SSE')

    return """
    a.reg = _mm_add_ps(a.reg, b.reg);
    return a;
"""
#=======================================


#---[ vdouble ]-------------------------
#=======================================


#---[ DON'T TOUCH ]---------------------
import re

INSTRUCTION_SETS = ['MMX',
                    'SSE', 'SSE2', 'SSE3', 'SSE4',
                    'AVX', 'AVX2']

MIN_INSTRUCTION_SET = { 'vfloat2' : 'MMX' ,
                        'vfloat4' : 'SSE' ,
                        'vfloat8' : 'AVX' ,
                        'vdouble2': 'SSE2',
                        'vdouble4': 'AVX' }

def min_instruction_set_dep(types, set1):
    sets = [MIN_INSTRUCTION_SET[t] for t in types if t in MIN_INSTRUCTION_SET]

    if len(sets) is 0:
        return set1

    indices = [INSTRUCTION_SETS.index(s) for s in sets]

    return INSTRUCTION_SETS[ max(indices) ]

def setup_function(func,
                   ret_type        = None,
                   operator       = None,
                   left_type       = None,
                   right_type      = None,
                   instruction_set = None):

    if hasattr(func, 'isAnIntrinsicFunction'):
        return

    if ret_type is None:
       raise ValueError('Return type must be passed to [setup_function]')
    if operator is None:
       raise ValueError('Operator must be passed to [setup_function]')
    if (left_type is None) and (right_type is None):
       raise ValueError('Left and/or right type must be passed to [setup_function]')

    func.isAnIntrinsicFunction = True

    func.ret_type  = ret_type
    func.operator = operator

    if left_type and right_type:
        func.operator_type = 'binary'
    elif left_type:
        func.operator_type = 'leftUnary'
    else:
        func.operator_type = 'rightUnary'

    func.left_by_reference  = ('&' in left_type)
    func.right_by_reference = ('&' in right_type)

    func.left_type  = re.sub('&', '', left_type).strip()  if left_type  else None
    func.right_type = re.sub('&', '', right_type).strip() if right_type else None

    func.full_left_type  = left_type
    func.full_right_type = right_type

    func.base_type = None
    func.vec_size  = None

    base_types = ['vfloat', 'vdouble']

    for base_type in base_types:
        type_ =                      left_type  if (left_type  and (base_type in left_type))  else None
        type_ = type_ if type_ else (right_type if (right_type and (base_type in right_type)) else None)

        if type_:
            func.base_type = base_type[1:]
            func.vec_size  = re.sub(base_type, '', type_)

    if func.base_type is None:
       raise ValueError('Left and/or right type must be vfloatX or vdoubleY (X in [2,4,8], Y in [2,4])')

    if instruction_set not in INSTRUCTION_SETS:
        raise ValueError('Instruction set must be in: ' + str(INSTRUCTION_SETS))

    func.instruction_set = min_instruction_set_dep([func.left_type, func.right_type],
                                                   instruction_set)

def make_function(f):
    import inspect

    argc = len(inspect.getargspec(f).args)

    source = ''

    if argc is not 0:
        return source

    intrinsic_source = f()

    if not hasattr(f, 'isAnIntrinsicFunction'):
        return source

    if   f.operator_type is 'binary':
        args = '{} a, {} b'.format(f.full_left_type, f.full_right_type)
    elif f.operator_type is 'leftUnary':
        args = '{} a'.format(f.full_left_type)
    elif f.operator_type is 'rightUnary':
        args = '{} a, int'.format(f.full_right_type)

    source += '#if OCCA_USING_CPU\n'

    source += '#if OCCA_{}\n'.format(f.instruction_set) # [-] Remove after fallback is implemented
    source += 'inline {} operator {} ({}) {{\n'.format(f.ret_type,
                                                       f.operator,
                                                       args)

    source += '#if OCCA_{}\n'.format(f.instruction_set)
    source += intrinsic_source
    source += '#else\n'
    source += '' # MISSING
    source += '#endif\n'

    source += '}\n';
    source += '#endif\n' # [-] Remove after fallback is implemented
    source += '#endif\n\n'

    return source

def get_functions():
    import sys, inspect

    module_self = sys.modules[__name__]

    return [f for (_,f) in inspect.getmembers(module_self, inspect.isfunction)]
#=======================================