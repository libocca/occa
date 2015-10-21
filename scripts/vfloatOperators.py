def vfloat4_plus_vfloat4():
    setup_function(vfloat4_plus_vfloat4,
                   retType        = 'vfloat4',
                   leftType       = 'vfloat4 &',
                   operator       = '+',
                   rightType      = 'vfloat4 &',
                   instructionSet = 'SSE2')

    return ''

#---[ DON'T TOUCH ]---------------------
import re

def setup_function(func,
                   retType        = None,
                   operator       = None,
                   leftType       = None,
                   rightType      = None,
                   instructionSet = None):

    if hasattr(func, 'isAnIntrinsicFunction'):
        return

    if retType is None:
       raise ValueError('Return type must be passed to [setup_function]')
    if operator is None:
       raise ValueError('Operator must be passed to [setup_function]')
    if (leftType is None) and (rightType is None):
       raise ValueError('Left and/or right type must be passed to [setup_function]')

    func.isAnIntrinsicFunction = True

    func.retType  = retType
    func.operator = operator

    if leftType and rightType:
        func.operatorType = 'binary'
    elif leftType:
        func.operatorType = 'leftUnary'
    else:
        func.operatorType = 'rightUnary'

    func.leftByReference  = ('&' in leftType)
    func.rightByReference = ('&' in rightType)

    func.leftType  = re.sub('&', '', leftType).strip()  if leftType  else None
    func.rightType = re.sub('&', '', rightType).strip() if rightType else None

    func.baseType = None
    func.vecSize  = None

    baseTypes = ['vfloat', 'vdouble']

    for baseType in baseTypes:
        type_ =                      leftType  if (leftType  and (baseType in leftType))  else None
        type_ = type_ if type_ else (rightType if (rightType and (baseType in rightType)) else None)

        if type_:
            func.baseType = baseType[1:]
            func.vecSize  = re.sub(baseType, '', type_)

    if func.baseType is None:
       raise ValueError('Left and/or right type must be vfloatX or vdoubleY (X in [2,4,8], Y in [2,4])')

    instructionSets = ['MMX',
                       'SSE', 'SSE2', 'SSE3', 'SSE4',
                       'AVX', 'AVX2']

    if instructionSet not in instructionSets:
        raise ValueError('Instruction set must be in: ' + str(instructionSets))

    func.instructionSet = instructionSet

def make_function(f):
    import inspect

    argc = len(inspect.getargspec(f).args)

    if argc is not 0:
        return ''

    source = f()

    if not hasattr(f, 'isAnIntrinsicFunction'):
        return ''

    return (source + '\n')

def get_functions():
    import sys, inspect

    module_self = sys.modules[__name__]

    return [f for (_,f) in inspect.getmembers(module_self, inspect.isfunction)]
#=======================================