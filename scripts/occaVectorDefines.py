types = ['bool',
         'char',
         'short',
         'int',
         'long',
         'float',
         'double'];

vTypes = [t + s for t in types for s in ['2', '3', '4', '8', '16']];

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
                'double2' : True };

def varN(n):
    if(n < 4):
        return chr(ord('w') + ((n + 1) % 4));
    else:
        return 's' + str(n);

def defineTypeN(type_, n):


hpp = open('../include/defines/occaVectorOperators.hpp', 'w')



hpp.close()