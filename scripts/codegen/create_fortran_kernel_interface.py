#! /usr/bin/env python3

"""
Generate the Fortran kernel module (occa_kernel_m.f90)
"""

import os
import re
import argparse


OCCA_DIR = os.environ.get(
    'OCCA_DIR',
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

CODEGEN_DIR = os.path.join(OCCA_DIR, 'scripts', 'codegen')
FORTRAN_DIR = os.path.join(OCCA_DIR, 'src', 'fortran')
C_SRC_DIR = os.path.join(OCCA_DIR, 'src', 'c')
C_INC_DIR = os.path.join(OCCA_DIR, 'include', 'occa', 'c')

OCCA_KERNEL_RUN_N = ('''
    subroutine occaKernelRunN%02d(kernel, argc, %s) &
               bind(C, name="occaKernelRunF%02d")
      import occaKernel, C_int, occaType
      implicit none
      type(occaKernel), value :: kernel
      integer(C_int), value, intent(in) :: argc
      type(occaType), value :: %s
    end subroutine
''').lstrip('\n')


OCCA_KERNEL_RUN = ('''
  subroutine occaKernelRun%02d(kernel, %s)
    type(occaKernel), value :: kernel
    type(occaType), value :: %s
    call occaKernelRunN%02d(kernel, %2d, %s)
  end subroutine
''').lstrip('\n')


def generate_edit_warning(fpath, comment_marker='!'):
    return \
        ('%s --------------[ DO NOT EDIT ]--------------\n' % comment_marker) + \
        ('%s  THIS IS AN AUTOMATICALLY GENERATED FILE\n' % comment_marker) + \
        ('%s  EDIT:\n' % comment_marker) + \
        ('%s    %s\n' % (comment_marker, os.path.relpath(__file__, OCCA_DIR))) + \
        (('%s    %s\n' %(comment_marker, os.path.relpath(fpath, OCCA_DIR))) if fpath else '') + \
        ('%s ===========================================\n' % comment_marker)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("-N","--NargsMax", type=int, default=20)
    args = parser.parse_args()

    MAX_ARGS = args.NargsMax
    if MAX_ARGS > 99:
        raise ValueError('The format was designed for max. %d arguments!' % MAX_ARGS)

    fname_in = os.path.join(CODEGEN_DIR, 'occa_kernel_m.f90.in')
    with open(fname_in, 'r') as f:
        f_in = f.readlines()

    fname = os.path.join(FORTRAN_DIR, 'occa_kernel_m.f90')
    print('Write OCCA Fortran kernel module to: %s' % (fname))
    with open(fname, 'w') as f:
        # Add edit warning to the top of the file
        f.write(generate_edit_warning(fname_in))
        f.write('\n')

        # Write header
        line = f_in.pop(0)
        while '@SUBROUTINE_occaKernelRunN@' not in line:
            f.write(line)
            line = f_in.pop(0)

        # Write occaKernelRunN subroutines
        for N in range(1, MAX_ARGS+1):
            arg_lst = ['arg%02d' % (i) for i in range(1, N+1)]
            if ((N-1)//4 > 0):
                args_f = []
                for i in range((N-1)//4 + 1):
                    args_f.append(', '.join(arg_lst[i*4:(i+1)*4]))
                args_f = (', &\n'+' '*46).join(args_f)
            else:
                args_f = ', '.join(arg_lst)

            if ((N-1)//6 > 0):
                args_d = []
                for i in range((N-1)//6 + 1):
                    args_d.append(', '.join(arg_lst[i*6:(i+1)*6]))
                args_d = (', &\n'+' '*31).join(args_d)
            else:
                args_d = ', '.join(arg_lst)

            if (N > 1): f.write('\n')
            f.write(OCCA_KERNEL_RUN_N % (N, args_f, N, args_d))

        # Write input file
        line = f_in.pop(0)
        while '@MODULE_PROCEDURE_occaKernelRun@' not in line:
            f.write(line)
            line = f_in.pop(0)

        # Write module procedure interface for occaKernelRun
        for N in range(1, MAX_ARGS+1):
            f.write('    module procedure occaKernelRun%02d\n' % (N))

        # Write input file
        line = f_in.pop(0)
        while '@SUBROUTINE_occaKernelRun@' not in line:
            f.write(line)
            line = f_in.pop(0)

        # Write occaKernelRun subroutines
        for N in range(1, MAX_ARGS+1):
            arg_lst = ['arg%02d' % (i) for i in range(1, N+1)]
            if ((N-1)//6 > 0):
                args_f = []
                for i in range((N-1)//6 + 1):
                    args_f.append(', '.join(arg_lst[i*6:(i+1)*6]))
                args_f = (', &\n'+' '*37).join(args_f)
            else:
                args_f = ', '.join(arg_lst)

            if ((N-1)//7 > 0):
                args_d = []
                for i in range((N-1)//7 + 1):
                    args_d.append(', '.join(arg_lst[i*7:(i+1)*7]))
                args_d = (', &\n'+' '*29).join(args_d)
            else:
                args_d = ', '.join(arg_lst)

            if ((N-1)//5 > 0):
                args_c = []
                for i in range((N-1)//5 + 1):
                    args_c.append(', '.join(arg_lst[i*5:(i+1)*5]))
                args_c = (', &\n'+' '*38).join(args_c)
            else:
                args_c = ', '.join(arg_lst)

            if (N > 1): f.write('\n')
            f.write(OCCA_KERNEL_RUN % (N, args_f, args_d, N, N, args_c))

        # Write remainder of the input file
        for line in f_in:
            f.write(line)

    # Generate interface functions with fixed arity to safely interface C with Fortran
    fname = os.path.join(C_SRC_DIR, 'kernel_fortran_interface.cpp')
    print('Write OCCA Fortran kernel interface C source to: %s' % fname)
    with open(fname, 'w') as f:
        f.write(generate_edit_warning(None, comment_marker='//'))
        f.write('\n')
        f.write('#include <occa/c/kernel_fortran_interface.h>\n')
        f.write('#include <occa/c/kernel.h>\n')
        f.write('\n')
        f.write('OCCA_START_EXTERN_C\n')
        f.write('\n')
        for N in range(1, MAX_ARGS+1):
            f.write('void occaKernelRunF%02d(occaKernel kernel,\n' % N);
            f.write('                      const int argc,\n');
            for n in range(1, N+1):
                f.write('                      occaType arg%02d%s\n' % (n, ') {' if n == N else ','))
            f.write('  occaKernelRunN(kernel, argc,\n')
            for n in range(1, N+1):
                f.write('%16s arg%02d%s' % ('', n, ');\n}\n\n' if n == N else ',\n'))
        f.write('OCCA_END_EXTERN_C\n')

    fname = os.path.join(C_INC_DIR, 'kernel_fortran_interface.h')
    print('Write OCCA Fortran kernel interface C headers to: %s' % fname)
    with open(fname, 'w') as f:
        f.write(generate_edit_warning(None, comment_marker='//'))
        f.write('\n')
        f.write('#include <occa/internal/c/types.hpp>\n')
        f.write('\n')
        f.write('OCCA_START_EXTERN_C\n')
        f.write('\n')
        for N in range(1, MAX_ARGS+1):
            f.write('void occaKernelRunF%02d(occaKernel kernel,\n' % N);
            f.write('                      const int argc,\n');
            for n in range(1, N+1):
                f.write('                      occaType arg%02d%s\n' % (n, ');\n\n' if n == N else ','))
        f.write('OCCA_END_EXTERN_C\n')
