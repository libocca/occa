#! /usr/bin/env python

""" Generate the Fortran kernel module (occa_kernel_m.f90)
"""

import os
import re
import argparse


OCCA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


EDIT_WARNING = ('''
! --------------[ DO NOT EDIT ]--------------
!  THIS IS AN AUTOMATICALLY GENERATED FILE
!  EDIT:
!    %s
!    %s
! ===========================================
'''.strip() % (os.path.relpath(__file__, OCCA_DIR),
               os.path.relpath('occa_kernel_m.f90.in', OCCA_DIR)))


OCCA_KERNEL_RUN_N = ('''
    subroutine occaKernelRunN%02d(kernel, argc, %s) &
               bind(C, name="occaKernelRunN")
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


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("-N","--NargsMax", type=int, default=20)
    args = parser.parse_args()

    MAX_ARGS = args.NargsMax
    if MAX_ARGS > 99:
        raise ValueError('The format was designed for max. 99 arguments!')

    with open('occa_kernel_m.f90.in', 'r') as f:
        f_in = f.readlines()

    fname = os.path.join(OCCA_DIR, 'src', 'fortran', 'occa_kernel_m.f90')
    print('Write OCCA Fortran kernel module to: %s' % (fname))
    with open(fname, 'w') as f:
        # Add edit warning to the top of the file
        f.write(EDIT_WARNING)
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
            f.write(OCCA_KERNEL_RUN_N % (N, args_f, args_d))

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
