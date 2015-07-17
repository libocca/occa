using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace liboccaSharp {
    public enum Mode {
        Pthreads = (1 << 20),
        OpenMP   = (1 << 21),
        OpenCL   = (1 << 22),
        CUDA     = (1 << 23),
        COI      = (1 << 24)
    }
}
