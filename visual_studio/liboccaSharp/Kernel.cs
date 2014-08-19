using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace liboccaSharp {
    public class Kernel : OccaBase {
        internal Kernel(IntPtr h) {
            base.OccaHandle = h;
            itemsDim = new occaDim();
            groupsDim = new occaDim();
            m_Dims = 1;
        }

        override public void Dispose() {
            CheckState();
            occaKernelFree(this.OccaHandle);
            this.OccaHandle = IntPtr.Zero;
        }

        public void Invoke(ArgumentList args) {
            base.CheckState();

            for(int i = this.Dims; i < 3; i++) {
                this.itemsDim[i] = 1;
                this.groupsDim[i] = 1;
            }

            occaKernelSetAllWorkingDims(this.OccaHandle, this.m_Dims, this.itemsDim.x, this.itemsDim.y, this.itemsDim.z, this.groupsDim.x, this.groupsDim.y, this.groupsDim.z);
            occaKernelRun_(this.OccaHandle, args.OccaHandle);
        }

        public void Invoke(params object[] args) {
            base.CheckState();
            using(var agl = new ArgumentList(args)) {
                this.Invoke(agl);
            }
        }

        public int PreferredDimSize {
            get {
                return occaKernelPreferredDimSize(this.OccaHandle);
            }
        }

        int m_Dims = 1;

        public int Dims {
            get {
                return m_Dims;
            }
            set {
                if(value < 1 || value > 3)
                    throw new ArgumentOutOfRangeException();
                this.m_Dims = value;
            }
        }

        public occaDim itemsDim {
            get;
            private set;
        }

        public occaDim groupsDim {
            get;
            private set;
        }

        public Mode Mode {
            get {
                return (Mode) Enum.Parse(typeof(Mode), occaKernelMode(this.OccaHandle));
            }
        }

        #region WRAPPERS

        [DllImport("occa_c")]
        extern unsafe static occaDim occaGenDim(UIntPtr x, UIntPtr y, UIntPtr z);
        
        [DllImport("occa_c")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        extern unsafe static string occaKernelMode(IntPtr occaKernel_kernel);
        
        [DllImport("occa_c")]
        extern unsafe static int occaKernelPreferredDimSize(IntPtr occaKernel_kernel);
        
        [DllImport("occa_c")]
        extern unsafe static void occaKernelSetWorkingDims(IntPtr occaKernel_kernel,
                                      int dims,
                                      occaDim items,
                                      occaDim groups);
        
        [DllImport("occa_c")]
        extern unsafe static void occaKernelSetAllWorkingDims(IntPtr occaKernel_kernel,
                                         int dims,
                                         UIntPtr itemsX, UIntPtr itemsY, UIntPtr itemsZ,
                                         UIntPtr groupsX, UIntPtr groupsY, UIntPtr groupsZ);
        
        [DllImport("occa_c")]
        extern unsafe static double occaKernelTimeTaken(IntPtr occaKernel_kernel);
        
        
        // Note the _
        //   Macro that is called > API function that is never seen
        [DllImport("occa_c")]
        extern unsafe static void occaKernelRun_(IntPtr occaKernel_kernel,
                            IntPtr occaArgumentList_list);

        //OCCA_C_KERNEL_RUN_DEFINITIONS;
        [DllImport("occa_c")]
        extern unsafe static void occaKernelFree(IntPtr occaKernel_kernel);
        
        #endregion
    }
}
