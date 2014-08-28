using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace liboccaSharp {
    public class KernelInfo : OccaBase {
        
        public KernelInfo() {
            base.OccaHandle = occaGenKernelInfo();
        }
        
        public void AddDefine(string macro, object value) {
            occaKernelInfoAddDefine(this.OccaHandle, macro, ArgumentList.Obj2OccaType(value));
        }
        
        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaKernelInfo*/ occaGenKernelInfo();

        [DllImport("occa_c")]
        extern unsafe static void occaKernelInfoAddDefine(IntPtr occaKernelInfo_info,
                                     [MarshalAs(UnmanagedType.LPStr)] string macro,
                                     IntPtr occaType_value);

        [DllImport("occa_c")]
        extern unsafe static void occaKernelInfoFree(IntPtr occaKernelInfo_info);


        override public void Dispose() {
            CheckState();
            occaKernelInfoFree(this.OccaHandle);
            this.OccaHandle = IntPtr.Zero;
        }
    }
}
