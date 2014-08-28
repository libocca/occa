using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace liboccaSharp {
    public class Device : OccaBase {

        public Mode Mode {
            get;
            private set;
        }

        public Device(Mode m, int platformID, int deviceID) {
            this.Mode = m;
            base.OccaHandle = occaGetDevice(m.ToString(), platformID, deviceID);
        }

        public Kernel buildKernelFromSource(string filename, string functionName, KernelInfo info = null) {
            CheckState();
            return new Kernel(occaBuildKernelFromSource(this.OccaHandle, filename, functionName, info != null ? info.OccaHandle : IntPtr.Zero));
        }

        public Memory malloc(int SizeInBytes) {
            UIntPtr __SizeInBytes = (UIntPtr)SizeInBytes;
            return new Memory(occaDeviceMalloc(this.OccaHandle, __SizeInBytes, IntPtr.Zero), SizeInBytes);
        }

        override public void Dispose() {
            CheckState();
            occaDeviceFree(this.OccaHandle);
            this.OccaHandle = IntPtr.Zero;
        }

        public void Flush() {
            CheckState();
            occaDeviceFlush(this.OccaHandle);
        }

        public void Finish() {
            CheckState();
            occaDeviceFinish(this.OccaHandle);
        }
        
        #region WRAPPERS

        [DllImport("occa_c")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        extern unsafe static byte* occaDeviceMode(IntPtr occaDevice_device);

        [DllImport("occa_c")]
        extern unsafe static void occaDeviceSetCompiler(IntPtr occaDevice_device,
                             [MarshalAs(UnmanagedType.LPStr)] string compiler);
        [DllImport("occa_c")]
        extern unsafe static void occaDeviceSetCompilerFlags(IntPtr occaDevice_device,
                                  [MarshalAs(UnmanagedType.LPStr)] string compilerFlags);
        
        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaDevice*/ occaGetDevice([MarshalAs(UnmanagedType.LPStr)] string mode, int arg1, int arg2);
        
        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaKernel*/ occaBuildKernelFromSource(IntPtr occaDevice_device, [MarshalAs(UnmanagedType.LPStr)] string filename, [MarshalAs(UnmanagedType.LPStr)] string functionName, IntPtr occaKernelInfo_info);

        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaKernel*/ occaBuildKernelFromBinary(IntPtr occaDevice_device,
                                       [MarshalAs(UnmanagedType.LPStr)] string filename,
                                       [MarshalAs(UnmanagedType.LPStr)] string functionName);

        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaKernel*/ occaBuildKernelFromLoopy(IntPtr occaDevice_device,
                                      [MarshalAs(UnmanagedType.LPStr)] string filename,
                                      [MarshalAs(UnmanagedType.LPStr)] string functionName,
                                      [MarshalAs(UnmanagedType.LPStr)] string pythonCode); // return 

        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaMemory*/ occaDeviceMalloc(IntPtr occaDevice_device,
                        UIntPtr bytes,
                        IntPtr source);

        [DllImport("occa_c")]
        extern unsafe static void occaDeviceFlush(IntPtr occaDevice_device);

        [DllImport("occa_c")]
        extern unsafe static void occaDeviceFinish(IntPtr occaDevice_device);

        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaStream*/ occaDeviceGenStream(IntPtr occaDevice_device);

        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaStream*/ occaDeviceGetStream(IntPtr occaDevice_device);

        [DllImport("occa_c")]
        extern unsafe static void occaDeviceSetStream(IntPtr occaDevice_device, IntPtr occaStream_stream);

        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaTag*/ occaDeviceTagStream(IntPtr occaDevice_device);

        [DllImport("occa_c")]
        extern unsafe static double occaDeviceTimeBetweenTags(IntPtr occaDevice_device,
                                   IntPtr occaTag_startTag, IntPtr occaTag_endTag);

        [DllImport("occa_c")]
        extern unsafe static void occaDeviceStreamFree(IntPtr occaDevice_device, IntPtr occaStream_stream);

        [DllImport("occa_c")]
        extern unsafe static void occaDeviceFree(IntPtr occaDevice_device);


        #endregion

    }
}
