using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace liboccaSharp  {
    public class Memory : OccaBase {


        public int SizeInBytes {
            get;
            private set;
        }

        internal Memory(IntPtr __occaMemory, int __sizeInBytes) {
            base.OccaHandle = __occaMemory;
            this.SizeInBytes = __sizeInBytes;
        }


        public void copyFrom<T>(T[] a, int size = -1, int offsetSrc = 0, int offsetDst = 0) where T : struct {
            CheckSize<T>(a, ref size, ref offsetSrc, ref offsetDst);
            
            {
                GCHandle lck = GCHandle.Alloc(a, GCHandleType.Pinned);
                IntPtr pa = Marshal.UnsafeAddrOfPinnedArrayElement(a, 0);
                this.copyFrom(pa, size, offsetSrc, offsetDst);
                lck.Free();
            }
        }

        private void CheckSize<T>(T[] a, ref int size, ref int offsetSrc, ref int offsetDst) where T : struct {
            if(size < 0)
                size = a.Length;
            int tsz = Marshal.SizeOf(typeof(T));
            size *= tsz;
            offsetSrc *= tsz;
            offsetDst *= tsz;

            if((size + offsetSrc) > this.SizeInBytes)
                throw new ArgumentException("buffer overflow for source buffer");
            if((size + offsetDst) > this.SizeInBytes)
                throw new ArgumentException("buffer overflow for destination buffer");
        }
                
        public void copyFrom(IntPtr p, int sizeInBytes, int offsetSrc, int offsetDst) {
            CheckState();
            unsafe {
                UIntPtr _sizeInBytes = (UIntPtr)sizeInBytes;
                UIntPtr _offset = (UIntPtr)offsetDst;
                occaCopyPtrToMem(base.OccaHandle, (void*)(p + offsetSrc), _sizeInBytes, _offset);
            }
        }

        public void copyTo<T>(T[] a, int size = -1, int offsetSrc = 0, int offsetDst = 0) where T : struct {
            CheckSize<T>(a, ref size, ref offsetSrc, ref offsetDst);

            {
                GCHandle lck = GCHandle.Alloc(a, GCHandleType.Pinned);
                IntPtr pa = Marshal.UnsafeAddrOfPinnedArrayElement(a, 0);
                this.copyTo(pa, size, offsetSrc, offsetDst);
                lck.Free();
            }
        }
        
        public void copyTo(IntPtr p, int sizeInBytes, int offsetSrc, int offsetDst) {
            CheckState();
            unsafe {
                UIntPtr _sizeInBytes = (UIntPtr) sizeInBytes;
                UIntPtr _offset = (UIntPtr) offsetSrc;
                occaCopyMemToPtr((void*)(p + offsetDst), base.OccaHandle, _sizeInBytes, _offset);
            }
        }

        public override void Dispose() {
            CheckState();
            occaMemoryFree(base.OccaHandle);
            this.OccaHandle = IntPtr.Zero;
        }

        #region WRAPPERS
        [DllImport("occa_c")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        extern unsafe static string occaMemoryMode(IntPtr occaMemory_memory);

        [DllImport("occa_c")]
        extern unsafe static void occaCopyMemToMem(IntPtr occaMemory_dest, IntPtr occaMemory_src,
                              UIntPtr bytes,
                              UIntPtr destOffset,
                              UIntPtr srcOffset);
        [DllImport("occa_c")]
        extern unsafe static void occaCopyPtrToMem(IntPtr occaMemory_dest, void* src,
                              UIntPtr bytes,
                              UIntPtr offset);
        [DllImport("occa_c")]
        extern unsafe static void occaCopyMemToPtr(void* dest, IntPtr occaMemory_src,
                              UIntPtr bytes,
                              UIntPtr offset);
        [DllImport("occa_c")]
        extern unsafe static void occaAsyncCopyMemToMem(IntPtr occaMemory_dest, IntPtr occaMemory_src,
                                   UIntPtr bytes,
                                   UIntPtr destOffset,
                                   UIntPtr srcOffset);
        [DllImport("occa_c")]
        extern unsafe static void occaAsyncCopyPtrToMem(IntPtr occaMemory_dest, void* src,
                                   UIntPtr bytes,
                                   UIntPtr offset);
        [DllImport("occa_c")]
        extern unsafe static void occaAsyncCopyMemToPtr(void* dest, IntPtr occaMemory_src,
                                   UIntPtr bytes,
                                   UIntPtr offset);
        [DllImport("occa_c")]
        extern unsafe static void occaMemorySwap(IntPtr occaMemory_memoryA, IntPtr occaMemory_memoryB);

        [DllImport("occa_c")]
        extern unsafe static void occaMemoryFree(IntPtr occaMemory_memory);
        #endregion
    }
}
