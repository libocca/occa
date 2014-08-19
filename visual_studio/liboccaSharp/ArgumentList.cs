using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace liboccaSharp {
    public class ArgumentList : OccaBase {

        public ArgumentList() {
            base.OccaHandle = occaGenArgumentList();
        }

        public ArgumentList(object[] args) : this() {
            for(int i = 0; i < args.Length; i++) {
                AddArg(i, args[i]);
            }
        }

        public override void Dispose() {
            base.CheckState();
            occaArgumentListClear(base.OccaHandle);
            occaArgumentListFree(base.OccaHandle);
            base.OccaHandle = IntPtr.Zero;
        }

        public void Free() {
            occaArgumentListFree(base.OccaHandle);
        }

        public void AddArg(int argPos, object type) {
            if(type is Memory) {
                occaArgumentListAddArg(this.OccaHandle, argPos, ((Memory)type).OccaHandle);
            } else {
                occaArgumentListAddArg(this.OccaHandle, argPos, Obj2OccaType(type));
            }
        }

        static public IntPtr Obj2OccaType(object type) {
            if(type is int) {
                return occaInt((int)type);
            } else if(type is uint) {
                return occaUInt((uint)type);
            } else if(type is sbyte) {
                return occaChar((sbyte)type);
            } else if(type is byte) {
                return occaUChar((byte)type);
            } else if(type is short) {
                return occaShort((short)type);
            } else if(type is ushort) {
                return occaUShort((ushort)type);
            } else if(type is float) {
                return occaFloat((float)type);
            } else if(type is double) {
                return occaDouble((double)type);
            } else {
                throw new NotSupportedException();
            }
        }
                
        [DllImport("occa_c")]
        extern unsafe static IntPtr /*occaArgumentList*/ occaGenArgumentList();
        [DllImport("occa_c")]
        extern unsafe static void occaArgumentListClear(IntPtr occaArgumentList_list);
        [DllImport("occa_c")]
        extern unsafe static void occaArgumentListFree(IntPtr occaArgumentList_list);
        [DllImport("occa_c")]
        extern unsafe static void occaArgumentListAddArg(IntPtr occaArgumentList_list, int argPos, IntPtr type);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaInt(int value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaUInt(uint value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaChar(sbyte value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaUChar(byte value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaShort(short value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaUShort(ushort value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaLong(int value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaULong(uint value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaFloat(float value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaDouble(double value);
        [DllImport("occa_c")]
        extern static IntPtr /*occaType*/ occaString([MarshalAs(UnmanagedType.LPStr)] string str);
    }
}
