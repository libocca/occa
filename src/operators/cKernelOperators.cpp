/* The MIT License (MIT)
 * 
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */


void OCCA_RFUNC occaKernelRun1(occaKernel kernel, occaType arg0) {
  occaType args[1] = { arg0 };
  occaKernelRunN(kernel, 1, args);
}


void OCCA_RFUNC occaKernelRun2(occaKernel kernel, occaType arg0,  occaType arg1) {
  occaType args[2] = { arg0, arg1 };
  occaKernelRunN(kernel, 2, args);
}


void OCCA_RFUNC occaKernelRun3(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2) {
  occaType args[3] = { arg0, arg1, arg2 };
  occaKernelRunN(kernel, 3, args);
}


void OCCA_RFUNC occaKernelRun4(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3) {
  occaType args[4] = { arg0, arg1, arg2, arg3 };
  occaKernelRunN(kernel, 4, args);
}


void OCCA_RFUNC occaKernelRun5(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4) {
  occaType args[5] = { arg0, arg1, arg2, arg3, arg4 };
  occaKernelRunN(kernel, 5, args);
}


void OCCA_RFUNC occaKernelRun6(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5) {
  occaType args[6] = { arg0, arg1, arg2, arg3, arg4, arg5 };
  occaKernelRunN(kernel, 6, args);
}


void OCCA_RFUNC occaKernelRun7(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6) {
  occaType args[7] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6 };
  occaKernelRunN(kernel, 7, args);
}


void OCCA_RFUNC occaKernelRun8(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7) {
  occaType args[8] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 };
  occaKernelRunN(kernel, 8, args);
}


void OCCA_RFUNC occaKernelRun9(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8) {
  occaType args[9] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 };
  occaKernelRunN(kernel, 9, args);
}


void OCCA_RFUNC occaKernelRun10(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9) {
  occaType args[10] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 };
  occaKernelRunN(kernel, 10, args);
}


void OCCA_RFUNC occaKernelRun11(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10) {
  occaType args[11] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10 };
  occaKernelRunN(kernel, 11, args);
}


void OCCA_RFUNC occaKernelRun12(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11) {
  occaType args[12] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11 };
  occaKernelRunN(kernel, 12, args);
}


void OCCA_RFUNC occaKernelRun13(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12) {
  occaType args[13] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12 };
  occaKernelRunN(kernel, 13, args);
}


void OCCA_RFUNC occaKernelRun14(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13) {
  occaType args[14] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13 };
  occaKernelRunN(kernel, 14, args);
}


void OCCA_RFUNC occaKernelRun15(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14) {
  occaType args[15] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14 };
  occaKernelRunN(kernel, 15, args);
}


void OCCA_RFUNC occaKernelRun16(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15) {
  occaType args[16] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15 };
  occaKernelRunN(kernel, 16, args);
}


void OCCA_RFUNC occaKernelRun17(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16) {
  occaType args[17] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16 };
  occaKernelRunN(kernel, 17, args);
}


void OCCA_RFUNC occaKernelRun18(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17) {
  occaType args[18] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17 };
  occaKernelRunN(kernel, 18, args);
}


void OCCA_RFUNC occaKernelRun19(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18) {
  occaType args[19] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18 };
  occaKernelRunN(kernel, 19, args);
}


void OCCA_RFUNC occaKernelRun20(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19) {
  occaType args[20] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19 };
  occaKernelRunN(kernel, 20, args);
}


void OCCA_RFUNC occaKernelRun21(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20) {
  occaType args[21] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20 };
  occaKernelRunN(kernel, 21, args);
}


void OCCA_RFUNC occaKernelRun22(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21) {
  occaType args[22] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21 };
  occaKernelRunN(kernel, 22, args);
}


void OCCA_RFUNC occaKernelRun23(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22) {
  occaType args[23] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22 };
  occaKernelRunN(kernel, 23, args);
}


void OCCA_RFUNC occaKernelRun24(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23) {
  occaType args[24] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23 };
  occaKernelRunN(kernel, 24, args);
}


void OCCA_RFUNC occaKernelRun25(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24) {
  occaType args[25] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24 };
  occaKernelRunN(kernel, 25, args);
}


void OCCA_RFUNC occaKernelRun26(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25) {
  occaType args[26] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25 };
  occaKernelRunN(kernel, 26, args);
}


void OCCA_RFUNC occaKernelRun27(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26) {
  occaType args[27] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26 };
  occaKernelRunN(kernel, 27, args);
}


void OCCA_RFUNC occaKernelRun28(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27) {
  occaType args[28] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27 };
  occaKernelRunN(kernel, 28, args);
}


void OCCA_RFUNC occaKernelRun29(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28) {
  occaType args[29] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28 };
  occaKernelRunN(kernel, 29, args);
}


void OCCA_RFUNC occaKernelRun30(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29) {
  occaType args[30] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29 };
  occaKernelRunN(kernel, 30, args);
}


void OCCA_RFUNC occaKernelRun31(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30) {
  occaType args[31] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30 };
  occaKernelRunN(kernel, 31, args);
}


void OCCA_RFUNC occaKernelRun32(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31) {
  occaType args[32] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31 };
  occaKernelRunN(kernel, 32, args);
}


void OCCA_RFUNC occaKernelRun33(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32) {
  occaType args[33] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32 };
  occaKernelRunN(kernel, 33, args);
}


void OCCA_RFUNC occaKernelRun34(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33) {
  occaType args[34] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33 };
  occaKernelRunN(kernel, 34, args);
}


void OCCA_RFUNC occaKernelRun35(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34) {
  occaType args[35] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34 };
  occaKernelRunN(kernel, 35, args);
}


void OCCA_RFUNC occaKernelRun36(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35) {
  occaType args[36] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35 };
  occaKernelRunN(kernel, 36, args);
}


void OCCA_RFUNC occaKernelRun37(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36) {
  occaType args[37] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36 };
  occaKernelRunN(kernel, 37, args);
}


void OCCA_RFUNC occaKernelRun38(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37) {
  occaType args[38] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37 };
  occaKernelRunN(kernel, 38, args);
}


void OCCA_RFUNC occaKernelRun39(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38) {
  occaType args[39] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38 };
  occaKernelRunN(kernel, 39, args);
}


void OCCA_RFUNC occaKernelRun40(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39) {
  occaType args[40] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39 };
  occaKernelRunN(kernel, 40, args);
}


void OCCA_RFUNC occaKernelRun41(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40) {
  occaType args[41] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40 };
  occaKernelRunN(kernel, 41, args);
}


void OCCA_RFUNC occaKernelRun42(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40,  occaType arg41) {
  occaType args[42] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41 };
  occaKernelRunN(kernel, 42, args);
}


void OCCA_RFUNC occaKernelRun43(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40,  occaType arg41, 
                      occaType arg42) {
  occaType args[43] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42 };
  occaKernelRunN(kernel, 43, args);
}


void OCCA_RFUNC occaKernelRun44(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40,  occaType arg41, 
                      occaType arg42,  occaType arg43) {
  occaType args[44] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43 };
  occaKernelRunN(kernel, 44, args);
}


void OCCA_RFUNC occaKernelRun45(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40,  occaType arg41, 
                      occaType arg42,  occaType arg43,  occaType arg44) {
  occaType args[45] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43, arg44 };
  occaKernelRunN(kernel, 45, args);
}


void OCCA_RFUNC occaKernelRun46(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40,  occaType arg41, 
                      occaType arg42,  occaType arg43,  occaType arg44, 
                      occaType arg45) {
  occaType args[46] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45 };
  occaKernelRunN(kernel, 46, args);
}


void OCCA_RFUNC occaKernelRun47(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40,  occaType arg41, 
                      occaType arg42,  occaType arg43,  occaType arg44, 
                      occaType arg45,  occaType arg46) {
  occaType args[47] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46 };
  occaKernelRunN(kernel, 47, args);
}


void OCCA_RFUNC occaKernelRun48(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40,  occaType arg41, 
                      occaType arg42,  occaType arg43,  occaType arg44, 
                      occaType arg45,  occaType arg46,  occaType arg47) {
  occaType args[48] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46, arg47 };
  occaKernelRunN(kernel, 48, args);
}


void OCCA_RFUNC occaKernelRun49(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40,  occaType arg41, 
                      occaType arg42,  occaType arg43,  occaType arg44, 
                      occaType arg45,  occaType arg46,  occaType arg47, 
                      occaType arg48) {
  occaType args[49] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46, arg47, arg48 };
  occaKernelRunN(kernel, 49, args);
}


void OCCA_RFUNC occaKernelRun50(occaKernel kernel, occaType arg0,  occaType arg1,  occaType arg2, 
                      occaType arg3,  occaType arg4,  occaType arg5, 
                      occaType arg6,  occaType arg7,  occaType arg8, 
                      occaType arg9,  occaType arg10,  occaType arg11, 
                      occaType arg12,  occaType arg13,  occaType arg14, 
                      occaType arg15,  occaType arg16,  occaType arg17, 
                      occaType arg18,  occaType arg19,  occaType arg20, 
                      occaType arg21,  occaType arg22,  occaType arg23, 
                      occaType arg24,  occaType arg25,  occaType arg26, 
                      occaType arg27,  occaType arg28,  occaType arg29, 
                      occaType arg30,  occaType arg31,  occaType arg32, 
                      occaType arg33,  occaType arg34,  occaType arg35, 
                      occaType arg36,  occaType arg37,  occaType arg38, 
                      occaType arg39,  occaType arg40,  occaType arg41, 
                      occaType arg42,  occaType arg43,  occaType arg44, 
                      occaType arg45,  occaType arg46,  occaType arg47, 
                      occaType arg48,  occaType arg49) {
  occaType args[50] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40, arg41, arg42, arg43, arg44, arg45, arg46, arg47, arg48, arg49 };
  occaKernelRunN(kernel, 50, args);
}
