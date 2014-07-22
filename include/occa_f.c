#ifndef OCCA_FBASE_HEADER
#define OCCA_FBASE_HEADER

#include "occaCBase.hpp"

#  ifdef __cplusplus
extern "C" {
#  endif

  //---[ TypeCasting ]------------------
  occaType occaInt_fc(int value){
    return occaInt(value);
  }

  occaType occaUInt_fc(unsigned int value){
    return occaUInt(value);
  }

  occaType occaChar_fc(char value){
    return occaChar(value);
  }
  occaType occaUChar_fc(unsigned char value){
    return occaUChar(value);
  }

  occaType occaShort_fc(short value){
    return occaShort(value);
  }
  occaType occaUShort_fc(unsigned short value){
    return occaUShort(value);
  }

  occaType occaLong_fc(long value){
    return occaLong(value);
  }
  occaType occaULong_fc(unsigned long value){
    return occaULong(value);
  }

  occaType occaFloat_fc(float value){
    return occaFloat(value);
  }
  occaType occaDouble_fc(double value){
    return occaDouble(value);
  }

  occaType occaString_fc(char *str OCCA_F2C_LSTR(str_l)
                         OCCA_F2C_RSTR(str_l)){
    char *str_c;
    OCCA_F2C_ALLOC_STR(str, str_l, str_c);

    occaType ret = occaString(str_c);

    OCCA_F2C_FREE_STR(str, str_c);

    return ret;
  }
  //====================================


  //---[ Device ]-----------------------
  const char* occaDeviceMode_fc(occaDevice device){ // [-]
    occaDeviceMode(device);
  }

  void occaDeviceSetCompiler_fc(occaDevice device,
                                const char *compiler OCCA_F2C_LSTR(compiler_l),
                                OCCA_F2C_RSTR(compiler_l)){
    char *compiler_c;
    OCCA_F2C_ALLOC_STR(compiler, compiler_l, compiler_c);

    occaDeviceSetCompiler(device, compiler_c);

    OCCA_F2C_FREE_STR(compiler, compiler_c);
  }

  void occaDeviceSetCompilerFlags_fc(occaDevice device,
                                     const char *compilerFlags OCCA_F2C_LSTR(compilerFlags_l)
                                     OCCA_F2C_RSTR(compilerFlags_l)){
    char *compilerFlags_c;
    OCCA_F2C_ALLOC_STR(compilerFlags, compilerFlags_l, compilerFlags_c);

    occaDeviceSetCompilerFlags(device, compilerFlags_c);

    OCCA_F2C_FREE_STR(compilerFlags, compilerFlags_c);
  }

  occaDevice occaGetDevice_fc(const char *mode OCCA_F2C_LSTR(mode_l),
                              int arg1, int arg2
                              OCCA_F2C_RSTR(mode_l)){
    char *mode_c;
    OCCA_F2C_ALLOC_STR(mode, mode_l, mode_c);

    occaDevice ret = occaGetDevice(mode_c, arg1, arg2);

    OCCA_F2C_FREE_STR(mode, mode_c);

    return ret;
  }

  occaKernel occaBuildKernelFromSource_fc(occaDevice device,
                                          const char *filename     OCCA_F2C_LSTR(filename_l),
                                          const char *functionName OCCA_F2C_LSTR(functionName_l),
                                          occaKernelInfo info
                                          OCCA_F2C_RSTR(filename_l)
                                          OCCA_F2C_RSTR(functionName_l)){
    char *filename_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    occaKernel ret = occaBuildKernelFromSource(device, filename_c, functionName_c, info);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);

    return ret;
  }

  occaKernel occaBuildKernelFromBinary_fc(occaDevice device,
                                          const char *filename     OCCA_F2C_LSTR(filename_l),
                                          const char *functionName OCCA_F2C_LSTR(functionName_l)
                                          OCCA_F2C_RSTR(filename_l)
                                          OCCA_F2C_RSTR(functionName_l)){
    char *filename_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    occaBuildKernelFromBinary(device, filename_c, functionName_c);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  occaKernel occaBuildKernelFromLoopy_fc(occaDevice device,
                                         const char *filename     OCCA_F2C_LSTR(filename_l),
                                         const char *functionName OCCA_F2C_LSTR(functionName_l),
                                         const char *pythonCode   OCCA_F2C_LSTR(pythonCode_l)
                                         OCCA_F2C_RSTR(filename_l)
                                         OCCA_F2C_RSTR(functionName_l)
                                         OCCA_F2C_RSTR(pythonCode_l)){
    char *filename_c, *functionName_c, *pythonCode_c;

    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);
    OCCA_F2C_ALLOC_STR(pythonCode  , pythonCode_l  , pythonCode_c);

    occaKernel ret = occaBuildKernelFromLoopy(device, filename_c, functionName_c, pythonCode_c);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
    OCCA_F2C_FREE_STR(pythonCode  , pythonCode_c);

    return ret;
  }

  occaMemory occaDeviceMalloc_fc(occaDevice device,
                                 uintptr_t bytes,
                                 void *source){
    return occaDeviceMalloc(device, bytes, source);
  }

  void occaDeviceFlush_fc(occaDevice device){
    occaDeviceFlush(device);
  }
  void occaDeviceFinish_fc(occaDevice device){
    occaDeviceFinish(device);
  }

  occaStream occaDeviceGenStream_fc(occaDevice device){
    return occaDeviceGenStream(device);
  }
  occaStream occaDeviceGetStream_fc(occaDevice device){
    return occaDeviceGetStream(device);
  }
  void occaDeviceSetStream_fc(occaDevice device, occaStream stream){
    return occaDeviceSetStream(device, stream);
  }

  occaTag occaDeviceTagStream_fc(occaDevice device){
    return occaDeviceTagStream(device);
  }
  double occaDeviceTimeBetweenTags_fc(occaDevice device,
                                      occaTag startTag, occaTag endTag){
    return occaDeviceTimeBetweenTags(device, startTag, endTag);
  }

  void occaDeviceStreamFree_fc(occaDevice device, occaStream stream){
    occaDeviceStreamFree(device, stream);
  }

  void occaDeviceFree_fc(occaDevice device){
    occaDeviceFree(device);
  }
  //====================================


  //---[ Kernel ]-----------------------
  const char* occaKernelMode_fc(occaKernel kernel){ //[-]
    occaKernelMode(kernel);
  }

  int occaKernelPreferredDimSize_fc(occaKernel kernel){
    return occaKernelPreferredDimSize(kernel);
  }

  void occaKernelSetWorkingDims_fc(occaKernel kernel,
                                   int dims,
                                   occaDim items,
                                   occaDim groups){
    occaKernelSetWorkingDims(kernel, dims, items, groups);
  }

  void occaKernelSetAllWorkingDims_fc(occaKernel kernel,
                                      int dims,
                                      uintptr_t itemsX, uintptr_t itemsY, uintptr_t itemsZ,
                                      uintptr_t groupsX, uintptr_t groupsY, uintptr_t groupsZ){
    occaKernelSetAllWorkingDims(kernel,
                                dims,
                                itemsX, itemsY, itemsZ,
                                groupsX, groupsY, groupsZ);
  }

  double occaKernelTimeTaken_fc(occaKernel kernel){
    return occaKernelTimeTaken(kernel);
  }

  occaArgumentList occaGenArgumentList_fc(){
    return occaGenArgumentList();
  }

  void occaArgumentListClear_fc(occaArgumentList list){
    occaArgumentListClear(list);
  }

  void occaArgumentListFree_fc(occaArgumentList list){
    occaArgumentListFree(list);
  }

  void occaArgumentListAddArg_fc(occaArgumentList list,
                                 int argPos,
                                 void *type){
    occaArgumentListAddArg(list, argPos, type);
  }

  void occaKernelRun__fc(occaKernel kernel,
                         occaArgumentList list){
    occaKernelRun_(kernel, list);
  }

  void occaKernelFree_fc(occaKernel kernel){
    occaKernelFree(kernel);
  }

  occaKernelInfo occaGenKernelInfo_fc(){
    return occaGenKernelInfo();
  }

  void occaKernelInfoAddDefine_fc(occaKernelInfo info,
                                  const char *macro OCCA_F2C_LSTR(macro_l),
                                  occaType value
                                  OCCA_F2C_RSTR(macro_l)){
    char *macro_c;
    OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

    occaKernelInfoAddDefine(info, macro_c, value);

    OCCA_F2C_FREE_STR(macro, macro_c);
  }

  void occaKernelInfoFree_fc(occaKernelInfo info){
    occaKernelInfoFree(info);
  }
  //====================================


  //---[ Memory ]-----------------------
  const char* occaMemoryMode_fc(occaMemory memory){ //[-]
    occaMemoryMode(memory);
  }

  void occaCopyMemToMem_fc(occaMemory dest, occaMemory src,
                           const uintptr_t bytes,
                           const uintptr_t destOffset,
                           const uintptr_t srcOffset){
    occaCopyMemToMem(dest, src,
                     bytes, destOffset, srcOffset);
  }

  void occaCopyPtrToMem_fc(occaMemory dest, const void *src,
                           const uintptr_t bytes, const uintptr_t offset){
    occaCopyPtrToMem(dest, src,
                     bytes, offset);
  }

  void occaCopyMemToPtr_fc(void *dest, occaMemory src,
                           const uintptr_t bytes, const uintptr_t offset){
    occaCopyMemToPtr(dest, src,
                     bytes, offset);
  }

  void occaAsyncCopyMemToMem_fc(occaMemory dest, occaMemory src,
                                const uintptr_t bytes,
                                const uintptr_t destOffset,
                                const uintptr_t srcOffset){
    occaAsyncCopyMemToMem(dest, src,
                          bytes, destOffset, srcOffset);
  }

  void occaAsyncCopyPtrToMem_fc(occaMemory dest, const void *src,
                                const uintptr_t bytes, const uintptr_t offset){
    occaAsyncCopyPtrToMem(dest, src,
                          bytes, offset);
  }

  void occaAsyncCopyMemToPtr_fc(void *dest, occaMemory src,
                                const uintptr_t bytes, const uintptr_t offset){
    occaAsyncCopyMemToPtr(dest, src,
                          bytes, offset);
  }

  void occaMemorySwap_fc(occaMemory memoryA, occaMemory memoryB){
    occaMemorySwap(memoryA, memoryB);
  }

  void occaMemoryFree_fc(occaMemory memory){
    occaMemoryFree(memory);
  }
  //====================================

#  ifdef __cplusplus
}
#  endif

#endif
