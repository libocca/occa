#ifndef JOCCA_DEVICE_HEADER
#define JOCCA_DEVICE_HEADER

#include <jni.h>

#include "occa_c.h"

OCCA_START_EXTERN_C

int joccaIsInitialized = 0;

jclass   jdClass  , jkClass  , jkiClass  , jsClass  , jstClass  , jmClass;
jfieldID jdHandleF, jkHandleF, jkiHandleF, jsHandleF, jstHandleF, jmHandleF;

// Keep last line so no warning will occur with the [;]
#define JOCCA_EXTRACT_CLASSES()                                    \
  if(joccaIsInitialized == 0){                                     \
    jdClass    = (*env)->FindClass(env, "occa/Device");            \
    jkClass    = (*env)->FindClass(env, "occa/Kernel");            \
    jkiClass   = (*env)->FindClass(env, "occa/KernelInfo");        \
    jsClass    = (*env)->FindClass(env, "occa/Stream");            \
    jstClass   = (*env)->FindClass(env, "occa/StreamTag");         \
    jmClass    = (*env)->FindClass(env, "occa/Memory");            \
    jdHandleF  = (*env)->GetFieldID(env, jdClass , "handle", "J"); \
    jkHandleF  = (*env)->GetFieldID(env, jkClass , "handle", "J"); \
    jkiHandleF = (*env)->GetFieldID(env, jkiClass, "handle", "J"); \
    jsHandleF  = (*env)->GetFieldID(env, jsClass , "handle", "J"); \
    jstHandleF = (*env)->GetFieldID(env, jstClass, "handle", "J"); \
    jmHandleF  = (*env)->GetFieldID(env, jmClass , "handle", "J"); \
  }                                                                \
  joccaIsInitialized = 1

#define JOCCA_GET_DEVICE(OBJ)                   \
  (*env)->GetLongField(env, OBJ, jdHandleF)
#define JOCCA_GET_KERNEL(OBJ)                   \
  (*env)->GetLongField(env, OBJ, jkHandleF)
#define JOCCA_GET_KERNELINFO(OBJ)               \
  (*env)->GetLongField(env, OBJ, jkiHandleF)
#define JOCCA_GET_STREAM(OBJ)                   \
  (*env)->GetLongField(env, OBJ, jsHandleF)
#define JOCCA_GET_STREAMTAG(OBJ)                \
  (*env)->GetLongField(env, OBJ, jstHandleF)
#define JOCCA_GET_MEMORY(OBJ)                   \
  (*env)->GetLongField(env, OBJ, jmHandleF)

#define JOCCA_SET_DEVICE(OBJ, VALUE)                \
  (*env)->SetLongField(env, OBJ, jdHandleF, VALUE)
#define JOCCA_SET_KERNEL(OBJ, VALUE)                \
  (*env)->SetLongField(env, OBJ, jkHandleF, VALUE)
#define JOCCA_SET_KERNELINFO(OBJ, VALUE)            \
  (*env)->SetLongField(env, OBJ, jkiHandleF, VALUE)
#define JOCCA_SET_STREAM(OBJ, VALUE)                \
  (*env)->SetLongField(env, OBJ, jsHandleF, VALUE)
#define JOCCA_SET_STREAMTAG(OBJ, VALUE)             \
  (*env)->SetLongField(env, OBJ, jstHandleF, VALUE)
#define JOCCA_SET_MEMORY(OBJ, VALUE)                \
  (*env)->SetLongField(env, OBJ, jmHandleF, VALUE)

#define JOCCA_GET_STRING(OBJ)                   \
  (*env)->GetStringUTFChars(env, OBJ, 0)
#define JOCCA_RELEASE_STRING(STR)               \
  (*env)->ReleaseStringUTFChars(env, STR##_, STR)

JNIEXPORT void JNICALL Java_occa_Device_setup(JNIEnv *env,
                                              jobject this_,
                                              jobjectArray infos_){
  JOCCA_EXTRACT_CLASSES();

  const char *infos = JOCCA_GET_STRING(infos_);

  JOCCA_SET_DEVICE(this_, (jlong) occaCreateDevice(infos));

  JOCCA_RELEASE_STRING(infos);
}

JNIEXPORT jlong JNICALL Java_occa_Device_memorySize(JNIEnv *env,
                                                    jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  return (jlong) occaDeviceMemorySize(device);
}

JNIEXPORT jlong JNICALL Java_occa_Device_memoryAllocated(JNIEnv *env,
                                                         jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  return (jlong) occaDeviceMemoryAllocated(device);
}

JNIEXPORT jobjectArray JNICALL Java_occa_Device_mode(JNIEnv *env,
                                                     jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  const char *mode = occaDeviceMode(device);

  return (*env)->NewStringUTF(env, mode);
}

JNIEXPORT void JNICALL Java_occa_Device_setCompiler(JNIEnv *env,
                                                    jobject this_,
                                                    jobjectArray compiler_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  const char *compiler = JOCCA_GET_STRING(compiler_);
  JOCCA_RELEASE_STRING(compiler);
}

JNIEXPORT void JNICALL Java_occa_Device_setCompilerEnvScript(JNIEnv *env,
                                                             jobject this_,
                                                             jobjectArray compilerEnvScript_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  const char *compilerEnvScript = JOCCA_GET_STRING(compilerEnvScript_);
  JOCCA_RELEASE_STRING(compilerEnvScript);
}

JNIEXPORT void JNICALL Java_occa_Device_setCompilerFlags(JNIEnv *env,
                                                         jobject this_,
                                                         jobjectArray compilerFlags_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  const char *compilerFlags = JOCCA_GET_STRING(compilerFlags_);
  JOCCA_RELEASE_STRING(compilerFlags);
}

JNIEXPORT jobjectArray JNICALL Java_occa_Device_getCompiler(JNIEnv *env,
                                                            jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobjectArray JNICALL Java_occa_Device_getCompilerEnvScript(JNIEnv *env,
                                                                     jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobjectArray JNICALL Java_occa_Device_getCompilerFlags(JNIEnv *env,
                                                                 jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT void JNICALL Java_occa_Device_flush(JNIEnv *env,
                                              jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT void JNICALL Java_occa_Device_finish(JNIEnv *env,
                                               jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT void JNICALL Java_occa_Device_waitFor(JNIEnv *env,
                                                jobject this_,
                                                jobject tag){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobject JNICALL Java_occa_Device_createStream(JNIEnv *env,
                                                        jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobject JNICALL Java_occa_Device_getStream(JNIEnv *env,
                                                     jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT void JNICALL Java_occa_Device_setStream(JNIEnv *env,
                                                  jobject this_,
                                                  jobject stream){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobject JNICALL Java_occa_Device_tagStream(JNIEnv *env,
                                                     jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jdouble JNICALL Java_occa_Device_timeBetween(JNIEnv *env,
                                                       jobject this_,
                                                       jobject startTag,
                                                       jobject endTag){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobject JNICALL Java_occa_Device_buildKernel(JNIEnv *env,
                                                       jobject this_,
                                                       jobjectArray str_,
                                                       jobjectArray functionName_,
                                                       jobject info){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  const char *str          = JOCCA_GET_STRING(str_);
  const char *functionName = JOCCA_GET_STRING(functionName_);

  JOCCA_RELEASE_STRING(str);
  JOCCA_RELEASE_STRING(functionName);
}

JNIEXPORT jobject JNICALL Java_occa_Device_buildKernelFromString(JNIEnv *env,
                                                                 jobject this_,
                                                                 jobjectArray content_,
                                                                 jobjectArray functionName_,
                                                                 jobject info,
                                                                 jint language){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  const char *content      = JOCCA_GET_STRING(content_);
  const char *functionName = JOCCA_GET_STRING(functionName_);

  JOCCA_RELEASE_STRING(content);
  JOCCA_RELEASE_STRING(functionName);
}

JNIEXPORT jobject JNICALL Java_occa_Device_buildKernelFromSource(JNIEnv *env,
                                                                 jobject this_,
                                                                 jobjectArray filename_,
                                                                 jobjectArray functionName_,
                                                                 jobject info){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  const char *filename      = JOCCA_GET_STRING(filename_);
  const char *functionName = JOCCA_GET_STRING(functionName_);

  JOCCA_RELEASE_STRING(filename);
  JOCCA_RELEASE_STRING(functionName);
}

JNIEXPORT jobject JNICALL Java_occa_Device_buildKernelFromBinary(JNIEnv *env,
                                                                 jobject this_,
                                                                 jobjectArray filename_,
                                                                 jobjectArray functionName_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);

  const char *filename      = JOCCA_GET_STRING(filename_);
  const char *functionName = JOCCA_GET_STRING(functionName_);

  JOCCA_RELEASE_STRING(filename);
  JOCCA_RELEASE_STRING(functionName);
}

JNIEXPORT jobject JNICALL Java_occa_Device_wrapMemory(JNIEnv *env,
                                                      jobject this_,
                                                      jobjectArray handle,
                                                      jlong bytes){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT void JNICALL Java_occa_Device_wrapManagedMemory(JNIEnv *env,
                                                          jobject this_,
                                                          jobjectArray handle,
                                                          jlong bytes){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobject JNICALL Java_occa_Device_malloc(JNIEnv *env,
                                                  jobject this_,
                                                  jlong bytes,
                                                  jobjectArray src){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobjectArray JNICALL Java_occa_Device_managedAlloc(JNIEnv *env,
                                                             jobject this_,
                                                             jlong bytes,
                                                             jobjectArray src){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobject JNICALL Java_occa_Device_mappedAlloc(JNIEnv *env,
                                                       jobject this_,
                                                       jlong bytes,
                                                       jobjectArray src){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT jobjectArray JNICALL Java_occa_Device_managedMappedAlloc(JNIEnv *env,
                                                                   jobject this_,
                                                                   jlong byts,
                                                                   jobjectArray src){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

JNIEXPORT void JNICALL Java_occa_Device_free(JNIEnv *env,
                                             jobject this_){
  JOCCA_EXTRACT_CLASSES();

  occaDevice device = (occaDevice) JOCCA_GET_DEVICE(this_);
}

OCCA_END_EXTERN_C

#endif
