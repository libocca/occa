#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_OPENCL_POLYFILL_HEADER
#define OCCA_INTERNAL_MODES_OPENCL_POLYFILL_HEADER

#if OCCA_OPENCL_ENABLED

// Remove warnings due to CL_TARGET_OPENCL_VERSION not being set
#  ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION 220
#  endif

#  if   (OCCA_OS & OCCA_LINUX_OS)
#    include <CL/cl.h>
#    include <CL/cl_gl.h>
#  elif (OCCA_OS & OCCA_MACOS_OS)
#    include <OpenCL/OpenCl.h>
#  else
#    include "CL/opencl.h"
#  endif

#else

// Wrap in the occa namespace so as long as we don't use ::cl_device_id, the two
//   - cl_device_id
//   - occa::cl_device_id
// are indistinguisable inside the occa namespace
namespace occa {
  //---[ Types ]------------------------
  typedef bool          cl_bool;
  typedef int           cl_int;
  typedef unsigned int  cl_uint;
  typedef unsigned long cl_ulong;

  typedef int cl_buffer_create_type;
  typedef int cl_command_queue_properties;
  typedef int cl_platform_info;
  typedef int cl_device_info;
  typedef int cl_device_type;
  typedef int cl_kernel_work_group_info;
  typedef int cl_map_flags;
  typedef int cl_mem_flags;
  typedef int cl_profiling_info;
  typedef int cl_program_build_info;
  typedef int cl_program_info;

  typedef struct _cl_command_queue*      cl_command_queue;
  typedef struct _cl_context*            cl_context;
  typedef struct _cl_context_properties* cl_context_properties;
  typedef struct _cl_device_id*          cl_device_id;
  typedef struct _cl_event*              cl_event;
  typedef struct _cl_kernel*             cl_kernel;
  typedef struct _cl_mem*                cl_mem;
  typedef struct _cl_platform_id*        cl_platform_id;
  typedef struct _cl_program*            cl_program;
  typedef struct _cl_queue_properties*   cl_queue_properties;

  typedef struct _cl_buffer_region {
    size_t origin;
    size_t size;
  } cl_buffer_region;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

  static cl_bool CL_FALSE = false;
  static cl_bool CL_TRUE = true;

  static cl_buffer_create_type CL_BUFFER_CREATE_TYPE_REGION = 0;

  static cl_command_queue_properties CL_QUEUE_PROFILING_ENABLE = 0;

  static cl_platform_info CL_PLATFORM_NAME = 0;
  static cl_platform_info CL_PLATFORM_VENDOR = 1;
  static cl_platform_info CL_PLATFORM_VERSION = 2;

  static cl_device_info CL_DEVICE_GLOBAL_MEM_SIZE = 0;
  static cl_device_info CL_DEVICE_MAX_COMPUTE_UNITS = 1;
  static cl_device_info CL_DEVICE_NAME = 2;
  static cl_device_info CL_DEVICE_TYPE = 3;
  static cl_device_info CL_DEVICE_VENDOR = 4;
  static cl_device_info CL_DEVICE_VERSION = 5;
  static cl_device_info CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 6;
  static cl_device_info CL_DEVICE_MAX_WORK_ITEM_SIZES = 7;

  static cl_device_type CL_DEVICE_TYPE_ACCELERATOR = 0;
  static cl_device_type CL_DEVICE_TYPE_CPU = 1;
  static cl_device_type CL_DEVICE_TYPE_GPU = 2;
  static cl_device_type CL_DEVICE_TYPE_ALL = 3;
  static cl_device_type CL_DEVICE_TYPE_DEFAULT = 4;

  static cl_kernel_work_group_info CL_KERNEL_WORK_GROUP_SIZE = 0;

  static cl_map_flags CL_MAP_READ = 0;
  static cl_map_flags CL_MAP_WRITE = 1;

  static cl_mem_flags CL_MEM_READ_WRITE = 0;
  static cl_mem_flags CL_MEM_COPY_HOST_PTR = 1;
  static cl_mem_flags CL_MEM_ALLOC_HOST_PTR = 2;

  static cl_profiling_info CL_PROFILING_COMMAND_END = 0;
  static cl_profiling_info CL_PROFILING_COMMAND_START = 1;

  static cl_program_build_info CL_PROGRAM_BUILD_LOG = 0;

  static cl_program_info CL_PROGRAM_BINARIES = 0;
  static cl_program_info CL_PROGRAM_BINARY_SIZES = 1;

#pragma GCC diagnostic pop

  //---[ Enums ]------------------------
  enum cl_result_enum {
    CL_SUCCESS = 0,
    CL_DEVICE_NOT_FOUND,
    CL_DEVICE_NOT_AVAILABLE,
    CL_COMPILER_NOT_AVAILABLE,
    CL_MEM_OBJECT_ALLOCATION_FAILURE,
    CL_OUT_OF_RESOURCES,
    CL_OUT_OF_HOST_MEMORY,
    CL_PROFILING_INFO_NOT_AVAILABLE,
    CL_MEM_COPY_OVERLAP,
    CL_IMAGE_FORMAT_MISMATCH,
    CL_IMAGE_FORMAT_NOT_SUPPORTED,
    CL_BUILD_PROGRAM_FAILURE,
    CL_MAP_FAILURE,
    CL_MISALIGNED_SUB_BUFFER_OFFSET,
    CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
    CL_INVALID_VALUE,
    CL_INVALID_DEVICE_TYPE,
    CL_INVALID_PLATFORM,
    CL_INVALID_DEVICE,
    CL_INVALID_CONTEXT,
    CL_INVALID_QUEUE_PROPERTIES,
    CL_INVALID_COMMAND_QUEUE,
    CL_INVALID_HOST_PTR,
    CL_INVALID_MEM_OBJECT,
    CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
    CL_INVALID_IMAGE_SIZE,
    CL_INVALID_SAMPLER,
    CL_INVALID_BINARY,
    CL_INVALID_BUILD_OPTIONS,
    CL_INVALID_PROGRAM,
    CL_INVALID_PROGRAM_EXECUTABLE,
    CL_INVALID_KERNEL_NAME,
    CL_INVALID_KERNEL_DEFINITION,
    CL_INVALID_KERNEL,
    CL_INVALID_ARG_INDEX,
    CL_INVALID_ARG_VALUE,
    CL_INVALID_ARG_SIZE,
    CL_INVALID_KERNEL_ARGS,
    CL_INVALID_WORK_DIMENSION,
    CL_INVALID_WORK_GROUP_SIZE,
    CL_INVALID_WORK_ITEM_SIZE,
    CL_INVALID_GLOBAL_OFFSET,
    CL_INVALID_EVENT_WAIT_LIST,
    CL_INVALID_EVENT,
    CL_INVALID_OPERATION,
    CL_INVALID_GL_OBJECT,
    CL_INVALID_BUFFER_SIZE,
    CL_INVALID_MIP_LEVEL,
    CL_INVALID_GLOBAL_WORK_SIZE,
    CL_INVALID_PROPERTY,
    OCCA_OPENCL_IS_NOT_ENABLED
  };

  //---[ Methods ]----------------------
  //   ---[ Context ]-------------------
  inline cl_context clCreateContext(cl_context_properties *props,
                                    cl_uint num_devices,
                                    const cl_device_id *devices,
                                    void *pfn_notify,
                                    void *user_data,
                                    cl_int *errcode_ret) {
    *errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  inline cl_int clFinish(cl_command_queue command_queue) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clReleaseContext(cl_context context) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Device ]--------------------
  inline cl_int clGetDeviceIDs(cl_platform_id platform,
                               cl_device_type device_type,
                               cl_uint num_entries,
                               cl_device_id *devices,
                               cl_uint *num_devices) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clGetDeviceInfo(cl_device_id device,
                                cl_device_info param_name,
                                size_t param_value_size,
                                void *param_value,
                                size_t *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }
  
  //   ---[ Platform ]--------------------

  inline cl_int clGetPlatformIDs(cl_uint num_entries,
                                 cl_platform_id *platforms,
                                 cl_uint *num_platforms) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clGetPlatformInfo(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret) {
      
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Event ]---------------------
  inline cl_int clEnqueueMarker(cl_command_queue command_queue,
                                cl_event *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clEnqueueMarkerWithWaitList(cl_command_queue  command_queue ,
                                            cl_uint  num_events_in_wait_list ,
                                            const cl_event  *event_wait_list ,
                                            cl_event  *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clEnqueueBarrier(cl_command_queue command_queue,
                                cl_event *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clEnqueueBarrierWithWaitList(cl_command_queue  command_queue ,
                                            cl_uint  num_events_in_wait_list ,
                                            const cl_event  *event_wait_list ,
                                            cl_event  *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clGetEventProfilingInfo(cl_event event,
                                        cl_profiling_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clReleaseEvent(cl_event event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clWaitForEvents(cl_uint num_events,
                                const cl_event *event_list) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Memory ]--------------------
  inline cl_mem clCreateBuffer(cl_context context,
                               cl_mem_flags flags,
                               size_t size,
                               void *host_ptr,
                               cl_int *errcode_ret) {
    *errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  inline cl_mem clCreateSubBuffer(cl_mem buffer,
                                  cl_mem_flags flags,
                                  cl_buffer_create_type buffer_create_type,
                                  const void *buffer_create_info,
                                  cl_int *errcode_ret) {
    *errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  inline cl_int clEnqueueCopyBuffer(cl_command_queue command_queue,
                                    cl_mem src_buffer,
                                    cl_mem dst_buffer,
                                    size_t src_offset,
                                    size_t dst_offset,
                                    size_t cb,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }


  inline void * clEnqueueMapBuffer(cl_command_queue command_queue,
                                   cl_mem buffer,
                                   cl_bool blocking_map,
                                   cl_map_flags map_flags,
                                   size_t offset,
                                   size_t cb,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event,
                                   cl_int *errcode_ret) {
    *errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  inline cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
                                    cl_mem buffer,
                                    cl_bool blocking_read,
                                    size_t offset,
                                    size_t cb,
                                    void *ptr,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue,
                                        cl_mem memobj,
                                        void *mapped_ptr,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event *event_wait_list,
                                        cl_event *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
                                     cl_mem buffer,
                                     cl_bool blocking_write,
                                     size_t offset,
                                     size_t cb,
                                     const void *ptr,
                                     cl_uint num_events_in_wait_list,
                                     const cl_event *event_wait_list,
                                     cl_event *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clReleaseMemObject(cl_mem memobj) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Kernel ]--------------------
  inline cl_int clBuildProgram(cl_program program,
                               cl_uint num_devices,
                               const cl_device_id *device_list,
                               const char *options,
                               void *pfn_notify,
                               void *user_data) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_program clCreateProgramWithBinary(cl_context context,
                                              cl_uint num_devices,
                                              const cl_device_id *device_list,
                                              const size_t *lengths,
                                              const unsigned char **binaries,
                                              cl_int *binary_status,
                                              cl_int *errcode_ret) {
    *errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }


  inline cl_program clCreateProgramWithSource(cl_context context,
                                              cl_uint count,
                                              const char **strings,
                                              const size_t *lengths,
                                              cl_int *errcode_ret) {
    *errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  inline cl_kernel clCreateKernel(cl_program  program,
                                  const char *kernel_name,
                                  cl_int *errcode_ret) {
    *errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  inline cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
                                       cl_kernel kernel,
                                       cl_uint work_dim,
                                       const size_t *global_work_offset,
                                       const size_t *global_work_size,
                                       const size_t *local_work_size,
                                       cl_uint num_events_in_wait_list,
                                       const cl_event *event_wait_list,
                                       cl_event *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clReleaseKernel(cl_kernel kernel) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clGetProgramBuildInfo(cl_program  program,
                                      cl_device_id  device,
                                      cl_program_build_info  param_name,
                                      size_t  param_value_size,
                                      void  *param_value,
                                      size_t  *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clGetProgramInfo(cl_program program,
                                 cl_program_info param_name,
                                 size_t param_value_size,
                                 void *param_value,
                                 size_t *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clGetKernelWorkGroupInfo(cl_kernel kernel,
                                         cl_device_id device,
                                         cl_kernel_work_group_info param_name,
                                         size_t param_value_size,
                                         void *param_value,
                                         size_t *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clSetKernelArg(cl_kernel kernel,
                               cl_uint arg_index,
                               size_t arg_size,
                               const void *arg_value) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Stream ]--------------------

  inline cl_command_queue clCreateCommandQueue(cl_context context,
                                               cl_device_id device,
                                               cl_command_queue_properties props,
                                               cl_int *errcode_ret) {
    *errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  inline cl_command_queue clCreateCommandQueueWithProperties(cl_context context,
                                                             cl_device_id device,
                                                             const cl_queue_properties *props,
                                                             cl_int *errcode_ret) {
    *errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  inline cl_int clRetainCommandQueue(cl_command_queue command_queue) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  inline cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

}

#endif
#endif
