#include <occa/defines.hpp>

#ifndef OCCA_MODES_OPENCL_POLYFILL_HEADER
#define OCCA_MODES_OPENCL_POLYFILL_HEADER

#if OCCA_OPENCL_ENABLED

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
  typedef int           cl_int;
  typedef unsigned int  cl_uint;
  typedef unsigned long cl_ulong;

  typedef struct _cl_buffer_region*      cl_buffer_region;
  typedef struct _cl_command_queue*      cl_command_queue;
  typedef struct _cl_context*            cl_context;
  typedef struct _cl_context_properties* cl_context_properties;
  typedef struct _cl_device_id*          cl_device_id;
  typedef struct _cl_device_info*        cl_device_info;
  typedef struct _cl_device_type*        cl_device_type;
  typedef struct _cl_event*              cl_event;
  typedef struct _cl_kernel*             cl_kernel;
  typedef struct _cl_mem*                cl_mem;
  typedef struct _cl_platform_id*        cl_platform_id;
  typedef struct _cl_program*            cl_program;
  typedef struct _cl_queue_properties*   cl_queue_properties;

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
  cl_context clCreateContext(cl_context_properties *properties,
                             cl_uint num_devices,
                             const cl_device_id *devices,
                             void *pfn_notify,
                             void *user_data,
                             cl_int *errcode_ret) {
    errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  cl_int clFinish(cl_command_queue command_queue) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clReleaseContext(cl_context context) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Device ]--------------------
  cl_int clGetDeviceIDs(cl_platform_id platform,
                        cl_device_type device_type,
                        cl_uint num_entries,
                        cl_device_id *devices,
                        cl_uint *num_devices) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clGetDeviceInfo(cl_device_id device,
                         cl_device_info param_name,
                         size_t param_value_size,
                         void *param_value,
                         size_t *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clGetPlatformIDs(cl_uint num_entries,
                          cl_platform_id *platforms,
                          cl_uint *num_platforms) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Event ]---------------------
  cl_int clEnqueueMarker(cl_command_queue command_queue,
                         cl_event *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clEnqueueMarkerWithWaitList(cl_command_queue  command_queue ,
                                     cl_uint  num_events_in_wait_list ,
                                     const cl_event  *event_wait_list ,
                                     cl_event  *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clGetEventProfilingInfo(cl_event event,
                                 cl_profiling_info param_name,
                                 size_t param_value_size,
                                 void *param_value,
                                 size_t *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clReleaseEvent(cl_event event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clWaitForEvents(cl_uint num_events,
                         const cl_event *event_list) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Memory ]--------------------

  cl_mem clCreateBuffer(cl_context context,
                        cl_mem_flags flags,
                        size_t size,
                        void *host_ptr,
                        cl_int *errcode_ret) {
    errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  cl_mem clCreateSubBuffer(cl_mem buffer,
                           cl_mem_flags flags,
                           cl_buffer_create_type buffer_create_type,
                           const void *buffer_create_info,
                           cl_int *errcode_ret) {
    errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  cl_int clEnqueueCopyBuffer(cl_command_queue command_queue,
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


  void * clEnqueueMapBuffer(cl_command_queue command_queue,
                            cl_mem buffer,
                            cl_bool blocking_map,
                            cl_map_flags map_flags,
                            size_t offset,
                            size_t cb,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event,
                            cl_int *errcode_ret) {
    errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
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

  cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue,
                                 cl_mem memobj,
                                 void *mapped_ptr,
                                 cl_uint num_events_in_wait_list,
                                 const cl_event *event_wait_list,
                                 cl_event *event) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
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

  cl_int clReleaseMemObject(cl_mem memobj) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Kernel ]--------------------
  cl_int clBuildProgram(cl_program program,
                        cl_uint num_devices,
                        const cl_device_id *device_list,
                        const char *options,
                        void *pfn_notify,
                        void *user_data) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_program clCreateProgramWithBinary(cl_context context,
                                       cl_uint num_devices,
                                       const cl_device_id *device_list,
                                       const size_t *lengths,
                                       const unsigned char **binaries,
                                       cl_int *binary_status,
                                       cl_int *errcode_ret) {
    errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }


  cl_program clCreateProgramWithSource(cl_context context,
                                       cl_uint count,
                                       const char **strings,
                                       const size_t *lengths,
                                       cl_int *errcode_ret) {
    errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  cl_kernel clCreateKernel(cl_program  program,
                           const char *kernel_name,
                           cl_int *errcode_ret) {
    errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
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

  cl_int clReleaseKernel(cl_kernel kernel) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clGetProgramBuildInfo(cl_program  program,
                               cl_device_id  device,
                               cl_program_build_info  param_name,
                               size_t  param_value_size,
                               void  *param_value,
                               size_t  *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clGetProgramInfo(cl_program program,
                          cl_program_info param_name,
                          size_t param_value_size,
                          void *param_value,
                          size_t *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clGetKernelWorkGroupInfo(cl_kernel kernel,
                                  cl_device_id device,
                                  cl_kernel_work_group_info param_name,
                                  size_t param_value_size,
                                  void *param_value,
                                  size_t *param_value_size_ret) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  cl_int clSetKernelArg(cl_kernel kernel,
                        cl_uint arg_index,
                        size_t arg_size,
                        const void *arg_value) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

  //   ---[ Stream ]--------------------

  cl_command_queue clCreateCommandQueue(cl_context context,
                                        cl_device_id device,
                                        cl_command_queue_properties properties,
                                        cl_int *errcode_ret) {
    errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  cl_command_queue clCreateCommandQueueWithProperties(cl_context context,
                                                      cl_device_id device,
                                                      const cl_queue_properties *properties,
                                                      cl_int *errcode_ret) {
    errcode_ret = OCCA_OPENCL_IS_NOT_ENABLED;
    return NULL;
  }

  cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    return OCCA_OPENCL_IS_NOT_ENABLED;
  }

}

#endif
#endif
