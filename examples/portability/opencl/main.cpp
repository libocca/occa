#include <iostream>

#include "occa.hpp"
// Has OpenCL includes
#include "occa/modes/opencl/utils.hpp"

int main(int argc, char **argv) {
  int entries = 5;

  //---[ Init OpenCL ]------------------
  cl_int error;

  cl_device_id clDeviceID = occa::opencl::deviceID(0,0);

  cl_context clContext = clCreateContext(NULL,
                                         1, &clDeviceID,
                                         NULL, NULL, &error);
  OCCA_OPENCL_ERROR("Device: Creating Context", error);

  cl_command_queue clStream = clCreateCommandQueue(clContext,
                                                   clDeviceID,
                                                   CL_QUEUE_PROFILING_ENABLE, &error);
  OCCA_OPENCL_ERROR("Device: createStream", error);

  cl_mem cl_a = clCreateBuffer(clContext,
                               CL_MEM_READ_WRITE,
                               entries*sizeof(float), NULL, &error);

  cl_mem cl_b = clCreateBuffer(clContext,
                               CL_MEM_READ_WRITE,
                               entries*sizeof(float), NULL, &error);

  cl_mem cl_ab = clCreateBuffer(clContext,
                                CL_MEM_READ_WRITE,
                                entries*sizeof(float), NULL, &error);
  //====================================

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  occa::device device = occa::opencl::wrapDevice(clDeviceID,
                                                 clContext);

  occa::stream stream = device.wrapStream(&clStream);
  device.setStream(stream);

  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  o_a  = occa::opencl::wrapMemory(device, cl_a , entries*sizeof(float));
  o_b  = occa::opencl::wrapMemory(device, cl_b , entries*sizeof(float));
  o_ab = occa::opencl::wrapMemory(device, cl_ab, entries*sizeof(float));

  addVectors = device.buildKernel("addVectors.okl",
                                  "addVectors");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for (int i = 0; i < 5; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}
