#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#import <Metal/Metal.h>

#include <occa/internal/api/metal/buffer.hpp>
#include <occa/internal/api/metal/device.hpp>
#include <occa/internal/api/metal/function.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/internal/modes/metal/memory.hpp>

namespace occa {
  namespace api {
    namespace metal {
      function_t::function_t() :
        device(NULL),
        libraryObj(NULL),
        functionObj(NULL),
        pipelineStateObj(NULL) {}

      function_t::function_t(device_t *device_,
                             void *libraryObj_,
                             void *functionObj_) :
        device(device_),
        libraryObj(libraryObj_),
        functionObj(functionObj_),
        pipelineStateObj(NULL) {

        id<MTLFunction> metalFunction = (__bridge id<MTLFunction>) functionObj;
        id<MTLDevice> metalDevice = (__bridge id<MTLDevice>) device->deviceObj;

        NSError* error = nil;
        id<MTLComputePipelineState> metalPipelineState = [
          metalDevice newComputePipelineStateWithFunction:metalFunction
                      error:&error
        ];
        if (metalPipelineState) {
          pipelineStateObj = (__bridge void*) metalPipelineState;
        } else {
          if (error) {
            std::string errorStr = [error.localizedDescription UTF8String];
            OCCA_FORCE_ERROR("Kernel: Unable to create compute pipeline."
                             << " Error: " << errorStr);
          } else {
            OCCA_FORCE_ERROR("Kernel: Unable to create compute pipeline");
          }
        }
      }

      function_t::function_t(const function_t &other) :
        device(other.device),
        libraryObj(other.libraryObj),
        functionObj(other.functionObj),
        pipelineStateObj(other.pipelineStateObj) {}

      function_t& function_t::operator = (const function_t &other) {
        device = other.device;
        libraryObj = other.libraryObj;
        functionObj = other.functionObj;
        pipelineStateObj = other.pipelineStateObj;
        return *this;
      }

      void function_t::free() {
        // Remove reference counts
        if (libraryObj) {
          id<MTLLibrary> metalLibrary = (__bridge id<MTLLibrary>) libraryObj;
          metalLibrary = nil;
          libraryObj = NULL;
        }
        if (functionObj) {
          id<MTLFunction> metalFunction = (__bridge id<MTLFunction>) functionObj;
          metalFunction = nil;
          functionObj = NULL;
        }
        if (pipelineStateObj) {
          id<MTLComputePipelineState> metalPipelineState = (
            (__bridge id<MTLComputePipelineState>) pipelineStateObj
          );
          metalPipelineState = nil;
          pipelineStateObj = NULL;
        }
      }

      void function_t::run(commandQueue_t &commandQueue,
                           occa::dim outerDims,
                           occa::dim innerDims,
                           const std::vector<kernelArgData> &arguments) {
        id<MTLCommandQueue> metalCommandQueue = (
          (__bridge id<MTLCommandQueue>) commandQueue.commandQueueObj
        );
        id<MTLComputePipelineState> metalPipelineState = (
          (__bridge id<MTLComputePipelineState>) pipelineStateObj
        );

        // Initialize Metal command
        id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
        OCCA_ERROR("Kernel: Create command buffer",
                   commandBuffer != nil);

        // The commandBuffer callback has to be set before commit is called on it
        commandQueue.setLastCommandBuffer((__bridge void*) commandBuffer);

        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        OCCA_ERROR("Kernel: Create compute command encoder",
                   computeEncoder != nil);

        // Start encoding the kernel
        [computeEncoder setComputePipelineState:metalPipelineState];

        // Add arguments
        const int argCount = (int) arguments.size();
        for (int index = 0; index < argCount; ++index) {
          const kernelArgData &arg = arguments[index];
          if (arg.modeMemory) {
            occa::metal::memory &memory = (
              *(dynamic_cast<occa::metal::memory*>(arg.getModeMemory()))
            );
            id<MTLBuffer> metalBuffer = (
              (__bridge id<MTLBuffer>) memory.getMetalBuffer().bufferObj
            );
            [computeEncoder setBuffer:metalBuffer
                               offset:memory.getOffset()
                              atIndex:index];
          } else {
            [computeEncoder setBytes:arg.ptr()
                              length:arg.size
                             atIndex:index];
          }
        }

        // Set the loop dimensions
        dim fullDims = outerDims * innerDims;
        MTLSize fullSize = MTLSizeMake(fullDims.x, fullDims.y, fullDims.z);
        MTLSize innerSize = MTLSizeMake(innerDims.x, innerDims.y, innerDims.z);

        [computeEncoder dispatchThreads:fullSize
                  threadsPerThreadgroup:innerSize];

        // Finish encoding and start executing the kernel
        [computeEncoder endEncoding];
        [commandBuffer commit];
      }
    }
  }
}

#endif
