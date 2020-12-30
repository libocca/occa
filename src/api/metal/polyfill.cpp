#include <occa/defines.hpp>

#if !OCCA_METAL_ENABLED

#include <occa/internal/api/metal.hpp>

namespace occa {
  namespace api {
    namespace metal {
      int getDeviceCount() {
        return 0;
      }

      device_t getDevice(const int deviceId) {
        return device_t();
      }

      //---[ Event ]--------------------
      event_t::event_t() {}

      event_t::event_t(commandQueue_t *commandQueue_,
                       void *eventObj_,
                       const int eventId_,
                       void *commandBufferObj_) {}

      event_t::event_t(const event_t &other) {}

      event_t& event_t::operator = (const event_t &other) {
        return *this;
      }

      void event_t::free() {}
      void event_t::freeCommandBuffer() {}

      void event_t::waitUntilCompleted() {}

      void event_t::setTime(const double eventTime_) {}

      double event_t::getTime() const {
        return 0;
      }

      //---[ Buffer ]-------------------
      buffer_t::buffer_t(void *bufferObj_) {}

      buffer_t::buffer_t(const buffer_t &other) {}

      buffer_t& buffer_t::operator = (const buffer_t &other) {
        return *this;
      }

      void buffer_t::free() {}

      void* buffer_t::getPtr() const {
        return (void*) 0;
      }

      //---[ Command Queue ]------------
      commandQueue_t::commandQueue_t() {}

      commandQueue_t::commandQueue_t(device_t *device_,
                                     void *commandQueueObj_) {}

      commandQueue_t::commandQueue_t(const commandQueue_t &other) {}

      commandQueue_t& commandQueue_t::operator = (const commandQueue_t &other) {
        return *this;
      }

      void commandQueue_t::free() {}
      void commandQueue_t::freeLastCommandBuffer() {}

      event_t commandQueue_t::createEvent() const {
        return event_t();
      }

      void commandQueue_t::clearCommandBuffer(void *commandBufferObj) {}
      void commandQueue_t::setLastCommandBuffer(void *commandBufferObj) {}

      void commandQueue_t::processEvents(const int eventId) {}

      void commandQueue_t::finish() {}

      void commandQueue_t::memcpy(buffer_t &dest,
                                  const udim_t destOffset,
                                  const buffer_t &src,
                                  const udim_t srcOffset,
                                  const udim_t bytes,
                                  const bool async) {}

      void commandQueue_t::memcpy(void *dest,
                                  const buffer_t &src,
                                  const udim_t srcOffset,
                                  const udim_t bytes,
                                  const bool async) {}

      void commandQueue_t::memcpy(buffer_t &dest,
                                  const udim_t destOffset,
                                  const void *src,
                                  const udim_t bytes,
                                  const bool async) {}

      //---[ Kernel ]-------------------
      function_t::function_t() {}

      function_t::function_t(device_t *device_,
                             void *libraryObj_,
                             void *functionObj_) {}

      function_t::function_t(const function_t &other) {}

      function_t& function_t::operator = (const function_t &other) {
        return *this;
      }

      void function_t::run(commandQueue_t &commandQueue,
                           occa::dim outerDims,
                           occa::dim innerDims,
                           const std::vector<kernelArgData> &arguments) {}

      void function_t::free() {}

      //---[ Device ]-------------------
      device_t::device_t(void *deviceObj_) {}

      device_t::device_t(const device_t &other) {}

      device_t& device_t::operator = (const device_t &other) {
        return *this;
      }

      void device_t::free() {}

      std::string device_t::getName() const {
        return "";
      }

      udim_t device_t::getMemorySize() const {
        return 0;
      }

      dim device_t::getMaxOuterDims() const {
        return dim();
      }

      dim device_t::getMaxInnerDims() const {
        return dim();
      }

      commandQueue_t device_t::createCommandQueue() const {
        return commandQueue_t();
      }

      function_t device_t::buildKernel(const std::string &metallibFilename,
                                       const std::string &kernelName,
                                       io::lock_t &lock) const {
        return function_t();
      }

      buffer_t device_t::malloc(const udim_t bytes,
                                const void *src) const {
        return buffer_t();
      }
    }
  }
}

#endif
