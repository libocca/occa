#include <occa/internal/utils/string.hpp>
#include <occa/internal/modes/serial/registration.hpp>
#include <occa/internal/utils/misc.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace serial {
    serialMode::serialMode() :
        mode_t("Serial") {}

    bool serialMode::init() {
      return true;
    }

    styling::section& serialMode::getDescription() {
      static styling::section section("CPU(s)");
      if (section.size() != 0) {
        return section;
      }

      sys::SystemInfo info = sys::SystemInfo::load();

      const std::string simdWidth = toString(32*OCCA_SIMD_WIDTH) + " bits";

      std::string l1d = stringifyBytes(info.processor.cache.l1d);
      std::string l1i = stringifyBytes(info.processor.cache.l1i);
      std::string l2  = stringifyBytes(info.processor.cache.l2);
      std::string l3  = stringifyBytes(info.processor.cache.l3);

      const size_t cacheWidth = max({
          l1d.size(),
          l1i.size(),
          l2.size(),
          l3.size()
        });

      if (cacheWidth) {
        l1d = styling::right(l1d, cacheWidth);
        l1i = styling::right(l1i, cacheWidth);
        l2  = styling::right(l2, cacheWidth);
        l3  = styling::right(l3, cacheWidth);
      }

      if (info.processor.name.size()) {
        section.add("Processor Name", info.processor.name);
      }
      if (info.processor.coreCount) {
        section.add("Cores", toString(info.processor.coreCount));
      }
      if (info.memory.total) {
        section.add("Memory", stringifyBytes(info.memory.total));
      }
      if (info.processor.frequency) {
        section.add("Clock Frequency", stringifyFrequency(info.processor.frequency));
      }
      section
        .add("SIMD Instruction Set", OCCA_VECTOR_SET)
        .add("SIMD Width", simdWidth);
      if (l1d.size()) {
        section.add("L1d Cache Size", l1d);
      }
      if (l1i.size()) {
        section.add("L1i Cache Size", l1i);
      }
      if (l2.size()) {
        section.add("L2 Cache Size", l2);
      }
      if (l3.size()) {
        section.add("L3 Cache Size", l3);
      }
      return section;
    }

    modeDevice_t* serialMode::newDevice(const occa::json &props) {
      return new device(setModeProp(props));
    }

    int serialMode::getDeviceCount(const occa::json &props) {
      return 1;
    }

    serialMode mode;
  }
}
