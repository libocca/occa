#include <occa/modes/serial/registration.hpp>

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

      std::stringstream ss;

      const std::string processorName = sys::getProcessorName();

      ss << sys::getCoreCount();
      const std::string coreCount = ss.str();
      ss.str("");

      udim_t ram = sys::installedRAM();
      const std::string ramStr = stringifyBytes(ram);

      const int freq = sys::getProcessorFrequency();
      if (freq < 1000) {
        ss << freq << " MHz";
      } else {
        ss << (freq/1000.0) << " GHz";
      }
      const std::string clockFrequency = ss.str();
      ss.str("");

      ss << (32*OCCA_SIMD_WIDTH) << " bits";
      const std::string simdWidth = ss.str();
      ss.str("");

      std::string l1 = sys::getProcessorCacheSize(1);
      std::string l2 = sys::getProcessorCacheSize(2);
      std::string l3 = sys::getProcessorCacheSize(3);

      const size_t maxSize = std::max(std::max(l1.size(), l2.size()), l3.size());
      if (maxSize) {
        l1 = styling::right(l1, maxSize);
        l2 = styling::right(l2, maxSize);
        l3 = styling::right(l3, maxSize);
      }

      if (processorName.size()) {
        section.add("Processor Name", processorName);
      }
      if (coreCount.size()) {
        section.add("Cores", coreCount);
      }
      if (ramStr.size()) {
        section.add("Memory (RAM)", ramStr);
      }
      if (clockFrequency.size()) {
        section.add("Clock Frequency", clockFrequency);
      }
      section
        .add("SIMD Instruction Set", OCCA_VECTOR_SET)
        .add("SIMD Width", simdWidth);
      if (l1.size()) {
        section.add("L1 Cache Size (d)", l1);
      }
      if (l2.size()) {
        section.add("L2 Cache Size", l2);
      }
      if (l3.size()) {
        section.add("L3 Cache Size", l3);
      }
      return section;
    }

    modeDevice_t* serialMode::newDevice(const occa::properties &props) {
      return new device(setModeProp(props));
    }

    serialMode mode;
  }
}
