/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa/modes/serial/registration.hpp"

namespace occa {
  namespace serial {
    modeInfo::modeInfo() {}

    void modeInfo::init() {}

    styling::section& modeInfo::getDescription() {
      static styling::section section("CPU Info");
      if (section.size() == 0) {
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
      }
      return section;
    }

    occa::properties& modeInfo::getProperties() {
      static occa::properties properties;
      return properties;
    }

    occa::mode<serial::modeInfo,
               serial::device,
               serial::kernel,
               serial::memory> mode("Serial");
  }
}
