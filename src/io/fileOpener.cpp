/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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
#include <occa/defines.hpp>
#include <occa/io/fileOpener.hpp>
#include <occa/io/utils.hpp>

namespace occa {
  namespace io {
    fileOpener::~fileOpener() {}

    std::vector<fileOpener*>& fileOpener::getOpeners() {
      static std::vector<fileOpener*> openers;
      return openers;
    }

    fileOpener& fileOpener::defaultOpener() {
      static defaultFileOpener fo;
      return fo;
    }

    void fileOpener::add(fileOpener* opener) {
      getOpeners().push_back(opener);
    }

    fileOpener& fileOpener::get(const std::string &filename) {
      std::vector<fileOpener*> &openers = getOpeners();
      for (size_t i = 0; i < openers.size(); ++i) {
        if (openers[i]->handles(filename)) {
          return *(openers[i]);
        }
      }
      return defaultOpener();
    }

    //---[ Default File Opener ]---------
    defaultFileOpener::defaultFileOpener() {}
    defaultFileOpener::~defaultFileOpener() {}

    bool defaultFileOpener::handles(const std::string &filename) {
      return true;
    }

    std::string defaultFileOpener::expand(const std::string &filename) {
      return filename;
    }
    //==================================

    //-----[ OCCA File Opener ]---------
    occaFileOpener::occaFileOpener() {}
    occaFileOpener::~occaFileOpener() {}

    bool occaFileOpener::handles(const std::string &filename) {
      return ((7 <= filename.size()) &&
              (filename.substr(0, 7) == "occa://"));
    }

    std::string occaFileOpener::expand(const std::string &filename) {
      if (filename.size() == 7) {
        return cachePath();
      }
      return (libraryPath() + filename.substr(7));
    }
    //==================================

    //-----[ Header File Opener ]-------
    headerFileOpener::headerFileOpener() {}
    headerFileOpener::~headerFileOpener() {}

    bool headerFileOpener::handles(const std::string &filename) {
      return ((2 <= filename.size()) &&
              (filename[0] == '"')   &&
              (filename[filename.size() - 1] == '"'));
    }

    std::string headerFileOpener::expand(const std::string &filename) {
      return filename.substr(1, filename.size() - 2);
    }
    //==================================

    //-----[ System Header File Opener ]---
    systemHeaderFileOpener::systemHeaderFileOpener() {}
    systemHeaderFileOpener::~systemHeaderFileOpener() {}

    bool systemHeaderFileOpener::handles(const std::string &filename) {
      return ((2 <= filename.size()) &&
              (filename[0] == '<')   &&
              (filename[filename.size() - 1] == '>'));
    }

    std::string systemHeaderFileOpener::expand(const std::string &filename) {
      return filename.substr(1, filename.size() - 2);
    }
    //==================================
  }
}
