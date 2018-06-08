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

#ifndef OCCA_TOOLS_IO_FILEOPENER_HEADER
#define OCCA_TOOLS_IO_FILEOPENER_HEADER

#include <iostream>
#include <vector>

namespace occa {
  namespace env {
    class envInitializer_t;
  }

  namespace io {
    class fileOpener {
      friend class occa::env::envInitializer_t;

    private:
      static std::vector<fileOpener*>& getOpeners();
      static fileOpener& defaultOpener();

    public:
      static fileOpener& get(const std::string &filename);
      static void add(fileOpener* opener);

      virtual ~fileOpener();

      virtual bool handles(const std::string &filename) = 0;
      virtual std::string expand(const std::string &filename) = 0;
    };

    //---[ Default File Opener ]---------
    class defaultFileOpener : public fileOpener {
    public:
      defaultFileOpener();
      virtual ~defaultFileOpener();

      bool handles(const std::string &filename);
      std::string expand(const std::string &filename);
    };
    //==================================

    //---[ OCCA File Opener ]------------
    class occaFileOpener : public fileOpener {
    public:
      occaFileOpener();
      virtual ~occaFileOpener();

      bool handles(const std::string &filename);
      std::string expand(const std::string &filename);
    };
    //==================================

    //---[ Header File Opener ]----------
    class headerFileOpener : public fileOpener {
    public:
      headerFileOpener();
      virtual ~headerFileOpener();

      bool handles(const std::string &filename);
      std::string expand(const std::string &filename);
    };
    //==================================

    //---[ System Header File Opener ]------
    class systemHeaderFileOpener : public fileOpener {
    public:
      systemHeaderFileOpener();
      virtual ~systemHeaderFileOpener();

      bool handles(const std::string &filename);
      std::string expand(const std::string &filename);
    };
    //==================================
  }
}

#endif
