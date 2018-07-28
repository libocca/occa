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

#include <sstream>

#include <occa/tools/exception.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  exception::exception(const std::string &header,
                       const std::string &filename,
                       const std::string &function,
                       const int line,
                       const std::string &message = "") {

    std::stringstream ss;

    std::string banner = "---[ " + header + " ]";
    ss << '\n'
       << banner << std::string(80 - banner.size(), '-') << '\n'
       << "    File     : " << filename << '\n'
       << "    Function : " << function << '\n'
       << "    Line     : " << line     << '\n'
       << "    Message  : " << message << '\n'
       << "    Stack    :\n"
       << sys::stacktrace(4, "      ")
       << std::string(80, '=') << '\n';

    output = ss.str();
  }

  exception::~exception() throw() {}

  const char* exception::what() const throw() {
    return output.c_str();
  }

  std::string exception::toString() const {
    return output;
  }

  std::ostream& operator << (std::ostream& out,
                             exception &exc) {
    out << exc.toString();
    return out;
  }
}
