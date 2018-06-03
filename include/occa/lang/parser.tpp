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

namespace occa {
  namespace lang {
    template <class funcType>
    void parser_t::setArgumentsFor(funcType &func) {
      tokenRangeVector argRanges;
      getArgumentRanges(argRanges);

      const int argCount = (int) argRanges.size();
      if (!argCount) {
        return;
      }

      for (int i = 0; i < argCount; ++i) {
        context.push(argRanges[i].start,
                     argRanges[i].end);

        func += loadVariable();

        context.pop();
        if (!success) {
          break;
        }
        context.set(argRanges[i].end + 1);
      }
    }

    template <class attributeType>
    void parser_t::addAttribute() {
      attributeType *attr = new attributeType();
      const std::string name = attr->name();

      OCCA_ERROR("Attribute [" << name << "] already exists",
                 attributeMap.find(name) == attributeMap.end());

      attributeMap[name] = attr;
    }
  }
}
