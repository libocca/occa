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
#include "modes/serial.hpp"
#include "builtins/types.hpp"
#include "builtins/attributes.hpp"
#include "builtins/transforms/finders.hpp"

namespace occa {
  namespace lang {
    namespace okl {
      serialParser::serialParser() :
        useRestrict(true),
        restrict_("__restrict__", qualifierType::custom) {
        addAttribute<attributes::kernel>();
        addAttribute<attributes::outer>();
        addAttribute<attributes::inner>();
        addAttribute<attributes::shared>();
        addAttribute<attributes::exclusive>();

        if (settings.has("serial/restrict")) {
          occa::json r = settings["serial/restrict"];
          if (r.isString()) {
            restrict_.name = r.string();
          } else if (r.isBoolean()
                     && !r.boolean()) {
            useRestrict = false;
          }
        }
        if (useRestrict) {
          addKeyword(keywords,
                     new qualifierKeyword(restrict_));
        }

        setupPreprocessor();
      }

      void serialParser::setupPreprocessor() {
        if (useRestrict) {
          preprocessor.compilerMacros["restrict"] = (
            macro_t::defineBuiltin(preprocessor,
                                   "restrict",
                                   restrict_.name)
          );
        }
      }

      void serialParser::onClear() {
        setupPreprocessor();
      }

      void serialParser::onPostParse() {
        setupKernels();
        setupExclusives();
      }

      void serialParser::setupKernels() {
        if (!success) {
          return;
        }
        // Get @kernels
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             root,
                             kernelSmnts);
        const int kernels = (int) kernelSmnts.size();
        for (int i = 0; i < kernels; ++i) {
          setupKernel(*((functionDeclStatement*) kernelSmnts[i]));
          if (!success) {
            break;
          }
        }
      }

      void serialParser::setupKernel(functionDeclStatement &kernelSmnt) {
        // @kernel -> extern "C"
        function_t &func = kernelSmnt.function;
        attributeToken_t &kernelAttr = kernelSmnt.attributes["kernel"];
        qualifiers_t &qualifiers = func.returnType.qualifiers;
        // Add extern "C"
        qualifiers.addFirst(kernelAttr.source->origin,
                            externC);
        // Remove other externs
        if (qualifiers.has(extern_)) {
          qualifiers -= extern_;
        }
        if (qualifiers.has(externCpp)) {
          qualifiers -= externCpp;
        }
        // Pass non-pointer arguments by reference
        const int argCount = (int) func.args.size();
        for (int i = 0; i < argCount; ++i) {
          variable_t &arg = *(func.args[i]);
          vartype_t &type = arg.vartype;
          if ((type.pointers.size() ||
               type.referenceToken)) {
            continue;
          }
          type.referenceToken = new operatorToken(arg.source->origin,
                                                  op::bitAnd);
        }
      }

      void serialParser::setupExclusives() {
        if (!success) {
          return;
        }
      }
    }
  }
}
