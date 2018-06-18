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
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/builtins/transforms/finders.hpp>
#include <occa/lang/modes/oklForStatement.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      bool checkKernels(statement_t &root) {
        // Get @kernels
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             root,
                             kernelSmnts);

        // Get @outer and @inner
        const int kernelCount = (int) kernelSmnts.size();
        if (kernelCount == 0) {
          occa::printError("No [@kernel] functions found");
          return false;
        }
        for (int i = 0; i < kernelCount; ++i) {
          statement_t *kernelSmnt = kernelSmnts[i];
          if (kernelSmnt->type() != statementType::functionDecl) {
            continue;
          }
          if (!checkKernel(*((functionDeclStatement*) kernelSmnt))) {
            return false;
          }
        }
        return true;
      }

      bool checkKernel(functionDeclStatement &kernelSmnt) {
        vartype_t &returnType = kernelSmnt.function.returnType;
        if (returnType.qualifiers.size()
            || (*returnType.type != void_)) {
          returnType.printError("[@kernel] functions must have a"
                                " [void] return type");
          return false;
        }

        return (checkLoops(kernelSmnt)
                && checkLoopOrders(kernelSmnt)
                && checkBreakAndContinue(kernelSmnt));
      }

      //---[ Declaration ]--------------
      bool checkLoops(functionDeclStatement &kernelSmnt) {
        // Make sure @outer and @inner loops exist
        // No @outer + @inner combo in for-loops
        // Proper simple declarations
        statementPtrVector outerSmnts, innerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             kernelSmnt,
                             outerSmnts);
        findStatementsByAttr(statementType::for_,
                             "inner",
                             kernelSmnt,
                             innerSmnts);

        return (checkForDoubleLoops(outerSmnts, "inner")
                && checkOklForStatements(kernelSmnt, outerSmnts, "outer")
                && checkOklForStatements(kernelSmnt, innerSmnts, "inner"));
      }

      bool checkForDoubleLoops(statementPtrVector &loopSmnts,
                               const std::string &badAttr) {
        int loopCount = (int) loopSmnts.size();
        for (int i = 0; i < loopCount; ++i) {
          statement_t &smnt = *(loopSmnts[i]);
          if (smnt.hasAttribute(badAttr)) {
            smnt.printError("for-loop cannot have both [@outer] and [@inner] attributes");
            return false;
          }
        }
        return true;
      }

      bool checkOklForStatements(functionDeclStatement &kernelSmnt,
                                 statementPtrVector &forSmnts,
                                 const std::string &attrName) {
        const int count = (int) forSmnts.size();
        if (!count) {
          kernelSmnt.printError("[@kernel] requires at least one [@"
                                + attrName
                                + "] for-loop");
          return false;
        }
        bool success = true;
        for (int i = 0; i < count; ++i) {
          success &= (
            oklForStatement::isValid(*((forStatement*) forSmnts[i]),
                                     attrName)
          );
        }
        return success;
      }
      //================================

      //---[ Loop Logic ]---------------
      bool oklLoopMatcher(statement_t &smnt) {
        return (smnt.hasAttribute("outer")
                || smnt.hasAttribute("inner"));
      }

      bool oklDeclAttrMatcher(statement_t &smnt,
                              const std::string &attr) {
        declarationStatement &declSmnt = (declarationStatement&) smnt;
        const int declCount = (int) declSmnt.declarations.size();
        for (int i = 0; i < declCount; ++i) {
          variableDeclaration &decl = declSmnt.declarations[i];
          variable_t &var = *(decl.variable);
          if (var.hasAttribute(attr)) {
            return true;
          }
        }
        return false;
      }

      bool oklAttrMatcher(statement_t &smnt,
                          const std::string &attr) {
        // TODO: Custom expr matcher for statements
        if (smnt.type() & statementType::declaration) {
          return oklDeclAttrMatcher(smnt, attr);
        }
        exprNode *expr = ((expressionStatement&) smnt).expr;
        exprNodeVector nodes;
        findExprNodesByAttr(exprNodeType::variable,
                            attr,
                            *expr,
                            nodes);
        return nodes.size();
      }

      bool oklSharedMatcher(statement_t &smnt) {
        return oklAttrMatcher(smnt, "shared");
      }

      bool oklExclusiveMatcher(statement_t &smnt) {
        return oklAttrMatcher(smnt, "exclusive");
      }

      bool checkLoopOrders(functionDeclStatement &kernelSmnt) {
        // @outer > @inner
        // Same # of @inner in each @outer
        transforms::smntTreeNode root;
        bool success = true;

        findStatementTree(statementType::for_,
                          kernelSmnt,
                          oklLoopMatcher,
                          root);
        success = checkLoopOrder(root);
        root.free();
        if (!success) {
          return false;
        }

        findStatementTree((statementType::declaration |
                           statementType::expression),
                          kernelSmnt,
                          oklSharedMatcher,
                          root);
        success = checkSharedOrder(root);
        root.free();
        if (!success) {
          return false;
        }

        findStatementTree((statementType::declaration |
                           statementType::expression),
                          kernelSmnt,
                          oklExclusiveMatcher,
                          root);
        success = checkExclusiveOrder(root);
        root.free();
        return success;
      }

      bool checkLoopOrder(transforms::smntTreeNode &root) {
        const int loops = (int) root.size();
        for (int i = 0; i < loops; ++i) {
          transforms::smntTreeNode &loopNode = *(root[i]);
          forStatement &forSmnt = *((forStatement*) loopNode.smnt);
          const bool isOuter = forSmnt.hasAttribute("outer");
          // Keep track of @outer/@inner stack and report errors
          int outerCount = isOuter;
          int innerCount = !isOuter;
          if (!checkLoopType(loopNode,
                             outerCount,
                             innerCount)) {
            return false;
          }
        }
        return true;
      }

      bool checkLoopType(transforms::smntTreeNode &node,
                         int &outerCount,
                         int &innerCount) {
        const int children = (int) node.size();
        if (!children) {
          return true;
        }

        int lastOuterCount, lastInnerCount;
        for (int i = 0; i < children; ++i) {
          forStatement &forSmnt = *((forStatement*) node[i]->smnt);
          const bool isOuter = forSmnt.hasAttribute("outer");

          if (!outerCount && !isOuter) {
            forSmnt.printError("[@inner] loops should be contained inside [@outer] loops");
            return false;
          }
          if (isOuter && innerCount) {
            forSmnt.printError("[@outer] loops shouldn't be contained inside [@inner] loops");
            return false;
          }

          int childOuterCount = outerCount + isOuter;
          int childInnerCount = innerCount + !isOuter;
          if (!checkLoopType(*node[i], childOuterCount, childInnerCount)) {
            return false;
          }

          // Make sure we have consistent counts
          if (i) {
            if (childOuterCount != lastOuterCount) {
              forSmnt.printError("Inconsistent number of [@outer] loops");
              node[i-1]->smnt->printError("Compared to this [@outer] loop");
              return false;
            }
            if (childInnerCount != lastInnerCount) {
              forSmnt.printError("Inconsistent number of [@inner] loops");
              node[i-1]->smnt->printError("Compared to this [@inner] loop");
              return false;
            }
          }
          lastOuterCount = childOuterCount;
          lastInnerCount = childInnerCount;
        }
        outerCount = lastOuterCount;
        innerCount = lastInnerCount;
        return true;
      }
      //================================

      //---[ Type Logic ]---------------
      bool checkSharedOrder(transforms::smntTreeNode &root) {
        // Decl: @outer > @shared > @inner
        //     : Array with evaluable sizes
        // Expr: @outer > @inner  > @shared
        const int children = (int) root.size();
        for (int i = 0; i < children; ++i) {
          transforms::smntTreeNode &node = *(root[i]);
          if (!checkOKLTypeInstance(*node.smnt, "shared")
              || !checkValidSharedArray(*node.smnt)) {
            return false;
          }
        }
        return true;
      }

      bool checkExclusiveOrder(transforms::smntTreeNode &root) {
        // Decl: @outer > @exclusive > @inner
        // Expr: @outer > @inner     > @exclusive
        const int children = (int) root.size();
        for (int i = 0; i < children; ++i) {
          transforms::smntTreeNode &node = *(root[i]);
          if (!checkOKLTypeInstance(*node.smnt, "exclusive")) {
            return false;
          }
        }
        return true;
      }

      bool checkOKLTypeInstance(statement_t &typeSmnt,
                                const std::string &attr) {
        bool inOuter = false;
        bool inInner = false;
        statement_t *smnt = &typeSmnt;
        while (smnt) {
          if (smnt->type() & statementType::for_) {
            inInner |= smnt->hasAttribute("inner");
            inOuter |= smnt->hasAttribute("outer");
          }
          smnt = smnt->up;
        }

        // TODO: Make sure it's in the inner-most @inner loop
        const bool isExpr = (typeSmnt.type() == statementType::expression);
        if (!isExpr) {
          if (inInner) {
            typeSmnt.printError("Cannot define [@" + attr + "] variables inside"
                                " an [@inner] loop");
            return false;
          }
          if (!inOuter) {
            typeSmnt.printError("Must define [@" + attr + "] variables between"
                                " [@outer] and [@inner] loops");
            return false;
          }
        } else if (!inInner) {
          typeSmnt.printError("Cannot use [@" + attr + "] variables outside"
                              " an [@inner] loop");
          return false;
        }
        return true;
      }

      bool checkValidSharedArray(statement_t &smnt) {
        if (!(smnt.type() == statementType::declaration)) {
          return true;
        }
        declarationStatement &declSmnt = (declarationStatement&) smnt;
        const int declCount = (int) declSmnt.declarations.size();
        for (int i = 0; i < declCount; ++i) {
          variableDeclaration &decl = declSmnt.declarations[i];
          vartype_t &vartype = decl.variable->vartype;
          const int arrayCount = vartype.arrays.size();
          if (!arrayCount) {
            decl.printError("[@shared] variables must be arrays");
            return false;
          }
          for (int ai = 0; ai < arrayCount; ++ai) {
            array_t &array = vartype.arrays[ai];
            if (!array.size ||
                !array.size->canEvaluate()) {
              array.printError("[@shared] variables must have sizes known at compile-time");
              return false;
            }
          }
        }
        return true;
      }
      //================================

      //---[ Skip Logic ]---------------
      bool checkBreakAndContinue(functionDeclStatement &kernelSmnt) {
        // No break or continue directly inside @outer/@inner loops
        // It's ok inside regular loops inside @outer/@inner
        statementPtrVector skipStatements;
        findStatementsByType((statementType::continue_ |
                              statementType::break_),
                             kernelSmnt,
                             skipStatements);

        const int count = (int) skipStatements.size();
        for (int i = 0; i < count; ++i) {
          statement_t &skipSmnt = *(skipStatements[i]);
          statement_t *s = skipSmnt.up;
          while (s) {
            const int sType = s->type();
            if (sType & (statementType::while_ |
                         statementType::switch_)) {
              break;
            }
            if (sType & statementType::for_) {
              if (s->hasAttribute("inner")) {
                skipSmnt.printError("Statement cannot be directly inside an [@inner] loop");
                s->printError("[@inner] loop is here");
                return false;
              }
              if (s->hasAttribute("outer")) {
                skipSmnt.printError("Statement cannot be directly inside an [@outer] loop");
                s->printError("[@outer] loop is here");
                return false;
              }
              break;
            }
            s = s->up;
          }
        }
        return true;
      }
      //================================

      //---[ Transformations ]----------
      void addAttributes(parser_t &parser) {
        parser.addAttribute<attributes::barrier>();
        parser.addAttribute<attributes::exclusive>();
        parser.addAttribute<attributes::inner>();
        parser.addAttribute<attributes::kernel>();
        parser.addAttribute<attributes::outer>();
        parser.addAttribute<attributes::restrict>();
        parser.addAttribute<attributes::shared>();
      }

      void setLoopIndices(functionDeclStatement &kernelSmnt) {
        statementPtrVector outerSmnts, innerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             kernelSmnt,
                             outerSmnts);
        findStatementsByAttr(statementType::for_,
                             "inner",
                             kernelSmnt,
                             innerSmnts);

        const int outerCount = (int) outerSmnts.size();
        for (int i = 0; i < outerCount; ++i) {
          setForLoopIndex(*((forStatement*) outerSmnts[i]),
                          "outer");
        }

        const int innerCount = (int) innerSmnts.size();
        for (int i = 0; i < innerCount; ++i) {
          setForLoopIndex(*((forStatement*) innerSmnts[i]),
                          "inner");
        }
      }

      void setForLoopIndex(forStatement &forSmnt,
                           const std::string &attr) {
        attributeToken_t &oklAttr = forSmnt.attributes[attr];
        if (oklAttr.args.size()) {
          return;
        }

        oklForStatement oklForSmnt(forSmnt);
        const int loopIndex = oklForSmnt.oklLoopIndex();

        oklAttr.args.push_back(
          new primitiveNode(oklAttr.source->clone(),
                            loopIndex)
        );
      }
      //================================
    }
  }
}
