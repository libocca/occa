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

#include "occa/parser/statement.hpp"
#include "occa/parser/parser.hpp"
#include "occa/tools/string.hpp"

namespace occa {
  namespace parserNS {
    //---[ Exp Node ]-------------------------------
    expNode::expNode() :
      sInfo(NULL),

      value(""),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL) {}

    expNode::expNode(const std::string &str) :
      sInfo(NULL),

      value(str),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL) {}

    expNode::expNode(const char *c) :
      sInfo(NULL),

      value(c),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL) {}

    expNode::expNode(const expNode &e) :
      sInfo(e.sInfo),

      value(e.value),
      info(e.info),

      up(e.up),

      leafCount(e.leafCount),
      leaves(e.leaves) {

      for (int i = 0; i < leafCount; ++i)
        leaves[i]->up = this;
    }

    expNode::expNode(statement &s) :
      sInfo(&s),

      value(""),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL) {}

    expNode& expNode::operator = (const expNode &e) {
      sInfo = e.sInfo;

      value = e.value;
      info  = e.info;

      up = e.up;

      leafCount = e.leafCount;
      leaves    = e.leaves;

      for (int i = 0; i < leafCount; ++i)
        leaves[i]->up = this;

      return *this;
    }

    bool expNode::operator == (expNode &e) {
      return sameAs(e);
    }

    hash_t expNode::hash() {
      if (info & expType::hasInfo)
        return occa::hash(leaves[0]);

      hash_t hash_ = occa::hash(value);

      for (int i = 0; i < leafCount; ++i) {
        hash_ ^= leaves[i]->hash();
      }
      return hash_;
    }

    bool expNode::sameAs(expNode &e, const bool nestedSearch) {
      if (info != e.info)
        return false;

      if (info & expType::hasInfo)
        return (leaves[0] == e.leaves[0]);

      if (value != e.value)
        return false;

      if (!nestedSearch)
        return true;

      if (leafCount != e.leafCount)
        return false;

      if (hash() != e.hash())
        return false;

      bool *found = new bool[leafCount];

      for (int i = 0; i < leafCount; ++i)
        found[i] = false;

      // [-] Ugh, N^2
      for (int i = 0; i < leafCount; ++i) {
        expNode &l1 = *(leaves[i]);
        bool iFound = false;

        for (int j = 0; j < leafCount; ++j) {
          if (found[j])
            continue;

          expNode &l2 = *(e.leaves[j]);

          if (l1.sameAs(l2, !nestedSearch) &&
             l1.sameAs(l2,  nestedSearch)) {

            iFound   = true;
            found[j] = true;
            break;
          }
        }

        if (!iFound)
          return false;
      }

      return true;
    }

    expNode expNode::makeFloatingLeaf() {
      expNode fLeaf;

      fLeaf.sInfo = sInfo;
      fLeaf.up    = this;

      return fLeaf;
    }

    void expNode::loadFromNode(expNode &allExp,
                               const int parsingLanguage) {

      int expPos = 0;

      loadFromNode(allExp, expPos, parsingLanguage);
    }

    void expNode::loadFromNode(expNode &allExp,
                               int &expPos,
                               const int parsingLanguage) {

      if (allExp.leafCount <= expPos) {
        sInfo->info = smntType::invalidStatement;
        return;
      }

      const bool parsingC       = (parsingLanguage & parserInfo::parsingC);
      const bool parsingFortran = (parsingLanguage & parserInfo::parsingFortran);

      int expStart = expPos;

      sInfo->labelStatement(allExp, expPos, parsingLanguage);

      expNode *firstLeaf = this;

      // expPos is not included, it starts the next expNode tree
      if ((1 < (expPos - expStart)) ||
         (0 < allExp[expStart].leafCount)) {
        useExpLeaves(allExp, expStart, (expPos - expStart));
        firstLeaf = leaves[0];
      }
      else {
        info  = allExp[expStart].info;
        value = allExp[expStart].value;
      }

      // printf("Copied expNode:\n");
      // print();

      // Don't need to load stuff
      if ((sInfo->info & (smntType::skipStatement      |
                          smntType::macroStatement     |
                          smntType::namespaceStatement |
                          smntType::gotoStatement      |
                          smntType::blockStatement))      ||
          (sInfo->info == smntType::occaFor)              ||
          (sInfo->info == smntType::elseStatement)        ||
          (sInfo->info == smntType::doWhileStatement)) {

        if (sInfo->info == smntType::elseStatement) {
          info  = expType::checkSInfo;
          value = "";
          freeThis();
        }
        else if (sInfo->info == smntType::gotoStatement) {
          leafCount = 1;
        }

        return;
      }

      std::string &firstValue = firstLeaf->value;

      //---[ Special Type ]---
      if (firstLeaf->info & expType::specialKeyword) {
        if ((firstValue == "break")    ||
           (firstValue == "continue")) {

          info = expType::transfer_;

          if ((firstValue == "continue") &&
             (sInfo->distToOccaForLoop() <= sInfo->distToForLoop())) {

            value = "occaContinue";
          }
          else {
            value = firstValue;
          }

          freeThis();

          return;
        }

        // [-] Doesn't support GCC's twisted [Labels as Values]
        if (firstValue == "goto") {
          OCCA_ERROR("Goto check [" << toString() << "] needs label",
                     1 < leafCount);

          info  = expType::goto_;
          value = allExp[expStart + 1];
          return;
        }

        // Case where nodeRoot = [case, return]

        if ((firstValue == "case") ||
           (firstValue == "default")) {
          info = expType::checkSInfo;
        }
        else if ((parsingC       && (firstValue == "return")) ||
                (parsingFortran && (firstValue == "RETURN"))) {

          info = expType::return_;

          // it's only return
          if (leafCount == 0) {
            value = "";
            return;
          }

          if (firstLeaf == leaves[0])
            removeNode(0);

          // Don't put the [;]
          if (leafCount &&
             (leaves[leafCount - 1]->value == ";")) {

            --leafCount;
          }
        }

        if (isInlinedASM(firstValue)) {
          info = expType::asm_;

          value = firstValue;

          if ((1 < leafCount) &&
             (leaves[1]->value == "(")) {

            setLeaf(*(leaves[1]), 0);
            leafCount = 1;
          }
          else
            leafCount = 0;

          return;
        }

        // [occaParallelFor][#]
        // 15              + 1 = 16
        if ((firstValue.find("occaParallelFor") != std::string::npos) &&
           (firstValue.size() == 16)) {

          sInfo->info = smntType::macroStatement;
          info        = expType::printValue;

          return;
        }

        if (firstValue == "occaUnroll") {
          organizeNode();

          value = (std::string) *this;

          free();

          sInfo->info  = smntType::macroStatement;
          info         = expType::printValue;

          return;
        }
      }
      //======================

      if (parsingLanguage & parserInfo::parsingC)
        organizeNode();
      else
        organizeFortranNode();

      // if (sInfo)
      //   sInfo->printDebugInfo();
    }

    // @(attributes)
    void expNode::loadAttributes() {
      int leafPos = 0;

      while(leafPos < leafCount) {
        if (isAnAttribute(*this, leafPos)) {
          const int leafStart = leafPos;

          loadAttributes(*this, leafPos);

          removeNodes(leafStart, leafPos - leafStart);
          leafPos = leafStart;

          continue;
        }

        ++leafPos;
      }
    }

    // @(attributes)
    void expNode::loadAttributes(expNode &allExp,
                                 int &expPos) {

      if ((sInfo == NULL)              ||
         !isAnAttribute(allExp, expPos)) {

        return;
      }

      expPos = updateAttributeMap(sInfo->attributeMap,
                                  allExp,
                                  expPos);
    }

    void expNode::organizeNode() {
      changeExpTypes();
      initOrganization();

      if (sInfo == NULL)
        organize();

      else if (sInfo->info & smntType::declareStatement) {
        organizeDeclareStatement();
      }

      else if (sInfo->info & smntType::updateStatement)
        organizeUpdateStatement();

      else if ((sInfo->info & (smntType::ifStatement    |
                              smntType::forStatement   |
                              smntType::whileStatement |
                              smntType::switchStatement)) &&
              (sInfo->info != smntType::elseStatement)) {

        organizeFlowStatement();
      }

      else if (sInfo->info & smntType::functionStatement)
        organizeFunctionStatement();

      else if (sInfo->info & smntType::structStatement)
        organizeStructStatement();

      else if (sInfo->info & smntType::caseStatement)
        organizeCaseStatement();

      else
        organize();
    }

    void expNode::organizeFortranNode() {
      changeExpTypes();

      if (leafCount == 0)
        return;

      if (leaves[leafCount - 1]->value == "\\n")
        --leafCount;

      if (sInfo == NULL)
        organize();

      else if (sInfo->info & smntType::declareStatement)
        organizeFortranDeclareStatement();

      else if (sInfo->info & smntType::updateStatement)
        organizeFortranUpdateStatement();

      else if ((sInfo->info & (smntType::ifStatement  |
                              smntType::forStatement |
                              smntType::whileStatement)) &&
              (sInfo->info != smntType::elseStatement)) {

        organizeFortranFlowStatement();
      }

      else if (sInfo->info & smntType::functionStatement)
        organizeFortranFunctionStatement();

      else if (sInfo->info & smntType::structStatement)
        organizeStructStatement();

      else if (sInfo->info & smntType::caseStatement)
        organizeCaseStatement(parserInfo::parsingFortran);

      else
        organize(parserInfo::parsingFortran);
    }

    void expNode::organize(const int parsingLanguage) {
      if (leafCount == 0)
        return;

      // @(attributes)
      loadAttributes();

      if (parsingLanguage & parserInfo::parsingC)
        organizeLeaves();
      else
        organizeFortranLeaves();
    }

    void expNode::organizeDeclareStatement(const info_t flags) {
      info = expType::declaration;

      int varCount = 1 + typeInfo::delimiterCount(*this, ",");
      int leafPos  = 0;

      varInfo *firstVar = NULL;

      // Store variables and stuff
      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(varCount);

      for (int i = 0; i < varCount; ++i) {
        expNode &leaf = newExp[i];
        varInfo &var  = leaf.addVarInfoNode(0);

        int nextLeafPos = var.loadFrom(*this, leafPos, firstVar);

        if (flags & expFlag::addVarToScope) {
          if (flags & expFlag::addToParent) {
            if (sInfo->up != NULL)
              sInfo->up->addVariable(&var, sInfo);
          }
          else {
            sInfo->addVariable(&var);
          }
        }

        leaf[0].info |= expType::declaration;

        // Make sure the first one prints out the type
        if (i == 0) {
          leaf[0].info |= expType::type;
          firstVar = &var;
        }

        removeNodes(leafPos, nextLeafPos - leafPos);

        int sExpStart = leafPos;
        int sExpEnd   = typeInfo::nextDelimiter(*this, leafPos, ",");

        leafPos = sExpEnd;

        // Don't put the [;]
        if (leafCount              &&
           (sExpEnd == leafCount) &&
           (leaves[sExpEnd - 1]->value == ";")) {

          --sExpEnd;
        }

        if (sExpStart < sExpEnd) {
          leaf.addNodes(1, sExpEnd - sExpStart);

          for (int j = sExpStart; j < sExpEnd; ++j)
            expNode::swap(*leaf.leaves[j - sExpStart + 1], *leaves[j]);

          leaf.organize();
        }

        if (leafPos < leafCount)
          removeNode(leafPos);
      }

      expNode::swap(*this, newExp);
    }

    void expNode::organizeUpdateStatement() {
      // Don't put the [;]
      if (leafCount &&
         (leaves[leafCount - 1]->value == ";")) {

        info |= expType::hasSemicolon;
        --leafCount;
      }
      else if (leafCount == 0) {
        info = expType::checkSInfo;
        return;
      }

      organize();
    }

    void expNode::organizeFlowStatement() {
      info = expType::checkSInfo;

      expNode &expDown = *(leaves[1]);

      int statementCount    = 1 + typeInfo::delimiterCount(expDown, ";");

      int maxStatementCount = ((sInfo->info & smntType::forStatement) ?
                               3 : 1);

      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(statementCount);

      int leafPos = 0;

      for (int i = 0; i < statementCount; ++i) {
        expNode &leaf = newExp[i];

        int nextLeafPos = typeInfo::nextDelimiter(expDown, leafPos, ";");

        if (leafPos < nextLeafPos) {
          leaf.useExpLeaves(expDown, leafPos, (nextLeafPos - leafPos));

          bool hasDeclare = ((sInfo->info & smntType::forStatement) && (i == 0));

          if (hasDeclare &&
             ((leaf.leafCount == 0) ||
              !(leaf[0].info & (expType::qualifier |
                                expType::type      |
                                expType::typeInfo)))) {

            hasDeclare = false;
          }

          if (!hasDeclare) {
            leaf.organize();

            if (maxStatementCount <= i) {
              std::stringstream ss;

              ss << "@(" << leaf << ")";

              sInfo->addAttribute(ss.str());

              sInfo->updateInitialLoopAttributes();

              leaf.free();
              newExp.removeNode(-1);
            }
          }
          else {
            leaf.organizeDeclareStatement(expFlag::addVarToScope);

            expNode &flatRoot = *(makeDumbFlatHandle());

            for (int j = 0; j < flatRoot.leafCount; ++j) {
              expNode &n = flatRoot[j];

              // Variables that were just defined
              if (n.info & expType::unknown) {
                varInfo *var = sInfo->hasVariableInLocalScope(n.value);

                if (var != NULL)
                  n.putVarInfo(*var);
              }
            }

            freeFlatHandle(flatRoot);
          }
        }

        leafPos = (nextLeafPos + 1);
      }

      expNode::swap(*this, newExp);
    }

    void expNode::organizeFunctionStatement() {
      const bool functionIsDefined = (sInfo->info & smntType::functionDefinition);

      if (functionIsDefined) {
        info = (expType::function | expType::declaration);
      } else {
        info = (expType::function | expType::prototype);
      }

      if (leafCount == 0) {
        return;
      }

      varInfo &var = addVarInfoNode(0);
      int leafPos  = var.loadFrom(*this, 1);

      if ((sInfo->up != NULL) &&
         (!sInfo->up->scope->hasLocalVariable(var.name))) {

        sInfo->up->addVariable(&var);
      }

      if (functionIsDefined) {
        leaves[0]->info |= expType::type;

        for (int i = 0; i < var.argumentCount; ++i) {
          sInfo->addVariable(var.argumentVarInfos[i]);
        }
      }

      removeNodes(1, leafPos);
    }

    void expNode::organizeStructStatement() {
      info = expType::struct_;

      int leafPos = 0;

      // Store type
      expNode newExp(*sInfo);
      newExp.info = info;

      // For readability
      const bool addTypeToScope = true;

      typeInfo &type = newExp.addTypeInfoNode(0);
      leafPos = type.loadFrom(*this, leafPos, addTypeToScope);

      if ((leafCount <= leafPos) ||
         ((*this)[leafPos].value == ";")) {

        expNode::swap(*this, newExp);
        return;
      }

      removeNodes(0, leafPos);

      organizeDeclareStatement();

      const int varCount = getVariableCount();

      for (int i = 0; i < varCount; ++i) {
        expNode *varNode = getVariableInfoNode(i);
        varInfo &var     = varNode->getVarInfo();

        var.baseType = &type;

        if (i == 0)
          varNode->info &= ~expType::type;
      }

      typeInfo &type2 = addTypeInfoNode(0);
      type2 = type;

      newExp.free();
    }

    void expNode::organizeCaseStatement(const int parsingLanguage) {
      // Fortran doesn't have [:] leaf at the end
      if (parsingLanguage & parserInfo::parsingC)
        --leafCount;

      // Remove [case] or [default]
      for (int i = 1; i < leafCount; ++i)
        leaves[i - 1] = leaves[i];

      --leafCount;
    }

    //  ---[ Fortran ]------------------
    void expNode::organizeFortranDeclareStatement() {
      info = expType::declaration;

      int varCount = 1;

      varInfo dummyVar;
      int varStart = dummyVar.loadTypeFromFortran(*this, 0);

      leafCount = typeInfo::nextDelimiter(*this, 0, "\\n");

      // [+] Needs to be updated on C++
      for (int i = varStart; i < leafCount; ++i) {
        if (leaves[i]->value == ",")
          ++varCount;
      }

      int leafPos  = 0;

      varInfo *firstVar = NULL;

      // Store variables and stuff
      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(varCount);

      for (int i = 0; i < varCount; ++i) {
        expNode &leaf = newExp[i];
        varInfo &var  = leaf.addVarInfoNode(0);

        int nextLeafPos = var.loadFromFortran(*this, leafPos, firstVar);

        if (var.stackPointerCount)
          var.stackPointersUsed = 1;

        leaf[0].info |= expType::declaration;

        // Make sure the first one prints out the type
        if (i == 0) {
          leaf[0].info |= expType::type;
          firstVar = &var;
        }

        removeNodes(leafPos, nextLeafPos - leafPos);

        int sExpStart = leafPos;
        int sExpEnd   = typeInfo::nextDelimiter(*this, leafPos, ",");

        leafPos = sExpEnd;

        if (sExpStart < sExpEnd) {
          leaf.addNodes(1, sExpEnd - sExpStart);

          for (int j = sExpStart; j < sExpEnd; ++j)
            expNode::swap(*leaf.leaves[j - sExpStart + 1], *leaves[j]);

          leaf.organizeFortranLeaves();
        }

        if (leafPos < leafCount)
          removeNode(leafPos);
      }

      expNode::swap(*this, newExp);

      //---[ Check DIMENSION ]----------
      if (firstVar->hasQualifier("DIMENSION")) {
        for (int i = 1; i < varCount; ++i) {
          varInfo &var = leaves[i]->getVarInfo(0);

          var.stackPointerCount = firstVar->stackPointerCount;
          var.stackPointersUsed = firstVar->stackPointersUsed;
          var.stackExpRoots     = firstVar->stackExpRoots;
        }

        firstVar->removeQualifier("DIMENSION");
      }

      //---[ Check INTENT ]-------------
      const bool hasIn    = firstVar->hasQualifier("INTENTIN");
      const bool hasOut   = firstVar->hasQualifier("INTENTOUT");
      const bool hasInOut = firstVar->hasQualifier("INTENTINOUT");

      if (hasIn || hasOut || hasInOut) {
        for (int i = 0; i < varCount; ++i) {
          varInfo &var = leaves[i]->getVarInfo(0);

          // Update here, also update in [Add variables to scope]
          //   for older Fortran codes without [INTENT]
          // Hide stack info in arguments
          var.stackPointersUsed = 0;

          // Make sure it registers as a pointer
          if ((var.pointerCount      == 0) &&
             (var.stackPointerCount != 0)) {

            var.pointerCount = 1;
            var.rightQualifiers.add("*", 0);
          }

          if (hasIn)
            var.removeQualifier("INTENTIN");
          else if (hasOut)
            var.removeQualifier("INTENTOUT");
          else if (hasInOut)
            var.removeQualifier("INTENTINOUT");

          varInfo *argVar = sInfo->hasVariableInScope(var.name);

          OCCA_ERROR("Error: variable [" << var << "] is not a function argument",
                     argVar != NULL);

          *(argVar) = var;
        }

        sInfo->info = smntType::skipStatement;
      }
      else { // Add variables to scope
        for (int i = 0; i < varCount; ++i) {

          varInfo &var = leaves[i]->getVarInfo(0);
          varInfo *pVar = sInfo->hasVariableInScope(var.name);

          // Check if it's a function argument
          if (pVar != NULL) {
            statement *s = sInfo->parser.varOriginMap[pVar];

            if (s &&
               (s->info & smntType::functionDefinition)) {

              // Hide stack info in arguments
              var.stackPointersUsed = 0;

              // Make sure it registers as a pointer
              if ((var.pointerCount      == 0) &&
                 (var.stackPointerCount != 0)) {

                var.pointerCount = 1;
                var.rightQualifiers.add("*", 0);
              }

              *(pVar) = var;

              sInfo->info = smntType::skipStatement;
            }
            // Will give error message
            else if (sInfo->up != NULL) {
              sInfo->up->addVariable(&var, sInfo);
            }
          }
          else {
            if (sInfo->up != NULL) {
              sInfo->up->addVariable(&var, sInfo);
            }
          }
        }
      }
    }

    void expNode::organizeFortranUpdateStatement() {
      if (leafCount == 0)
        return;

      // Don't put the [;]
      if (leafCount &&
         (leaves[leafCount - 1]->value == ";")) {

        info |= expType::hasSemicolon;
        --leafCount;
      }

      // Function call
      if (leaves[0]->value == "CALL") {
        // Only [CALL]
        if (leafCount == 1) {
          sInfo->info = smntType::skipStatement;
          return;
        }

        if (sInfo->hasVariableInScope(leaves[1]->value)) {
          removeNode(0);

          leaves[0]->info = expType::function;

          organize(parserInfo::parsingFortran);

          info |= expType::hasSemicolon;
        }
        else {
          OCCA_ERROR("Function [" << (leaves[0]->value) << "] is not defined in ["
                     << toString() << "]",
                     false);
        }

        return;
      }

      organize(parserInfo::parsingFortran);

      varInfo *funcExp = sInfo->getFunctionVar();

      if ((funcExp == NULL)            ||
         ((*this)[0].value    != "=") ||
         ((*this)[0][0].value != funcExp->name)) {

        info |= expType::hasSemicolon;

        return;
      }

      expNode &retValueLeaf = *((*this)[0][1].clonePtr());

      free();

      info = expType::return_;

      reserve(1);
      setLeaf(retValueLeaf, 0);
    }

    void expNode::organizeFortranFlowStatement() {
      info = expType::checkSInfo;

      if (leafCount == 0)
        return;

      if (sInfo->info & smntType::forStatement) {
        organizeFortranForStatement();
      }
      // [IF/ELSE IF/DO WHILE]( EXPR )
      else if ((sInfo->info == smntType::ifStatement)     ||
              (sInfo->info == smntType::elseIfStatement) ||
              (sInfo->info == smntType::whileStatement)) {

        OCCA_ERROR("No expression in if-statement: " << *this << '\n',
                   leafCount != 0);

        leaves[0]       = leaves[1];
        leaves[0]->info = expType::root;
        leaves[0]->organize();

        leafCount = 1;
      }
      // [ELSE]
      else if (sInfo->info & smntType::elseStatement) {
        if (leafCount)
          free();
      }
    }

    void expNode::organizeFortranForStatement() {
      // [DO] iter=start,end[,stride][,loop]
      // Infinite [DO]
      if (leafCount == 1) {
        leaves[0]->value = "true";
        leaves[0]->info  = expType::presetValue;
        leafCount = 1;

        sInfo->info = smntType::whileStatement;

        return;
      }

      leafCount = typeInfo::nextDelimiter(*this, 0, "\\n");

      int statementCount = 1 + typeInfo::delimiterCount(*this, ",");

      OCCA_ERROR("Error: Wrong [DO] format [" << *this << "]",
                 (2 <= statementCount) && (statementCount <= 4));

      int pos[5];

      // Skip [DO], [iter], and [=]
      pos[0] = 3;

      // Find [,] positions
      for (int i = 0; i < statementCount; ++i) {
        pos[i + 1] = typeInfo::nextDelimiter(*this, pos[i], ",") + 1;

        OCCA_ERROR("Error: No expression given in [" << *this << "]",
                   pos[i] != (pos[i + 1] + 1));
      }

      // Check if last expressiong is an OCCA tag
      std::string &lastLeafValue = leaves[pos[statementCount - 1]]->value;

      const bool hasOccaTag = isAnOccaTag(lastLeafValue);

      expNode newExp(*sInfo);
      newExp.info = info;
      newExp.addNodes(3);

      if (hasOccaTag) {
        std::stringstream ss;

        ss << "@(" << lastLeafValue << ")";

        sInfo->addAttribute(ss.str());

        sInfo->updateInitialLoopAttributes();

        // Get rid of the tag
        --statementCount;
      }

      // Get iter var name
      const std::string &iter = leaves[1]->value;
      varInfo *var = sInfo->hasVariableInScope(iter);

      OCCA_ERROR("Error: Iterator [" << iter
                 << "] is not defined before [" << *this << "]",
                 var != NULL);

      // Fortran iterations are not modified
      std::vector<std::string> doNames;

      doNames.push_back("doStart");
      doNames.push_back("doEnd");
      doNames.push_back("doStride");
      doNames.push_back("doStrideSign");

      std::string &doStart      = doNames[0];
      std::string &doEnd        = doNames[1];
      std::string &doStride     = doNames[2];
      std::string &doStrideSign = doNames[3];

      sInfo->createUniqueVariables(doNames);

      expNode exp0 = sInfo->createOrganizedExpNodeFrom(*this,
                                                       pos[0],
                                                       (pos[1] - pos[0] - 1));
      expNode exp1 = sInfo->createOrganizedExpNodeFrom(*this,
                                                       pos[1],
                                                       (pos[2] - pos[1] - 1));

      const std::string exp0Str = exp0.toString();
      const std::string exp1Str = exp1.toString();

      OCCA_ERROR("Error, missing 1st statement in the [DO]: "
                 << toString() << '\n',
                 exp0Str.size() != 0);
      OCCA_ERROR("Error, missing 2nd statement in the [DO]: "
                 << toString() << '\n',
                 exp1Str.size() != 0);

      const std::string decl0 = "const int " + doStart + " = " + exp0Str;
      const std::string decl1 = "const int " + doEnd   + " = " + exp1Str;

      sInfo->up->addStatementFromSource(decl0);
      sInfo->up->addStatementFromSource(decl1);

      if (statementCount == 3) {
        expNode exp2 = sInfo->createOrganizedExpNodeFrom(*this,
                                                         pos[2],
                                                         (pos[3] - pos[2] - 1));

        const std::string exp2Str = exp2.toString();

        const std::string decl2 = "const int " + doStride + " = " + exp2Str;

        OCCA_ERROR("Error, missing 3rd statement in the [DO]: " << toString() << '\n',
                   exp2Str.size() != 0);

        sInfo->up->addStatementFromSource(decl2);

        const std::string decl3 = "const int " + doStrideSign + " = (1 - (2*(" + doStride + " < 0)))";

        sInfo->up->addStatementFromSource(decl3);
      }

      newExp[0] = sInfo->createOrganizedExpNodeFrom(iter + " = " + doStart);

      if (statementCount == 3) {
        newExp[1] = sInfo->createOrganizedExpNodeFrom("0 <= (" + doStrideSign + "* (" + doEnd + " - " + iter + "))");
        newExp[2] = sInfo->createOrganizedExpNodeFrom(iter + " += " + doStride);
      }
      else {
        newExp[1] = sInfo->createOrganizedExpNodeFrom(iter + " <= " + doEnd);
        newExp[2] = sInfo->createOrganizedExpNodeFrom("++" + iter);
      }

      expNode::swap(*this, newExp);
    }

    void expNode::organizeFortranFunctionStatement() {
      info = (expType::function | expType::declaration);

      if (leafCount == 0)
        return;

      varInfo &var = addVarInfoNode(0);
      int leafPos  = var.loadFromFortran(*this, 1);

      if ((sInfo->up != NULL)              &&
         !sInfo->up->scope->hasLocalVariable(var.name)) {

        sInfo->up->addVariable(&var);

        // Add initial arguments (they get updated later)
        for (int i = 0; i < var.argumentCount; ++i)
          sInfo->addVariable( &(var.getArgument(i)) );
      }

      removeNodes(1, leafPos);
    }
    //  ======================

    void expNode::translateOccaKeyword(expNode &exp,
                                       info_t preInfo,
                                       const int parsingLanguage_) {

      if (preInfo & expType::occaKeyword) {

        const bool parsingC       = (parsingLanguage_ & parserInfo::parsingC);
        const bool parsingFortran = (parsingLanguage_ & parserInfo::parsingFortran);

        if ((parsingC       && (exp.value == "directLoad")) ||
           (parsingFortran && upStringCheck(exp.value, "DIRECTLOAD"))) {

          exp.value = "occaDirectLoad";
        }

      }
    }

    bool expNode::isOrganized() {
      if (needsExpChange())
        return false;

      for (int i = 0; i < leafCount; ++i) {
        if (leaves[i]->needsExpChange())
          return false;
      }

      return true;
    }

    bool expNode::needsExpChange() {
      if ((info == expType::root) ||
         (info & expType::hasInfo)) {

        return false;
      }

      return (info & expType::firstPass);
    }

    void expNode::changeExpTypes(const int leafPos) {
      if (info & expType::hasInfo)
        return;

      const int upLeafCount = (up ? up->leafCount : 0);
      expNode *leftLeaf     = ((up && leafPos)         ?
                               up->leaves[leafPos - 1] :
                               NULL);

      if (info & expType::unknown) {
        info &= expType::firstPassMask;

        if (sInfo != NULL) {
          varInfo *nodeVar   = sInfo->hasVariableInScope(value);
          typeInfo *nodeType = sInfo->hasTypeInScope(value);

          if (nodeVar && nodeType) {
            // [const] [type]
            if ((leftLeaf != NULL) &&
               (leftLeaf->info & expType::qualifier)) {

              info = expType::type;
            }
            // [type] [X]
            else if ((leafPos == 0)  &&
                    (1 < upLeafCount) &&
                    ((*up)[1].info & (expType::varInfo  |
                                      expType::function |
                                      expType::unknown  |
                                      expType::variable))) {
              info = expType::type;
            }
            else {
              putVarInfo(*nodeVar);
              return;
            }
          }
          else if (nodeVar) {
            if ( !(nodeVar->info & varType::function) ) {
              putVarInfo(*nodeVar);
              return;
            }
            else
              info = expType::function; // [<>] Change to funcInfo
          }
          else if (nodeType) {
            // [type] [varName]
            if ((leftLeaf != NULL) &&
               (leftLeaf->info & expType::type)) {

              info = expType::unknown;
            }
            else {
              info = expType::type;
            }
          }
          else {
            info = expType::unknown;
          }
        }
      }
      else if (needsExpChange()) {
        info_t preInfo = info;

        info &= expType::firstPassMask;

        if (preInfo & expType::occaKeyword)
          translateOccaKeyword(*this, preInfo, true);

        if (preInfo & expType::descriptor) {
          if (up != NULL) {
            if (expHasQualifier(*up, leafPos))
              info = expType::qualifier;
            else
              info = expType::type;
          }
          else if (preInfo & expType::struct_) {
            info = expType::qualifier;
          }
          else {
            info = expType::type;
          }

          // For [*] and [&]
          if (preInfo & expType::operator_)
            info = (preInfo & expType::firstPassMask);
        }

        else if (preInfo & expType::C) {
          if (leafCount)
            changeExpTypes();
        }

        else if ( !(preInfo & (expType::presetValue |
                              expType::operator_)) ) {
          info = expType::printValue;
        }
      }

      for (int i = 0; i < leafCount; ++i)
        leaves[i]->changeExpTypes(i);
    }

    void expNode::initOrganization() {
      if (leafCount == 0)
        return;

      // Init ()'s bottom -> up
      // Organize leaves bottom -> up
      for (int i = 0; i < leafCount; ++i) {
        if ((leaves[i]->leafCount) &&
           !(leaves[i]->info & expType::hasInfo)) {

          leaves[i]->initOrganization();
        }
      }

      //---[ Level 0 ]------
      // [a][::][b]
      mergeNamespaces();

      // const int [*] x
      labelReferenceQualifiers();
      //====================
    }

    void expNode::organizeLeaves(const bool inRoot) {
      if (info & expType::hasInfo)
        return;

      // Organize leaves bottom -> up
      for (int i = 0; i < leafCount; ++i) {
        if ((leaves[i]->leafCount) &&
           !(leaves[i]->info & expType::hasInfo)) {

          leaves[i]->organizeLeaves(false);
        }
      }

      //---[ Level 1 ]------
      // [(class)]
      labelCasts();

      // <const int,float>
      mergeTypes();

      // a[3]
      mergeArrays();

      // class(...), class{1,2,3}
      mergeClassConstructs();

      // static_cast<>()
      mergeCasts();

      // [max(a,b)]
      mergeFunctionCalls();

      // ptr(a,b)
      mergePointerArrays();

      organizeLeaves(1);
      //====================

      //---[ Level 2 ]------
      // (class) x
      mergeClassCasts();

      // sizeof x
      mergeSizeOf();

      // new, new [], delete, delete []
      mergeNewsAndDeletes();

      organizeLeaves(2);
      //====================

      //---[ Level 3-14 ]---
      for (int i = 3; i <= 14; ++i)
        organizeLeaves(i);
      //====================

      //---[ Level 15 ]-----
      // throw x
      mergeThrows();
      //====================

      //---[ Level 16 ]-----
      organizeLeaves(16);
      //====================
    }

    void expNode::organizeFortranLeaves() {
      if (info & expType::hasInfo)
        return;

      // Organize leaves bottom -> up
      for (int i = 0; i < leafCount; ++i) {
        if ((leaves[i]->leafCount) &&
           !(leaves[i]->info & expType::hasInfo)) {

          leaves[i]->organizeFortranLeaves();
        }
      }

      mergeFortranArrays();
      mergeArrays();

      for (int i = 0; i < 12; ++i)
        organizeLeaves(i);

      translateFortranMemberCalls();
      translateFortranPow();
    }

    void expNode::organizeLeaves(const int level) {
      bool leftToRight = *(opLevelL2R[level]);

      int leafPos  = (leftToRight ? 0 : (leafCount - 1));
      const int ls = (leftToRight ? 1 : -1);

      while(true) {
        if (( (leftToRight) && (leafCount <= leafPos)) ||
           ((!leftToRight) && (leafPos < 0)))
          break;

        if ((leaves[leafPos]->leafCount)                  ||
           (leaves[leafPos]->info &  expType::hasInfo)   ||
           (leaves[leafPos]->info == expType::qualifier)) {

          leafPos += ls;
          continue;
        }

        std::string &lStr     = leaves[leafPos]->value;
        opLevelMapIterator it = opLevelMap[level]->find(lStr);

        if (it == opLevelMap[level]->end()) {
          leafPos += ls;
          continue;
        }

        const int levelType = it->second;

        if (levelType & expType::L_R) {
          bool updateNow = true;

          const int targetOff = ((levelType & expType::L) ? 1 : -1);
          const int target    = leafPos + targetOff;

          if ((target < 0) || (leafCount <= target)) {
            updateNow = false;
          }
          else {
            info_t lInfo = (*keywordType)[lStr];

            // Cases: & * + -
            if ((lInfo & expType::LR) ||
               ((lInfo & expType::L_R) == expType::L_R)) {
              const int invTarget = leafPos + ((targetOff == 1) ?
                                               -1 : 1);

              updateNow = false;

              if ((invTarget < 0) || (leafCount <= invTarget)) {
                updateNow = true;
              }
              else {
                // [-] Add
                // varInfo tVar  = leaves[target]->evaluateType();
                // varInfo itVar = leaves[invTarget]->evaluateType();

                if (leaves[invTarget]->info & expType::operator_)
                  updateNow = (leaves[invTarget]->leafCount == 0);
              }
            }
          }

          if (!updateNow) {
            leafPos += ls;
          }
          else {
            if (levelType & expType::L)
              leafPos = mergeLeftUnary(leafPos, leftToRight);
            else
              leafPos = mergeRightUnary(leafPos, leftToRight);
          }
        }
        else if (levelType & expType::LR)
          leafPos = mergeBinary(leafPos, leftToRight);
        else if (levelType & expType::LCR)
          leafPos = mergeTernary(leafPos, leftToRight);
        else
          leafPos += ls;
      }
    }

    int expNode::mergeRange(const int newLeafType,
                            const int leafPosStart,
                            const int leafPosEnd) {
      expNode *newLeaf = new expNode( makeFloatingLeaf() );

      newLeaf->info = newLeafType;
      newLeaf->addNodes(leafPosEnd - leafPosStart + 1);

      for (int i = 0; i < newLeaf->leafCount; ++i)
        newLeaf->setLeaf(*(leaves[leafPosStart + i]), i);

      setLeaf(*newLeaf, leafPosStart);

      for (int i = (leafPosEnd + 1); i < leafCount; ++i)
        leaves[leafPosStart + i - leafPosEnd] = leaves[i];

      leafCount -= (newLeaf->leafCount - 1);

      return (leafPosStart + 1);
    }

    // [a][::][b]
    void expNode::mergeNamespaces() {
#if 0
      int leafPos = 0;

      while(leafPos < leafCount) {
        if (leaves[leafPos]->value == "::") {
          // [-] Change after keeping track of namespaces
          int leafPosStart = leafPos - (leafPos &&
                                        (leaves[leafPos - 1]->info & expType::unknown));
          // int leafPosStart = leafPos - (leafPos &&
          //                               (leaves[leafPos - 1]->info & expType::namespace_));
          int leafPosEnd   = leafPos + 2;

          while((leafPosEnd < leafCount) &&
                (leaves[leafPosEnd]->value == "::"))
            leafPosEnd += 2;

          --leafPosEnd;

          leafPos = mergeRange(expType::type | expType::namespace_,
                               leafPosStart, leafPosEnd);
        }
        else
          ++leafPos;
      }
#endif
    }

    // const int [*] x
    void expNode::labelReferenceQualifiers() {
      int leafPos = 0;

      const info_t opQ = cKeywordType["*"];

      while(leafPos < leafCount) {
        expNode &leaf = *(leaves[leafPos]);

        if ((((leaf.info & expType::operator_) == 0) &&
            ((leaf.info & expType::qualifier) == 0))   ||
           ((*keywordType)[leaf.value] != opQ)) {

          ++leafPos;
          continue;
        }

        if (leafPos == 0) {
          leaf.info = (opQ & ~expType::qualifier);
          ++leafPos;
          continue;
        }

        expNode &lLeaf = *(leaves[leafPos - 1]);

        if (lLeaf.info & (expType::qualifier |
                         expType::type)) {

          leaf.info = (opQ & ~expType::operator_);
        }

        else if (lLeaf.info & expType::unknown) {
          if (sInfo) {
            if (!sInfo->hasTypeInScope(lLeaf.value))
              leaf.info = (opQ & ~expType::qualifier);
            else
              leaf.info = (opQ & ~expType::operator_);
          }
          else {
            std::cout << "sInfo is NULL\n";
          }
        }

        else if ((lLeaf.value == ",") &&
                (up == NULL)         &&
                (sInfo->info == smntType::declareStatement)) {

          leaf.info = (opQ & ~expType::operator_);
        }

        else {
          leaf.info = (opQ & ~expType::qualifier);
        }

        ++leafPos;
      }
    }

    // [(class)]
    void expNode::labelCasts() {
      // Don't mistake:
      //   int main(int) -> int main[(int)]
      if ((sInfo == NULL) ||
         (sInfo->info & smntType::functionStatement)) {

        return;
      }

      int leafPos = 0;

      while(leafPos < leafCount) {
        expNode &leaf = *(leaves[leafPos]);

        if ((leaf.value == "(")                      &&
           (leaf.leafCount)                         &&
           ((leaf[0].info & (expType::type      |
                             expType::qualifier |
                             expType::typeInfo))     ||
            sInfo->hasTypeInScope(leaves[leafPos]->value))) {

          bool isCast = true;

          for (int i = 1; i < leaf.leafCount; ++i) {
            if (!(leaf[i].info & (expType::type      |
                                 expType::qualifier |
                                 expType::typeInfo))     &&
               !sInfo->hasTypeInScope(leaves[leafPos]->value)) {

              isCast = false;
              break;
            }
          }

          if (isCast)
            leaf.info = expType::cast_;
        }

        ++leafPos;
      }
    }

    // <const int,float>
    void expNode::mergeTypes() {
      if (sInfo == NULL)
        return;

      int leafPos = 0;

      while(leafPos < leafCount) {
        if (leaves[leafPos]->info & expType::hasInfo) {
          ++leafPos;
          continue;
        }

        if (sInfo->hasTypeInScope(leaves[leafPos]->value) ||
           (leaves[leafPos]->info == expType::qualifier)) {

          varInfo &var = addVarInfoNode(leafPos);

          leaves[leafPos++]->info |= expType::type;

          const int nextLeafPos = var.loadFrom(*this, leafPos);

          removeNodes(leafPos, nextLeafPos - leafPos);
        }

        ++leafPos;
      }
    }

    // class(...), class{1,2,3}
    void expNode::mergeClassConstructs() {
    }

    // static_cast<>()
    void expNode::mergeCasts() {
    }

    // [max(a,b)]
    void expNode::mergeFunctionCalls() {
      int leafPos = 0;

      while(leafPos < leafCount) {
        if ((leaves[leafPos]->info  & expType::C) &&
           (leaves[leafPos]->value == "(")) {

          if ((leafPos) &&
             (leaves[leafPos - 1]->info & expType::function)) {
            expNode &fNode    = *(leaves[leafPos - 1]);
            expNode &argsNode = *(leaves[leafPos    ]);

            fNode.addNode(argsNode.info);

            delete fNode.leaves[0];
            fNode.leaves[0] = &argsNode;

            removeNode(leafPos);
            --leafPos;
          }
        }

        ++leafPos;
      }
    }

    void expNode::mergeArguments() {
      for (int i = 0; i < leafCount; i += 2) {
        leaves[i/2] = leaves[i];

        if ((i + 1) < leafCount)
          freeLeaf(i + 1);
      }

      leafCount = ((leafCount / 2) + 1);
    }

    // a[2]
    void expNode::mergeArrays() {
      int leafPos = 1;

      while(leafPos < leafCount) {
        expNode &preLeaf = *(leaves[leafPos - 1]);
        expNode &leaf    = *(leaves[leafPos]);

        if ((leaf.info & expType::C)  && // ( or [
           ((leaf.value == "[") ||
            (leaf.value == "("))     &&
           (preLeaf.info & (expType::typeInfo |     // that should be merged with a type, var, or ?
                            expType::varInfo  |
                            expType::unknown))) {

          leaf.info = expType::LR;

          leaf.reserveAndShift(1, 1);

          leaf.leaves[1] = &preLeaf;
          expNode::swap(leaf[0], leaf[1]);

          removeNode(leafPos - 1);
          continue;
        }

        ++leafPos;
      }
    }

    // ptr(a,b)
    void expNode::mergePointerArrays() {
      int leafPos = 0;

      while(leafPos < leafCount) {
        expNode &leaf = *(leaves[leafPos]);

        if ((leaf.info & expType::LR)             &&
           (leaf.value == "(")                   && // () operator
           (leaf[0].info & expType::varInfo)     && // varInfo
           (leaf[0].getVarInfo().pointerDepth())) {  // that is a pointer

          varInfo &var      = leaf[0].getVarInfo();
          expNode &arrNode  = leaf[1];

          const int dims     = var.dimAttr.argCount;
          const bool reorder = var.idxOrdering.size();

          expNode &csvFlatRoot = *(arrNode.makeCsvFlatHandle());
          expVector_t indices;

          OCCA_ERROR("Variable use [" << toString() << "] cannot be used without the @(dim(...)) attribute",
                     dims != 0);

          OCCA_ERROR("Variable use [" << toString() << "] has different index count of its attribute [" << var.dimAttr << "]",
                     dims == csvFlatRoot.leafCount);

          leaf.value = "[";

          if (dims == 1) {
            ++leafPos;
            continue;
          }

          for (int i = 0; i < dims; ++i) {
            if (reorder)
              indices.push_back( csvFlatRoot[ var.idxOrdering[i] ].clonePtr() );
            else
              indices.push_back( csvFlatRoot[i].clonePtr() );
          }

          expNode::freeFlatHandle(csvFlatRoot);
          arrNode.free();

          expNode *plusNode_ = &(arrNode);

          for (int i = 0; i < (dims - 1); ++i) {
            const int i2 = (reorder ? var.idxOrdering[i] : i);

            expNode &plusNode = *plusNode_;

            plusNode.info  = expType::LR;
            plusNode.value = "+";

            plusNode.addNodes(2);

            expNode &timesNode = plusNode[1];

            plusNode[0].info  = expType::C;
            plusNode[0].value = "(";

            plusNode[0].addNode( *(indices[i]) );

            timesNode.info  = expType::LR;
            timesNode.value = "*";

            timesNode.addNodes(2);

            timesNode[0].info  = expType::C;
            timesNode[0].value = "(";

            timesNode[0].addNode(*(var.dimAttr[i2].clonePtr()));

            if (i < (dims - 2)) {
              timesNode[1].info  = expType::C;
              timesNode[1].value = "(";

              timesNode[1].addNode();

              plusNode_ = &(timesNode[1][0]);
            }
            else {
              timesNode[1].info  = expType::C;
              timesNode[1].value = "(";

              timesNode[1].addNode(*(indices[i + 1]));
            }
          }
        }

        ++leafPos;
      }
    }

    // (class) x
    void expNode::mergeClassCasts() {
      int leafPos = leafCount - 2;

      while(0 <= leafPos) {
        if ( !(leaves[leafPos]->info & expType::cast_) ) {
          --leafPos;
          continue;
        }

        expNode *leaf  = leaves[leafPos];
        expNode *sLeaf = leaves[leafPos + 1];

        for (int i = (leafPos + 2); i < leafCount; ++i)
          leaves[i - 1] = leaves[i];

        --leafCount;

        leaf->addNode(*sLeaf);

        sLeaf->up = leaf;

        --leafPos;
      }
    }

    // sizeof x
    void expNode::mergeSizeOf() {
    }

    // new, new [], delete, delete []
    void expNode::mergeNewsAndDeletes() {
    }

    // throw x
    void expNode::mergeThrows() {
    }

    // [++]i
    int expNode::mergeLeftUnary(const int leafPos, const bool leftToRight) {
      const int retPos = (leftToRight ? (leafPos + 1) : (leafPos - 1));

      if (leafCount <= (leafPos + 1))
        return retPos;

      expNode *leaf  = leaves[leafPos];
      expNode *sLeaf = leaves[leafPos + 1];

      for (int i = (leafPos + 2); i < leafCount; ++i)
        leaves[i - 1] = leaves[i];

      --leafCount;

      leaf->info      = expType::L;
      leaf->leafCount = 1;
      leaf->leaves    = new expNode*[1];
      leaf->leaves[0] = sLeaf;

      sLeaf->up = leaf;

      return retPos;
    }

    // i[++]
    int expNode::mergeRightUnary(const int leafPos, const bool leftToRight) {
      const int retPos = (leftToRight ? (leafPos + 1) : (leafPos - 1));

      if (0 == leafPos)
        return retPos;

      expNode *leaf  = leaves[leafPos];
      expNode *sLeaf = leaves[leafPos - 1];

      leaves[leafPos - 1] = leaf;

      for (int i = (leafPos + 1); i < leafCount; ++i)
        leaves[i - 1] = leaves[i];

      --leafCount;

      leaf->info      = expType::R;
      leaf->leafCount = 1;
      leaf->leaves    = new expNode*[1];
      leaf->leaves[0] = sLeaf;

      sLeaf->up = leaf;

      return retPos;
    }

    // a [+] b
    int expNode::mergeBinary(const int leafPos, const bool leftToRight) {
      const int retPos = (leftToRight ? (leafPos + 1) : (leafPos - 1));

      if ((0 == leafPos) || (leafCount <= (leafPos + 1)))
        return retPos;

      expNode *leaf   = leaves[leafPos];
      expNode *sLeafL = leaves[leafPos - 1];
      expNode *sLeafR = leaves[leafPos + 1];

      leaves[leafPos - 1] = leaf;

      for (int i = (leafPos + 2); i < leafCount; ++i)
        leaves[i - 2] = leaves[i];

      leafCount -= 2;

      leaf->info      = expType::LR;
      leaf->leafCount = 2;
      leaf->leaves    = new expNode*[2];
      leaf->leaves[0] = sLeafL;
      leaf->leaves[1] = sLeafR;

      sLeafL->up = leaf;
      sLeafR->up = leaf;

      return (leftToRight ? leafPos : leafPos - 2);
    }

    // a [?] b : c
    int expNode::mergeTernary(const int leafPos, const bool leftToRight) {
      const int retPos = (leftToRight ? (leafPos + 1) : (leafPos - 1));

      if ((0 == leafPos) || (leafCount <= (leafPos + 3)))
        return retPos;

      expNode *leaf   = leaves[leafPos];
      expNode *sLeafL = leaves[leafPos - 1];
      expNode *sLeafC = leaves[leafPos + 1];
      expNode *sLeafR = leaves[leafPos + 3];

      leaves[leafPos - 1] = leaf;

      leafCount -= 4;

      for (int i = leafPos; i < leafCount; ++i)
        leaves[i] = leaves[i + 4];

      leaf->info      = expType::LCR;
      leaf->leafCount = 3;
      leaf->leaves    = new expNode*[3];
      leaf->leaves[0] = sLeafL;
      leaf->leaves[1] = sLeafC;
      leaf->leaves[2] = sLeafR;

      sLeafL->up = leaf;
      sLeafC->up = leaf;
      sLeafR->up = leaf;

      return (leftToRight ? leafPos : leafPos - 2);
    }

    //---[ Custom Type Info ]---------
    bool expNode::qualifierEndsWithStar() {
      if ( !(info & expType::qualifier) )
        return false;

      if (leafCount)
        return leaves[leafCount - 1]->qualifierEndsWithStar();
      else
        return (value == "*");
    }

    bool expNode::typeEndsWithStar() {
      if ( !(info & expType::type) ||
          (leafCount == 0) )
        return false;

      if (leaves[leafCount - 1]->info & expType::qualifier)
        return leaves[leafCount - 1]->qualifierEndsWithStar();

      return false;
    }

    bool expNode::hasAnArrayQualifier(const int pos) {
      if ( !(info & expType::qualifier) ||
          (leafCount <= pos) )
        return false;

      return ((leaves[pos]->value == "*") ||
              (leaves[pos]->value == "&") ||
              (leaves[pos]->value == "[") ||
              (leaves[pos]->value == "("));
    }

    void expNode::mergeFortranArrays() {
      if (sInfo == NULL)
        return;

      int leafPos = 0;

      while(leafPos < leafCount) {
        expNode &leaf = *(leaves[leafPos]);

        if ((leaf.info & expType::C)                       &&
           (leaf.value == "(")                            && // Is ()
           (leaf.leafCount)                               && //   and has stuff
           (0 < leafPos)                                  && //   and follows
           (leaves[leafPos - 1]->info & (expType::varInfo | //   something [-] Fortran::variable ?
                                         expType::unknown))) {

          leaf.value = "[";

          expNode &csvFlatRoot = *(leaf[0].makeCsvFlatHandle());

          const int entries = csvFlatRoot.leafCount;

          if (entries == 1) {
            if (leaf.leafCount)
              subtractOneFrom(leaf[0]);
          }
          else {
            expNode &varLeaf = *(leaves[leafPos - 1]);
            varInfo *pVar = sInfo->hasVariableInScope(varLeaf.value);

            const bool mergeEntries = ((pVar != NULL) &&
                                       (pVar->stackPointersUsed <= 1));

            if (!mergeEntries) {
              addNodes(expType::C, leafPos, (entries - 1));

              for (int i = 0; i < entries; ++i) {
                expNode &sLeaf = *(leaves[leafPos + i]);

                sLeaf.value     = "[";
                sLeaf.leafCount = 0;

                expNode entry = csvFlatRoot[i].clone();
                subtractOneFrom(entry);

                sLeaf.addNode( *(entry.clonePtr()) );
              }
            }
            else {
              varInfo &var       = *pVar;
              expNode *plusNode_ = &leaf[0];

              if (entries != var.stackPointerCount) {
                // Revert [var] back to original dimensions
                var.stackPointersUsed = var.stackPointerCount;

                // [+] Print [var] Fortran-style
                OCCA_ERROR("Incorrect dimensions on variable ["
                           << var << "], in statement ["
                           << *(leaf.up) << "]",
                           false);
              }

              expVector_t indices;

              // Feed the indices backwards
              for (int i = 0; i < entries; ++i) {
                indices.push_back( csvFlatRoot[entries - i - 1].clonePtr() );
                subtractOneFrom( *(indices.back()) );
              }

              leaf[0].free();

              for (int i = 0; i < (entries - 1); ++i) {
                expNode &plusNode = *plusNode_;

                plusNode.info  = expType::LR;
                plusNode.value = "+";

                plusNode.addNodes(2);

                expNode &timesNode = plusNode[1];

                plusNode[0].free();
                expNode::swap(plusNode[0], *(indices[i]));

                timesNode.info  = expType::LR;
                timesNode.value = "*";

                timesNode.addNodes(2);

                timesNode[0].free();
                timesNode[0] = var.stackSizeExpNode(i).clone();

                if (i < (entries - 2)) {
                  timesNode[1].info  = expType::C;
                  timesNode[1].value = "(";

                  timesNode[1].addNode();

                  plusNode_ = &(timesNode[1][0]);
                }
                else {
                  timesNode[1].free();
                  expNode::swap(timesNode[1], *(indices[i + 1]));
                }
              }
            }
              leafPos += (entries - 1);
          }

          expNode::freeFlatHandle(csvFlatRoot);
        }

        ++leafPos;
      }
    }

    void expNode::subtractOneFrom(expNode &e) {
      expNode entry;
      expNode::swap(e, entry);

      e.info  = expType::C;
      e.value = "(";

      e.addNode(expType::LR, "-");

      e[0].addNode( *(entry.clonePtr()) );
      e[0].addNode(expType::presetValue, "1");
    }

    void expNode::translateFortranMemberCalls() {
      expNode &flatRoot = *(makeDumbFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &n = flatRoot[i];

        if ((n.info  & expType::LR) &&
           (n.value == "%")) {

          n.value = ".";
        }
      }

      freeFlatHandle(flatRoot);
    }

    void expNode::translateFortranPow() {
      expNode &flatRoot = *(makeDumbFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &n = flatRoot[i];

        if ((n.info  & expType::LR) &&
           (n.value == "**")) {

          expNode &x = n[0];
          expNode &y = n[1];

          n.value     = "occaPow";
          n.info      = expType::function;
          n.leafCount = 0;

          n.addNode(expType::C, "(");

          expNode &sLeaf1 = n[0];
          sLeaf1.addNode(expType::LR, ",");

          expNode &sLeaf2 = sLeaf1[0];

          sLeaf2.addNode(x);
          sLeaf2.addNode(y);
        }
      }

      freeFlatHandle(flatRoot);
    }
    //================================

    void expNode::swap(expNode &a, expNode &b) {
      swapValues(a.sInfo, b.sInfo);

      swapValues(a.value, b.value);
      swapValues(a.info , b.info);

      swapValues(a.up, b.up);

      swapValues(a.leafCount, b.leafCount);
      swapValues(a.leaves   , b.leaves);

      if ( !(a.info & expType::hasInfo) ) {
        for (int i = 0; i < a.leafCount; ++i)
          a.leaves[i]->up = &a;
      }

      if ( !(b.info & expType::hasInfo) ) {
        for (int i = 0; i < b.leafCount; ++i)
          b.leaves[i]->up = &b;
      }

      a.setNestedSInfo(a.sInfo);
      b.setNestedSInfo(b.sInfo);
    }

    expNode expNode::clone() {
      expNode newExp;
      newExp.sInfo = sInfo;

      cloneTo(newExp);

      return newExp;
    }

    expNode expNode::clone(statement &s) {
      expNode newExp(s);

      cloneTo(newExp);

      return newExp;
    }

    expNode* expNode::clonePtr() {
      return (new expNode(clone()));
    }

    expNode* expNode::clonePtr(statement &s) {
      return (new expNode(clone(s)));
    }

    void expNode::cloneTo(expNode &newExp) {
      statement *sUp = ((newExp.sInfo != NULL) ?
                        newExp.sInfo->up       :
                        NULL);

      const bool sChanged = ((newExp.sInfo != NULL) &&
                             (newExp.sInfo != sInfo));

      newExp.info = info;

      const bool isAVarInfo  = (info & expType::varInfo);
      const bool isATypeInfo = (info & expType::typeInfo);
      const bool isAFuncInfo = ((info == (expType::function |
                                          expType::declaration)) ||
                                (info == (expType::function |
                                          expType::prototype)));

      const bool inForStatement = ((newExp.sInfo != NULL) &&
                                   (newExp.sInfo->info & smntType::forStatement));

      if (isAVarInfo | isATypeInfo | isAFuncInfo) {
        if (isAVarInfo) {
          // Var is created if it also has [expType::type]
          if (info & expType::declaration) {
            varInfo &var = newExp.addVarInfoNode();
            var = getVarInfo().clone();

            // addVarInfoNode() sets info
            newExp.info = info;

            if (sChanged) {
              if (inForStatement) {
                newExp.sInfo->addVariable(&var);
              }
              else if ((sUp != NULL) &&
                      !(sUp->hasVariableInLocalScope(var.name))) {

                sUp->addVariable(&var, newExp.sInfo);
              }
            }
          }
          else { // (info == expType::varInfo)
            varInfo &var = getVarInfo();

            newExp.putVarInfo(var);

            newExp.info = info;
          }
        }
        else if (isATypeInfo) {
          typeInfo &type = newExp.addTypeInfoNode();
          type = getTypeInfo().clone();

          if (sChanged      &&
             (sUp != NULL) &&
             !(sUp->hasVariableInLocalScope(type.name))) {

            sUp->addType(type);
          }
        }
        else if (isAFuncInfo) {
          // Get function variable
          varInfo &var    = leaves[0]->getVarInfo();
          varInfo &newVar = *(new varInfo(var.clone()));

          newExp.addVarInfoNode(0);
          newExp.setVarInfo(0, newVar);

          // Make sure we haven't initialized it
          //   from the original or an extern
          if (sChanged      &&
             (sUp != NULL) &&
             !(sUp->hasVariableInLocalScope(newVar.name)) ) {

            sUp->addVariable(&newVar);
          }

          for (int i = 0; i < newVar.argumentCount; ++i) {
            varInfo &argVar = *(new varInfo(var.getArgument(i).clone()));

            newExp.sInfo->addVariable(&argVar);
            newVar.setArgument(i, argVar);
          }
        }
      }
      else {
        newExp.value     = value;
        newExp.leafCount = leafCount;

        if (leafCount) {
          newExp.leaves = new expNode*[leafCount];

          for (int i = 0; i < leafCount; ++i) {
            newExp.leaves[i] = new expNode( newExp.makeFloatingLeaf() );
            leaves[i]->cloneTo(newExp[i]);
          }
        }
      }
    }

    expNode* expNode::lastLeaf() {
      if (leafCount != 0)
        return leaves[leafCount - 1];

      return NULL;
    }

    //---[ Exp Info ]-----------------
    int expNode::depth() {
      expNode *up_ = up;
      int depth_   = 0;

      while(up_) {
        ++depth_;
        up_ = up_->up;
      }

      return depth_;
    }

    int expNode::whichLeafAmI() {
      if (up) {
        const int upLeafCount = up->leafCount;

        for (int i = 0; i < upLeafCount; ++i)
          if (up->leaves[i] == this)
            return i;
      }

      return -1;
    }

    int expNode::nestedLeafCount() {
      if (info & expType::hasInfo)
        return 0;

      int ret = leafCount;

      for (int i = 0; i < leafCount; ++i) {
        if (leaves[i]->leafCount)
          ret += leaves[i]->nestedLeafCount();
      }

      return ret;
    }

    expNode& expNode::lastNode() {
      return *(leaves[leafCount - 1]);
    }

    expNode* expNode::makeDumbFlatHandle() {
      expNode *flatNode;

      if (sInfo != NULL)
        flatNode = new expNode(*sInfo);
      else
        flatNode = new expNode( makeFloatingLeaf() );

      const bool addMe = (info != 0);

      if ((leafCount == 0) && !addMe)
        return flatNode;

      flatNode->info   = expType::printLeaves;
      flatNode->leaves = new expNode*[addMe + nestedLeafCount()];

      int offset = 0;
      makeDumbFlatHandle(offset, flatNode->leaves);

      if (addMe)
        flatNode->setLeaf(*this, offset++);

      flatNode->leafCount = offset;

      return flatNode;
    }

    void expNode::makeDumbFlatHandle(int &offset,
                                     expNode **flatLeaves) {
      if (info & expType::hasInfo)
        return;

      for (int i = 0; i < leafCount; ++i) {
        leaves[i]->makeDumbFlatHandle(offset, flatLeaves);
        flatLeaves[offset++] = leaves[i];
      }
    }

    expNode* expNode::makeFlatHandle() {
      expNode *flatNode;

      if (sInfo != NULL)
        flatNode = new expNode(*sInfo);
      else
        flatNode = new expNode( makeFloatingLeaf() );

      const bool addMe = (info != 0);

      if ((leafCount == 0) && !addMe)
        return flatNode;

      flatNode->info   = expType::printLeaves;
      flatNode->leaves = new expNode*[addMe + nestedLeafCount()];

      int offset = 0;
      makeFlatHandle(offset, flatNode->leaves);

      if (addMe)
        flatNode->setLeaf(*this, offset++);

      flatNode->leafCount = offset;

      return flatNode;
    }

    void expNode::makeFlatHandle(int &offset,
                                 expNode **flatLeaves) {
      if (info & expType::hasInfo)
        return;

      for (int i = 0; i < leafCount; ++i) {
        switch(leaves[i]->info) {
        case (expType::L):{
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i];
          flatLeaves[offset++] = leaves[i]->leaves[0];

          break;
        }

        case (expType::R):{
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i]->leaves[0];
          flatLeaves[offset++] = leaves[i];

          break;
        }

        case (expType::LR):{
          // Assignments happen from R -> L
          const bool isUpdating = isAnAssOperator(value);

          const int a = (isUpdating ? 1 : 0);
          const int b = (isUpdating ? 0 : 1);

          leaves[i]->leaves[a]->makeFlatHandle(offset, flatLeaves);
          leaves[i]->leaves[b]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i]->leaves[a];
          flatLeaves[offset++] = leaves[i];
          flatLeaves[offset++] = leaves[i]->leaves[b];

          break;
        }

        case (expType::LCR):{
          leaves[i]->leaves[0]->makeFlatHandle(offset, flatLeaves);
          leaves[i]->leaves[1]->makeFlatHandle(offset, flatLeaves);
          leaves[i]->leaves[2]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i]->leaves[0];
          flatLeaves[offset++] = leaves[i]->leaves[1];
          flatLeaves[offset++] = leaves[i]->leaves[2];

          break;
        }
        default:
          leaves[i]->makeFlatHandle(offset, flatLeaves);
          flatLeaves[offset++] = leaves[i];

          break;
        }
      }
    }

    void expNode::freeFlatHandle(expNode &flatRoot) {
      if (flatRoot.leafCount)
        delete [] flatRoot.leaves;

      delete &flatRoot;
    }

    expNode* expNode::makeCsvFlatHandle() {
      expNode *flatNode;

      if (sInfo != NULL)
        flatNode = new expNode(*sInfo);
      else
        flatNode = new expNode( makeFloatingLeaf() );

      if ((leafCount == 0) && (info == expType::root))
        return flatNode;

      int csvCount = 1;

      for (int pass = 0; pass < 2; ++pass) {
        expNode *cNode = ((info == expType::root) ? leaves[0] : this);

        if (pass == 1) {
          flatNode->info      = expType::printLeaves;
          flatNode->leaves    = new expNode*[csvCount];
          flatNode->leafCount = csvCount;

          if (csvCount == 1) {
            flatNode->leaves[0] = cNode;
            return flatNode;
          }

          csvCount = 0;
        }

        while(cNode                         &&
              (cNode->info  &  expType::LR) &&
              (cNode->value == ",")) {

          if (pass == 0) {
            ++csvCount;
          }
          else {
            flatNode->leaves[csvCount++] = cNode->leaves[0];

            if (csvCount == (flatNode->leafCount - 1))
              flatNode->leaves[csvCount++] = cNode->leaves[1];
          }

          cNode = cNode->leaves[1];
        }
      }

      return flatNode;
    }

    void expNode::addNodes(const int count) {

      addNodes(expType::root, 0, count);
    }

    void expNode::addNodes(const int pos_,
                           const int count) {

      addNodes(expType::root, pos_, count);
    }

    void expNode::addNodes(const info_t info_,
                           const int pos_,
                           const int count) {

      const int pos = ((0 <= pos_) ? pos_ : leafCount);

      reserveAndShift(pos, count);

      for (int i = pos; i < (pos + count); ++i) {
        leaves[i] = new expNode( makeFloatingLeaf() );

        leaves[i]->info      = info_;
        leaves[i]->leafCount = 0;
        leaves[i]->leaves    = NULL;
      }
    }

    void expNode::addNode(const info_t info_,
                          const int pos) {
      addNodes(info_, pos, 1);
    }

    void expNode::addNode(const info_t info_,
                          const std::string &value_,
                          const int pos) {

      addNodes(info_, pos, 1);

      if (0 <= pos)
        leaves[pos]->value = value_;
      else
        leaves[leafCount - 1]->value = value_;
    }

    void expNode::addNode(expNode &node_,
                          const int pos_) {

      const int pos = ((0 <= pos_) ? pos_ : leafCount);

      reserveAndShift(pos, 1);

      leaves[pos] = &node_;

      node_.up = this;
    }

    int expNode::insertExpAt(expNode &exp, int pos) {
      reserveAndShift(pos, exp.leafCount);

      for (int i = pos; i < (pos + exp.leafCount); ++i)
        leaves[i] = exp.leaves[i - pos];

      return (pos + exp.leafCount);
    }

    void expNode::useExpLeaves(expNode &exp,
                               const int pos,
                               const int count) {
      reserveAndShift(0, count);

      for (int i = pos; i < (pos + count); ++i) {
        leaves[i - pos]     = exp.leaves[i];
        leaves[i - pos]->up = this;
      }

      if (sInfo)
        setNestedSInfo(*sInfo);
    }

    void expNode::copyAndUseExpLeaves(expNode &exp,
                                      const int pos,
                                      const int count) {
      reserveAndShift(0, count);

      for (int i = pos; i < (pos + count); ++i) {
        leaves[i - pos]     = exp.leaves[i]->clonePtr();
        leaves[i - pos]->up = this;
      }

      if (sInfo)
        setNestedSInfo(*sInfo);
    }

    void expNode::reserve(const int count) {
      reserveAndShift(0, count);
    }

    void expNode::reserveAndShift(const int pos,
                                  const int count) {

      expNode **newLeaves = new expNode*[leafCount + count];

      //---[ Add Leaves ]-----
      for (int i = 0; i < pos; ++i)
        newLeaves[i] = leaves[i];

      for (int i = pos; i < leafCount; ++i)
        newLeaves[i + count] = leaves[i];
      //======================

      if (leafCount)
        delete [] leaves;

      leaves = newLeaves;

      leafCount += count;
    }

    void expNode::setLeaf(expNode &leaf, const int pos) {
      leaves[pos] = &leaf;
      leaf.up     = this;
    }

    void expNode::removeNodes(int pos, const int count) {
      if (pos < 0)
        pos += leafCount;

      int removed = (((pos + count) <= leafCount) ?
                     count : (leafCount - pos));

      for (int i = (pos + removed); i < leafCount; ++i)
        leaves[i - count] = leaves[i];

      leafCount -= removed;
    }

    void expNode::removeNode(const int pos) {
      removeNodes(pos, 1);
    }

    // typeInfo
    typeInfo& expNode::addTypeInfoNode() {
      return addInfoNode<typeInfo>();
    }

    typeInfo& expNode::addTypeInfoNode(const int pos) {
      return addInfoNode<typeInfo>(expType::typeInfo, pos);
    }

    void expNode::putTypeInfo(typeInfo &type) {
      putInfo<typeInfo>(expType::typeInfo, type);
    }

    void expNode::putTypeInfo(const int pos, typeInfo &type) {
      putInfo<typeInfo>(expType::typeInfo, pos, type);
    }

    typeInfo& expNode::getTypeInfo() {
      return getInfo<typeInfo>();
    }

    typeInfo& expNode::getTypeInfo(const int pos) {
      return getInfo<typeInfo>(pos);
    }

    void expNode::setTypeInfo(typeInfo &type) {
      setInfo<typeInfo>(type);
    }

    void expNode::setTypeInfo(const int pos, typeInfo &type) {
      setInfo<typeInfo>(pos, type);
    }

    // varInfo
    varInfo& expNode::addVarInfoNode() {
      return addInfoNode<varInfo>();
    }

    varInfo& expNode::addVarInfoNode(const int pos) {
      return addInfoNode<varInfo>(expType::varInfo, pos);
    }

    void expNode::putVarInfo(varInfo &var) {
      putInfo<varInfo>(expType::varInfo, var);
    }

    void expNode::putVarInfo(const int pos, varInfo &var) {
      putInfo<varInfo>(expType::varInfo, pos, var);
    }

    varInfo& expNode::getVarInfo() {
      return getInfo<varInfo>();
    }

    varInfo& expNode::getVarInfo(const int pos) {
      return getInfo<varInfo>(pos);
    }

    void expNode::setVarInfo(varInfo &var) {
      setInfo<varInfo>(var);
    }

    void expNode::setVarInfo(const int pos, varInfo &var) {
      setInfo<varInfo>(pos, var);
    }

    bool expNode::hasVariable() {
      if (info & (expType::variable |  // [-] Fortran::variable ?
                 expType::varInfo  |
                 expType::function)) {

        if ( (info & expType::varInfo) ||
            (value.size() != 0) ) {

          return true;
        }
      }

      return false;
    }

    varInfo expNode::typeInfoOf(const std::string &str) {
      varInfo var;

      if (sInfo == NULL)
        return var;

      if (isAnInt(str)) {
        // if (isALongInt(str))
        //   var.baseType = sInfo->hasTypeInScope("long");
        // else
          var.baseType = sInfo->hasTypeInScope("int");
      }
      else if (isAFloat(str)) {
        // if (isADouble(str))
        //   var.baseType = sInfo->hasTypeInScope("double");
        // else
          var.baseType = sInfo->hasTypeInScope("float");
      }
      else if ((str == "false") || (str == "true")) {
        var.baseType = sInfo->hasTypeInScope("bool");
      }
      else if (isAString(str)) {
        var.baseType = sInfo->hasTypeInScope("char");

        var.stackPointerCount = 1;
        var.stackPointersUsed = 1;

        var.stackExpRoots        = new expNode();
        var.stackExpRoots->info  = expType::presetValue;
        var.stackExpRoots->value = occa::toString<size_t>(str.size() - 2);
      }
      else if (str == "NULL") {
        var.baseType = sInfo->hasTypeInScope("void");

        var.rightQualifiers.add("*");
        ++var.pointerCount;
      }

      return var;
    }

    varInfo expNode::evaluateType() {
      varInfo var;

      if (info & expType::hasInfo) {
        if (info & expType::varInfo)
          var.baseType = getVarInfo().baseType;
        else if (info & expType::typeInfo)
          var.baseType = &(getTypeInfo());
      }
      else if (info & expType::presetValue) {
        var = typeInfoOf(value);
      }
      else if (info & expType::operator_) {
        if (leafCount == 0)
          return var;

        if (info & expType::LR) {
          varInfo var0 = leaves[0]->evaluateType();

          if (var0.baseType == NULL)
            return var;

          if (sInfo) {
            const int thType0 = var0.baseType->thType;

            if (var0.baseType->thType & noType) {
              varInfo *funcVar = sInfo->parser.hasOperator(info,
                                                           value,
                                                           var0);

              if (funcVar != NULL)
                var = *funcVar;
            }
            else {
              return sInfo->parser.thOperatorReturnType(info,
                                                        value,
                                                        thType0);
            }
          }

          return var;
        }
        else if (info & expType::LR) {
          if (leafCount != 2)
            return var;

          varInfo var0 = leaves[0]->evaluateType();
          varInfo var1 = leaves[1]->evaluateType();

          if ((var0.baseType == NULL) ||
             (var1.baseType == NULL)) {

            return var;
          }

          if (sInfo) {
            const int thType0 = var0.baseType->thType;
            const int thType1 = var1.baseType->thType;

            if ((thType0 & noType) ||
               (thType1 & noType)) {

              varInfo *funcVar = sInfo->parser.hasOperator(info,
                                                           value,
                                                           var0, var1);

              if (funcVar != NULL)
                var = *funcVar;
            }
            else {
              return sInfo->parser.thOperatorReturnType(info,
                                                        value,
                                                        thType0, thType1);
            }
          }

          return var;
        }
        else if (info & expType::C) {
          return leaves[0]->evaluateType();
        }
        else if (info & expType::LCR) {
          if (leafCount != 3)
            return var;

          varInfo var1 = leaves[1]->evaluateType();
          varInfo var2 = leaves[2]->evaluateType();

          if ((var1.baseType == NULL) ||
             (var2.baseType == NULL)) {

            return var;
          }

          if (sInfo) {
            const int thType1 = var1.baseType->thType;
            const int thType2 = var2.baseType->thType;

            if ((thType1 & noType) ||
               (thType2 & noType)) {

              OCCA_ERROR("Oops, not implemented yet",
                         false);
            }
            else {
              return sInfo->parser.thVarInfo((thType1 < thType2) ?
                                             thType2 : thType1);
            }
          }

          return var;
        }
      }

      return var;
    }

    bool expNode::hasQualifier(const std::string &qualifier) {
      if (info & expType::varInfo) {
        return getVarInfo().hasQualifier(qualifier);
      }
      else if (info & expType::type) {
        if (!leafCount ||
           !(leaves[0]->info & expType::qualifier))
          return false;

        return leaves[0]->hasQualifier(qualifier);
      }
      else if (info & expType::qualifier) {
        if (leafCount) {
          for (int i = 0; i < leafCount; ++i) {
            if (leaves[i]->value == qualifier)
              return true;
          }

          return false;
        }
        else
          return value == qualifier;
      }
      else if (info & expType::variable) {
        OCCA_ERROR("Oops, forgot to check this",
                   false);
      }

      return false;
    }

    void expNode::removeQualifier(const std::string &qualifier) {
      if (info & expType::type) {
        if (leafCount) {
          expNode &qNode = *(leaves[0]);

          if ( !(qNode.info & expType::qualifier) )
            return;

          for (int i = 0; i < qNode.leafCount; ++i) {
            if (qNode.leaves[i]->value == qualifier) {
              qNode.removeNode(i);

              // Erase if there are no qualifiers
              if (qNode.leafCount == 0)
                removeNode(0);

              return;
            }
          }
        }
      }
    }

    int expNode::getVariableCount() {
      if (info == expType::declaration) {
        return leafCount;
      }

      return 0;
    }

    bool expNode::variableHasInit(const int pos) {
      if (info == expType::declaration) {
        const expNode &varNode = *(getVariableNode(pos));

        return (varNode.leafCount &&
                (varNode.leaves[0]->value == "="));
      }

      return false;
    }

    expNode* expNode::getVariableNode(const int pos) {
      if (info == expType::declaration) {
        return leaves[pos];
      }

      return NULL;
    }

    expNode* expNode::getVariableInfoNode(const int pos) {
      if (info == expType::declaration) {
        expNode &varNode = *(getVariableNode(pos));

        expNode *varLeaf = ((varNode.info & expType::varInfo) ?
                            &varNode :
                            varNode.leaves[0]);

        if (varLeaf->info & expType::varInfo) {
          return varLeaf;
        }
        else if (varNode.leafCount &&
                (varLeaf->value == "=")) {

          return varLeaf->leaves[0];
        }
      }

      return NULL;
    }

    expNode* expNode::getVariableOpNode(const int pos) {
      if (info == expType::declaration) {
        expNode &varNode = *(getVariableNode(pos));

        if (varNode.leafCount &&
           (varNode[0].info & expType::LR)) {

          return &(varNode[0]);
        }
      }

      return NULL;
    }

    expNode* expNode::getVariableInitNode(const int pos) {
      if (info == expType::declaration) {
        if (variableHasInit(pos)) {
          const expNode &varNode = *(getVariableNode(pos));

          const expNode *varLeaf = ((varNode.info & expType::varInfo) ?
                                    &varNode :
                                    varNode.leaves[0]);

          if (varLeaf->value == "=")
            return varLeaf->leaves[1];
        }
      }

      return NULL;
    }

    std::string expNode::getVariableName(const int pos) {
      if (info == expType::declaration) {
        expNode &leaf = *(leaves[pos]);

        if (leaf.info & expType::varInfo) {
          return leaf.getVarInfo().name;
        }
        else if (leaf.leafCount &&
                (leaf[0].value == "=")) {

          return leaf[0].getVarInfo(0).name;
        }
      }

      return "";
    }

    int expNode::getUpdatedVariableCount() {
      if (leafCount == 0)
        return 0;

      expNode *cNode = leaves[0];
      int count = 0;

      while(cNode &&
            (cNode->value == ",")) {

        if (2 <= cNode->leafCount)
          count += (isAnAssOperator((*cNode)[1].value));

        cNode = cNode->leaves[0];
      }

      if (cNode)
        count += isAnAssOperator(cNode->value);

      return count;
    }

    bool expNode::updatedVariableIsSet(const int pos) {
      expNode *n = getUpdatedNode(pos);

      if (n == NULL)
        return false;

      return ((n->info & expType::LR) &&
              isAnAssOperator(n->value));
    }

    expNode* expNode::getUpdatedNode(const int pos) {
      if (leafCount == 0)
        return NULL;

      int count = getUpdatedVariableCount();

      if (count <= pos)
        return NULL;

      expNode *cNode = leaves[0];

      while(cNode &&
            (cNode->value == ",")) {

        if (2 <= cNode->leafCount)
          count -= (isAnAssOperator((*cNode)[1].value));

        if (pos == count)
          return cNode->leaves[1];

        cNode = cNode->leaves[0];
      }

      if (cNode) {
        count -= isAnAssOperator(cNode->value);

        if (pos == count)
          return cNode;
      }

      return cNode;
    }

    expNode* expNode::getUpdatedVariableOpNode(const int pos) {
      return getUpdatedNode(pos);
    }

    expNode* expNode::getUpdatedVariableInfoNode(const int pos) {
      expNode *n = getUpdatedNode(pos);

      if (n == NULL)
        return NULL;

      return n->leaves[0];
    }

    expNode* expNode::getUpdatedVariableSetNode(const int pos) {
      expNode *n = getUpdatedNode(pos);

      if (n == NULL)
        return NULL;

      return n->leaves[1];
    }

    // [-] Fix
    int expNode::getVariableBracketCount() {
      return 0;
    }

    // [-] Fix
    expNode* expNode::getVariableBracket(const int pos) {
      // Returns the variable inside "["
      return NULL;
    }

    //  ---[ Node-based ]----------
    std::string expNode::getMyVariableName() {
      if (info & expType::varInfo) {
        return getVarInfo().name;
      }
      else if (info & expType::function) {
        return value;
      }

      return "";
    }
    //  ===========================

    //  ---[ Statement-based ]-----
    void expNode::setNestedSInfo(statement *sInfo_) {
      sInfo = sInfo_;

      if (info & expType::hasInfo)
        return;

      for (int i = 0; i < leafCount; ++i)
        leaves[i]->setNestedSInfo(sInfo_);
    }

    void expNode::setNestedSInfo(statement &sInfo_) {
      setNestedSInfo(&sInfo_);
    }
    //  ===========================
    //================================


    //---[ Analysis Info ]------------
    bool expNode::valueIsKnown(const strToStrMap_t &stsMap) {
      bool isKnown = true;

      expNode &flatRoot = *(makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &n = flatRoot[i];

        if (n.info & (expType::unknown  |
                     expType::variable | // [-] Fortran::variable ?
                     expType::function | // [-] Check function later
                     expType::varInfo)) {

          cStrToStrMapIterator it;

          if (n.info & expType::varInfo)
            it = stsMap.find(n.getVarInfo().name);
          else
            it = stsMap.find(n.value);

          if (it == stsMap.end()) {
            isKnown = false;
            break;
          }
        }
        else if ((n.info  & expType::C) && // [-] Don't load constant arrays yet
                (n.value == "[")) {

          isKnown = false;
          break;
        }
      }

      freeFlatHandle(flatRoot);

      return isKnown;
    }

    typeHolder expNode::calculateValue(const strToStrMap_t &stsMap) {
      if (valueIsKnown() == false)
        return typeHolder();

      expNode this2 = clone();

      expNode &flatRoot = *(this2.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &n = flatRoot[i];

        if (n.info & (expType::unknown  |
                     expType::variable | // [-] Fortran::variable ?
                     expType::function | // [-] Check function later
                     expType::varInfo)) {

          cStrToStrMapIterator it;

          if (n.info & expType::varInfo)
            it = stsMap.find(n.getVarInfo().name);
          else
            it = stsMap.find(n.value);

          n.info  = expType::presetValue;
          n.value = it->second;
        }
      }

      freeFlatHandle(flatRoot);

      return evaluateExpression(this2);
    }
    //================================

    void expNode::freeLeaf(const int leafPos) {
      leaves[leafPos]->free();
      delete leaves[leafPos];
    }

    void expNode::free() {
      // Let the parser free all varInfos
      if (info & expType::hasInfo) {
        delete [] leaves;
        return;
      }

      for (int i = 0; i < leafCount; ++i) {
        leaves[i]->free();
        // delete leaves[i]; [--] Segfault?
      }

      leafCount = 0;
      delete [] leaves;
    }

    void expNode::freeThis() {
      if (leafCount) {
        leafCount = 0;

        if (leaves)
          delete [] leaves;

        leaves = NULL;
      }
    }

    void expNode::print(const std::string &tab) {
      if ( !(info & expType::hasInfo) ) {

        std::cout << tab << "[" << getBits(info) << "] " << value << '\n';

        for (int i = 0; i < leafCount; ++i)
          leaves[i]->print(tab + "    ");
      }
      else if (info & expType::varInfo) {
        if (info & expType::type)
          std::cout << tab << "[VT: " << getBits(info) << "] " << getVarInfo() << '\n';
        else
          std::cout << tab << "[V: " << getBits(info) << "] " << getVarInfo().name << '\n';
      }
      else if (info & expType::typeInfo) {
        std::cout << tab << "[T: " << getBits(info) << "]\n" << getTypeInfo().toString(tab + "        ") << '\n';
      }
    }

    void expNode::printOnString(std::string &str,
                                const std::string &tab,
                                const info_t flags) {

      const bool hasSemicolon = (info & expType::hasSemicolon);
      const info_t info_      = (info & expType::removeFlags);

      switch(info_) {
      case (expType::root):{
        str += tab;

        for (int i = 0; i < leafCount; ++i)
          leaves[i]->printOnString(str);

        if (hasSemicolon)
          str += ';';

        break;
      }

      case (expType::L):{
        str += value;
        leaves[0]->printOnString(str);

        break;
      }

      case (expType::R):{
        leaves[0]->printOnString(str);
        str += value;

        break;
      }

      case (expType::LR):{
        if (startsSection(value)) {
          leaves[0]->printOnString(str);
          str += value;
          leaves[1]->printOnString(str);
          str += segmentPair(value);
        }
        else if ((value != ".") && (value != "->") && (value != ",")) {
          leaves[0]->printOnString(str);
          str += ' ';
          str += value;
          str += ' ';
          leaves[1]->printOnString(str);
        }
        else if (value == ",") {
          leaves[0]->printOnString(str);
          str += value;
          str += ' ';
          leaves[1]->printOnString(str);
        }
        else {
          leaves[0]->printOnString(str);
          str += value;
          leaves[1]->printOnString(str);
        }

        break;
      }

      case (expType::LCR):{
        leaves[0]->printOnString(str);
        str += " ? ";
        leaves[1]->printOnString(str);
        str += " : ";
        leaves[2]->printOnString(str);

        break;
      }

      case (expType::C):{
        const char startChar = value[0];

        str += startChar;

        for (int i = 0; i < leafCount; ++i)
          leaves[i]->printOnString(str);

        str += segmentPair(startChar);

        break;
      }

      case (expType::qualifier):{
        if (leafCount) {
          leaves[0]->printOnString(str);

          for (int i = 1; i < leafCount; ++i) {
            if ( !hasAnArrayQualifier(i) )
              str += ' ';

            leaves[i]->printOnString(str);
          }
        }
        else {
          str += value;
        }

        break;
      }

      case (expType::type):{
        // [const] [int] [*]
        if (leafCount) {
          leaves[0]->printOnString(str);

          for (int i = 1; i < leafCount; ++i) {
            if ( !leaves[i - 1]->hasAnArrayQualifier() )
              str += ' ';

            leaves[i]->printOnString(str);
          }

          if (leaves[leafCount - 1]->info & expType::type)
            str += ' ';
        }
        // [int]
        else {
          str += value;
        }

        break;
      }

      case (expType::presetValue):{
        str += value;

        break;
      }

      case (expType::presetValue | expType::occaKeyword):{
        str += value;

        break;
      }

      case (expType::operator_):{
        str += value;

        break;
      }

      case (expType::unknown):{
        str += value;

        break;
      }

      case (expType::unknown | expType::attribute):{
        str += value;

        break;
      }

      case (expType::variable):{
        OCCA_ERROR("Oops, forgot to check this",
                   false);

        break;
      }

      case (expType::function | expType::prototype):{
        if (leafCount) {
          str += tab;
          getVarInfo(0).printOnString(str);
          str += ";\n";
        }

        break;
      }

      case (expType::function | expType::declaration):{
        if (leafCount) {
          str += tab;
          getVarInfo(0).printOnString(str);
        }

        break;
      }

      case (expType::function):{
        str += value;

        if (leafCount)
          leaves[0]->printOnString(str);

        break;
      }

      case (expType::declaration):{
        if (leafCount) {
          // Case where a struct-type is loaded with variables:
          //   union {} a, b;
          if (leaves[0]->info & expType::typeInfo) {
            leaves[0]->printOnString(str, tab, (expFlag::noSemicolon |
                                                expFlag::noNewline));

            str += ' ';
            leaves[1]->printOnString(str);

            for (int i = 2; i < leafCount; ++i) {
              str += ", ";
              leaves[i]->printOnString(str);
            }
          }
          else {
            str += tab;
            leaves[0]->printOnString(str);

            for (int i = 1; i < leafCount; ++i) {
              str += ", ";
              leaves[i]->printOnString(str);
            }
          }

          if ( !(flags & expFlag::noSemicolon) )
            str += ';';

          if ( !(flags & expFlag::noNewline) )
            str += '\n';
        }

        break;
      }

      case (expType::struct_):{
        if (leafCount) {
          typeInfo &type = *((typeInfo*) leaves[0]->leaves[0]);
          type.printOnString(str, tab);

          if (flags & expFlag::endWithComma)
            str += ',';
          else if ( !(flags & expFlag::noSemicolon) )
            str += ';';
        }

        break;
      }

      case (expType::varInfo | expType::declaration | expType::type):
      case (expType::varInfo | expType::type):{
        getVarInfo().printOnString(str, true);

        break;
      }

      case (expType::varInfo | expType::declaration):{
        getVarInfo().printOnString(str, false);

        break;
      }

      case (expType::varInfo):{
        str += getVarInfo().name;

        break;
      }

      case (expType::typeInfo):{
        getTypeInfo().printOnString(str, tab);

        if ( !(flags & expFlag::noSemicolon) )
          str += ';';

        if ( !(flags & expFlag::noNewline) )
          str += '\n';

        break;
      }

      case (expType::cast_):{
        str += '(';
        leaves[0]->printOnString(str);
        str += ')';

        if (1 < leafCount) {
          str += ' ';
          leaves[1]->printOnString(str);
        }

        break;
      }

      case (expType::macro_):{
        str += tab;
        str += value;
        str += '\n';

        break;
      }

      case (expType::goto_):{
        str += tab;
        str += "goto ";
        str += value;
        str += ";\n";

        break;
      }

      case (expType::gotoLabel_):{
        str += tab;
        str += value;
        str += ':';

        break;
      }

      case (expType::return_):{
        str += tab;
        str += "return";

        if (leafCount)
          str += ' ';

        for (int i = 0; i < leafCount; ++i)
          leaves[i]->printOnString(str);

        str += ';';

        break;
      }

      case (expType::transfer_):{
        str += tab;
        str += value;

        if (leafCount) {
          str += ' ';

          for (int i = 0; i < leafCount; ++i)
            leaves[i]->printOnString(str);
        }

        str += ";\n";

        break;
      }

      case (expType::occaFor):{
        str += value;
        str += ' ';

        break;
      }

      case (expType::checkSInfo):{
        if (sInfo->info & smntType::updateStatement) {
          if (leafCount) {
            leaves[0]->printOnString(str, tab, (expFlag::noNewline |
                                                expFlag::noSemicolon));

            for (int i = 1; i < leafCount; ++i) {
              str += ", ";

              leaves[i]->printOnString(str, "", (expFlag::noNewline |
                                                 expFlag::noSemicolon));
            }

            str += ';';
          }
          else {
            str += tab;
            str += ';';
          }

          break;
        }

        else if (sInfo->info & smntType::flowStatement) {
          str += tab;

          if (sInfo->info & smntType::forStatement)
            str += "for(";
          else if (sInfo->info & smntType::whileStatement)
            str += "while(";
          else if (sInfo->info & smntType::ifStatement) {
            if (sInfo->info == smntType::ifStatement)
              str += "if (";
            else if (sInfo->info == smntType::elseIfStatement)
              str += "else if (";
            else
              str += "else";
          }
          else if (sInfo->info & smntType::switchStatement)
            str += "switch(";

          if (leafCount) {
            if (leaves[0]->info == expType::declaration) {
              leaves[0]->printOnString(str, "", (expFlag::noNewline |
                                                 expFlag::noSemicolon));
            }
            else {
              leaves[0]->printOnString(str);
            }

            for (int i = 1; i < leafCount; ++i) {
              str += "; ";
              leaves[i]->printOnString(str);
            }
          }

          if ( !(sInfo->info & smntType::gotoStatement) &&
              (sInfo->info != smntType::elseStatement) ) {

            str += ')';
          }
          else if (sInfo->info & smntType::gotoStatement) {
            str += ":";
          }

        }
        else if (sInfo->info & smntType::caseStatement) {
          const size_t tabChars = tab.size();

          if (2 < tabChars)
            str += tab.substr(0, tabChars - 2);

          if (leafCount) {
            str += "case ";
            leaves[0]->printOnString(str);
            str += ':';
          }
          else {
            str += "default:";
          }
        }

        break;
      }

      case (expType::asm_):{
        str += value;

        if (leafCount) {
          const char startSeg = (*this)[0].value[0];
          const char endSeg   = segmentPair(startSeg);

          str += startSeg;
          str += (*this)[0][1].value;
          str += endSeg;
        }

        break;
      }

      case (expType::printValue):{
        str += value;

        break;
      }

      case (expType::printLeaves):{
        if (leafCount) {
          for (int i = 0; i < leafCount; ++i) {
            leaves[i]->printOnString(str);
            str += ' ';
          }

          str += '\n';
        }

        break;
      }

      default:{
        str += value;
        str += ' ';
      }
      };
    }

    void expNode::printVec(expVector_t &v) {
      const int vCount = (int) v.size();

      for (int i = 0; i < vCount; ++i)
        v[i]->print();
    }

    std::ostream& operator << (std::ostream &out, expNode &n) {
      out << (std::string) n;

      return out;
    }
    //==============================================


    //---[ Statement Functions ]--------------------
    statement::statement(parserBase &pb) :
      parser(pb),
      scope(new scopeInfo()),

      info(smntType::blockStatement),

      up(NULL),

      expRoot(*this),

      statementStart(NULL),
      statementEnd(NULL) {}

    statement::statement(const statement &s) :
      parser(s.parser),
      scope(s.scope),

      info(s.info),

      up(s.up),

      expRoot(s.expRoot),

      statementStart(s.statementStart),
      statementEnd(s.statementEnd),

      attributeMap(s.attributeMap) {}


    statement::statement(const info_t info_,
                         statement *up_) :
      parser(up_->parser),
      scope(new scopeInfo()),

      info(info_),

      up(up_),

      expRoot(*this),

      statementStart(NULL),
      statementEnd(NULL) {}

    statement::~statement() {
      if (scope) {
        delete scope;
        scope = NULL;
      }
    }

    statement& statement::operator [] (const int snPos) {
      statementNode *sn = statementStart;

      for (int i = 0; i < snPos; ++i)
        sn = sn->right;

      return *(sn->value);
    }

    statement& statement::operator [] (intVector_t &path) {
      statement *s = this;

      const int pathCount = (int) path.size();

      for (int i = 0; i < pathCount; ++i)
        s = &( (*s)[path[i]] );

      return *s;
    }

    int statement::getSubIndex() {
      if (up == NULL)
        return -1;

      statementNode *sn = up->statementStart;
      int pos = 0;

      while(sn) {
        if (sn->value == this)
          return pos;

        sn = sn->right;
        ++pos;
      }

      return -1;
    }

    int statement::depth() {
      statement *up_ = up;
      int depth_     = -1;

      while(up_) {
        ++depth_;
        up_ = up_->up;
      }

      return depth_;
    }

    int statement::statementCount() {
      statementNode *sn = statementStart;
      int count = 0;

      while(sn) {
        ++count;
        sn = sn->right;
      }

      return count;
    }

    void statement::setIndexPath(intVector_t &path,
                                 statement *target) {
      int depth_ = depth();

      path.clear();
      path.reserve(depth_);

      statement *s = this;

      for (int i = 0; i < depth_; ++i) {
        path.push_back(s->getSubIndex());
        s = s->up;

        if (s == target) {
          depth_ = (i + 1);
          break;
        }
      }

      // Place in right order
      for (int i = 0; i < (depth_/2); ++i) {
        int si               = path[i];
        path[i]              = path[depth_ - i - 1];
        path[depth_ - i - 1] = si;
      }
    }

    statement* statement::makeSubStatement() {
      return new statement(0, this);
    }

    std::string statement::getTab() {
      std::string ret  = "";
      const int depth_ = depth();

      for (int i = 0; i < depth_; ++i)
        ret += "  ";

      return ret;
    }

    //---[ Find Statement ]-------------
    void statement::labelStatement(expNode &allExp,
                                   int &expPos,
                                   const int parsingLanguage) {

      info = findStatementType(allExp, expPos, parsingLanguage);
    }

    info_t statement::findStatementType(expNode &allExp,
                                        int &expPos,
                                        const int parsingLanguage) {

      if (allExp.leafCount <= expPos)
        return 0;

      if (parsingLanguage & parserInfo::parsingFortran)
        return findFortranStatementType(allExp, expPos);

      if (allExp[expPos].info & expType::macroKeyword)
        return checkMacroStatementType(allExp, expPos);

      if (isAnAttribute(allExp, expPos)) {
        expPos = skipAttribute(allExp, expPos);

        return findStatementType(allExp, expPos, parsingLanguage);
      }

      else if (allExp[expPos].info == 0)
        return 0;

      else if (allExp[expPos].info & expType::occaFor)
        return checkOccaForStatementType(allExp, expPos);

      else if (allExp[expPos].info & expType::struct_)
        return checkStructStatementType(allExp, expPos);

      else if (allExp[expPos].info & expType::C) {
        if (allExp[expPos].value == "{")
          return checkBlockStatementType(allExp, expPos);
        else if (allExp[expPos].value == "(") // Statement: (int) 3;
          return checkUpdateStatementType(allExp, expPos);
        else
          return smntType::updateStatement;
      }

      else if (allExp[expPos].info & (expType::operator_ |
                                     expType::presetValue))
        return checkUpdateStatementType(allExp, expPos);

      else if (expHasDescriptor(allExp, expPos))
        return checkDescriptorStatementType(allExp, expPos);

      else if (allExp[expPos].info & expType::unknown) {
        if (((expPos + 1) < allExp.leafCount) &&
           (allExp[expPos + 1].value == ":")) {

          return checkGotoStatementType(allExp, expPos);
        }

        return checkUpdateStatementType(allExp, expPos);
      }

      else if (allExp[expPos].info & expType::flowControl)
        return checkFlowStatementType(allExp, expPos);

      else if (allExp[expPos].info & expType::namespace_)
        return checkNamespaceStatementType(allExp, expPos);

      else if (allExp[expPos].info & expType::specialKeyword)
        return checkSpecialStatementType(allExp, expPos);

      // Statement: [;]
      else if (allExp[expPos].info & expType::endStatement)
        return checkUpdateStatementType(allExp, expPos);

      else {
        skipAfterStatement(allExp, expPos);

        return smntType::updateStatement;
      }
    }

    info_t statement::findFortranStatementType(expNode &allExp,
                                               int &expPos) {

      if (allExp.leafCount <= expPos)
        return 0;

      if (allExp[expPos].info & expType::endStatement)
        return smntType::skipStatement;

      if (allExp[expPos].info & expType::macroKeyword)
        return checkMacroStatementType(allExp, expPos);

      else if (allExp[expPos].info == 0)
        return 0;

      else if (expHasDescriptor(allExp, expPos))
        return checkFortranDescriptorStatementType(allExp, expPos);

      else if (allExp[expPos].info & expType::unknown)
        return checkFortranUpdateStatementType(allExp, expPos);

      else if (allExp[expPos].info & expType::flowControl)
        return checkFortranFlowStatementType(allExp, expPos);

      else if (allExp[expPos].info & expType::specialKeyword)
        return checkFortranSpecialStatementType(allExp, expPos);

      else {
        while((expPos < allExp.leafCount) &&
              !(allExp[expPos].info & expType::endStatement)) {

          ++expPos;
        }

        return smntType::updateStatement;
      }
    }

    info_t statement::checkMacroStatementType(expNode &allExp, int &expPos) {
      if (expPos < allExp.leafCount) {
        allExp[expPos].info = expType::macro_;
        ++expPos;
      }

      return smntType::macroStatement;
    }

    info_t statement::checkOccaForStatementType(expNode &allExp, int &expPos) {
      if (expPos < allExp.leafCount) {
        allExp[expPos].info = expType::occaFor;
        ++expPos;
      }

      return smntType::occaFor;
    }

    info_t statement::checkStructStatementType(expNode &allExp, int &expPos) {
      if (!typeInfo::statementIsATypeInfo(*this, allExp, expPos))
        return checkDescriptorStatementType(allExp, expPos);

      skipAfterStatement(allExp, expPos);

      return smntType::structStatement;
    }

    info_t statement::checkUpdateStatementType(expNode &allExp, int &expPos) {
      skipAfterStatement(allExp, expPos);

      return smntType::updateStatement;
    }

    info_t statement::checkDescriptorStatementType(expNode &allExp, int &expPos) {
      if (typeInfo::statementIsATypeInfo(*this, allExp, expPos))
        return checkStructStatementType(allExp, expPos);

      varInfo var;
      expPos = var.loadFrom(*this, allExp, expPos);

      if ((var.info & varType::functionDef) == 0)
        skipAfterStatement(allExp, expPos);

      if (var.info & varType::var)
        return smntType::declareStatement;
      else if (var.info & varType::functionDec)
        return smntType::functionPrototype;
      else
        return smntType::functionDefinition;
    }

    info_t statement::checkGotoStatementType(expNode &allExp, int &expPos) {
      if (expPos < allExp.leafCount) {
        allExp[expPos].info = expType::gotoLabel_;
        expPos += 2;
      }

      return smntType::gotoStatement;
    }

    info_t statement::checkFlowStatementType(expNode &allExp, int &expPos) {
      if (expPos < allExp.leafCount) {
        std::string &expValue = allExp[expPos].value;
        ++expPos;

        if ((expValue != "else") &&
           (expValue != "do")) {

          ++expPos;
        }

        if (expValue == "for")
          return smntType::forStatement;
        else if (expValue == "while")
          return smntType::whileStatement;
        else if (expValue == "do")
          return smntType::doWhileStatement;
        else if (expValue == "if")
          return smntType::ifStatement;
        else if (expValue == "else if")
          return smntType::elseIfStatement;
        else if (expValue == "else")
          return smntType::elseStatement;
        else if (expValue == "switch")
          return smntType::switchStatement;
      }

      OCCA_ERROR("You found the [Waldo 2] error in:\n"
                 << allExp.toString("  "),
                 false);

      return 0;
    }

    info_t statement::checkNamespaceStatementType(expNode &allExp, int &expPos) {
      // [namespace] A::B::C {
      ++expPos;

      while(expPos < allExp.leafCount) {
        expNode &leaf = allExp[expPos];

        // namespace [A::B::C] {
        if (leaf.info & expType::unknown) {
          ++expPos;

          if ((allExp.leafCount <= expPos) ||
             (allExp[expPos].value != "::")) {

            break;
          }

          ++expPos;
        }
        // namespace A::B::C [{]
        else if ((leaf.info  == expType::C) &&
           (leaf.value == "{")) {

          break;
        }
        else {
          OCCA_ERROR("Wrong namespace format",
                     false);
        }
      }

      return smntType::namespaceStatement;
    }

    info_t statement::checkSpecialStatementType(expNode &allExp, int &expPos) {
      if (allExp.leafCount <= expPos)
        return smntType::blankStatement;

      const std::string &expValue = allExp[expPos].value;

      if (expValue == "occaUnroll") {
        expPos += 2;
        return smntType::blankStatement;
      }

      const bool isCaseStatement = ((expValue == "case") ||
                                    (expValue == "default"));

      if (isCaseStatement) {
        while((expPos < allExp.leafCount) &&
              (allExp[expPos].value != ":")) {

          ++expPos;
        }

        // Skip the [:]
        if (expPos < allExp.leafCount)
          ++expPos;

        return smntType::caseStatement;
      }

      skipAfterStatement(allExp, expPos);

      return smntType::blankStatement;
    }

    info_t statement::checkBlockStatementType(expNode &allExp, int &expPos) {
      return smntType::blockStatement;
    }
    //============================================


    //  ---[ Attributes ]---------------
    attribute_t& statement::attribute(const std::string &attr) {
      return *(attributeMap[attr]);
    }

    attribute_t* statement::hasAttribute(const std::string &attr) {
      attributeMapIterator it = attributeMap.find(attr);

      if (it == attributeMap.end())
        return NULL;

      return (it->second);
    }

    void statement::addAttribute(attribute_t &attr) {
      attributeMap[attr.name] = &attr;
    }

    void statement::addAttribute(const std::string &attrSource) {
      expNode attrNode = createPlainExpNodeFrom(attrSource);

      updateAttributeMap(attributeMap, attrNode, 0);

      attrNode.free();
    }

    void statement::addAttributeTag(const std::string &attrName) {
      updateAttributeMap(attributeMap, attrName);
    }

    void statement::removeAttribute(const std::string &attr) {
      attributeMapIterator it = attributeMap.find(attr);

      if (it != attributeMap.end())
        attributeMap.erase(it);
    }

    std::string statement::attributeMapToString() {
      return parserNS::attributeMapToString(attributeMap);
    }

    void statement::printAttributeMap() {
      parserNS::printAttributeMap(attributeMap);
    }

    void statement::updateInitialLoopAttributes() {
      attributeMapIterator it = attributeMap.begin();

      if (it == attributeMap.end())
        return;

      strVector_t attributesToAdd;
      strVector_t attributesToErase;

      while(it != attributeMap.end()) {
        attribute_t &attr = *(it->second);

        if (isAnOccaTag(attr.name)) {
          info = smntType::occaFor;

          std::string loopTag, loopNest;

          // [-----][#]
          if (5 < attr.name.size()) {
            loopTag  = attr.name.substr(0,5);
            loopNest = attr.name.substr(5,1);
          }
          // [-----]
          else {
            loopTag  = attr.name;
            loopNest = "auto";
          }
          // [-] Missing outer(X), if attr has arguments

          attributesToAdd.push_back("@(occaTag  = " + loopTag  + ", "
                                    + "occaNest = " + loopNest + ")");

          attributesToErase.push_back(attr.name);

          updateOccaOMLoopAttributes(loopTag, loopNest);
        }
        else if (attr.name == "tile") {
          info = smntType::occaFor;

          const int tileDim = attr.argCount;

          OCCA_ERROR("tile() attribute can only supports 1, 2, or 3D transformations",
                     (1 <= tileDim) && (tileDim <= 3));

          std::stringstream ss;

          ss << "@(occaTag = tile, tileDim(";

          for (int i = 0; i < tileDim; ++i) {
            if (i)
              ss << ',';

            ss << attr[i];
          }

          ss << "))";

          attributesToAdd.push_back(ss.str());
          attributesToErase.push_back(attr.name);
        }

        ++it;
      }

      const int updateCount = (int) attributesToAdd.size();

      for (int i = 0; i < updateCount; ++i) {
        addAttribute(attributesToAdd[i]);
        removeAttribute(attributesToErase[i]);
      }
    }

    void statement::updateOccaOMLoopAttributes(const std::string &loopTag,
                                               const std::string &loopNest) {

      // Get outer-most loop
      statement *sOuterLoop_ = this;
      statement *sUp         = this;

      while(sUp) {
        if (sUp->info == smntType::occaFor)
          sOuterLoop_ = sUp;

        sUp = sUp->up;
      }

      statement &sOuterLoop = *sOuterLoop_;

      attribute_t *maxNestAttr = sOuterLoop.hasAttribute("occaMaxNest_" + loopTag);
      int nest = ::atoi(loopNest.c_str());

      if (maxNestAttr) {
        const std::string maxNestStr = maxNestAttr->valueStr();

        if (maxNestStr != "auto") {
          int maxNest = ::atoi(maxNestStr.c_str());

          if (maxNest < nest)
            maxNestAttr->value->value = occa::toString(nest);
        }
        else
          maxNestAttr->value->value = occa::toString(nest);
      }
      else {
        sOuterLoop.addAttribute("@(occaMaxNest_" + loopTag + " = " + loopNest + ")");
      }
    }
    //==================================

    void statement::addType(typeInfo &type) {
      scope->add(type);
    }

    void statement::addTypedef(const std::string &typedefName) {
      typeInfo &type = *(new typeInfo);
      type.name      = typedefName;

      scope->add(type);
    }

    bool statement::expHasSpecifier(expNode &allExp, int expPos) {
      return ((allExp[expPos].info & expType::type)     ||
              ((allExp[expPos].info & expType::unknown) &&
               ( hasTypeInScope(allExp[expPos].value) )));
    }

    bool statement::expHasDescriptor(expNode &allExp, int expPos) {
      if (expHasSpecifier(allExp, expPos) ||
         expHasQualifier(allExp, expPos) ||
         isAnAttribute(allExp, expPos)) {

        return true;
      }

      return false;
    }

    typeInfo* statement::hasTypeInScope(const std::string &typeName) {
      typeInfo *type = scope->hasLocalType(typeName);

      if (type != NULL)
        return type;

      if (up == NULL)
        return NULL;

      return up->hasTypeInScope(typeName);
    }

    varInfo* statement::hasVariableInScope(const std::string &varName) {
      varInfo *var = scope->hasLocalVariable(varName);

      if (var != NULL)
        return var;

      if (up == NULL)
        return NULL;

      return up->hasVariableInScope(varName);
    }

    varInfo* statement::hasVariableInLocalScope(const std::string &varName) {
      return scope->hasLocalVariable(varName);
    }

    bool statement::hasDescriptorVariable(const std::string descriptor) {
      return hasQualifier(descriptor);
    }

    bool statement::hasDescriptorVariableInScope(const std::string descriptor) {
      if (hasDescriptorVariable(descriptor))
        return true;

      if (up != NULL)
        return up->hasDescriptorVariable(descriptor);

      return false;
    }

    void statement::removeFromScope(typeInfo &type) {
      if (scope->removeLocalType(type))
        return;

      if (up)
        up->removeFromScope(type);
    }

    void statement::removeFromScope(varInfo &var) {
      if (scope->removeLocalVariable(var))
        return;

      if (up)
        up->removeFromScope(var);
    }

    void statement::removeTypeFromScope(const std::string &typeName) {
      if (scope->removeLocalType(typeName))
        return;

      if (up)
        up->removeTypeFromScope(typeName);
    }

    void statement::removeVarFromScope(const std::string &varName) {
      if (scope->removeLocalVariable(varName))
        return;

      if (up)
        up->removeVarFromScope(varName);
    }

    //---[ Loading ]--------------------
    void statement::loadAllFromNode(expNode allExp, const int parsingLanguage) {
      int expPos = 0;

      while(expPos < allExp.leafCount)
        loadFromNode(allExp, expPos, parsingLanguage);
    }

    void statement::loadFromNode(expNode allExp) {
      int expPos = 0;
      loadFromNode(allExp, expPos, parserInfo::parsingC);
    }

    void statement::loadFromNode(expNode allExp, const int parsingLanguage) {
      int expPos = 0;
      loadFromNode(allExp, expPos, parsingLanguage);
    }

    void statement::loadFromNode(expNode &allExp,
                                 int &expPos,
                                 const int parsingLanguage) {

      statement *newStatement = makeSubStatement();

      newStatement->expRoot.loadFromNode(allExp, expPos, parsingLanguage);
      const int st = newStatement->info;

      OCCA_ERROR("Not a valid statement",
                 (st & smntType::invalidStatement) == 0);

      if (st & smntType::skipStatement) {
        skipAfterStatement(allExp, expPos);

        delete newStatement;
        return;
      }

      addStatement(newStatement);

      if (st & smntType::simpleStatement) {
        newStatement->loadSimpleFromNode(st,
                                         allExp,
                                         expPos,
                                         parsingLanguage);
      }

      else if (st & smntType::flowStatement) {
        if (st & smntType::forStatement)
          newStatement->loadForFromNode(st,
                                        allExp,
                                        expPos,
                                        parsingLanguage);

        else if (st & smntType::whileStatement)
          newStatement->loadWhileFromNode(st,
                                          allExp,
                                          expPos,
                                          parsingLanguage);

        else if (st & smntType::ifStatement)
          loadIfFromNode(st,
                         allExp,
                         expPos,
                         parsingLanguage);

        else if (st & smntType::switchStatement)
          newStatement->loadSwitchFromNode(st,
                                           allExp,
                                           expPos,
                                           parsingLanguage);

        else if (st & smntType::gotoStatement)
          newStatement->loadGotoFromNode(st,
                                         allExp,
                                         expPos,
                                         parsingLanguage);
      }

      else if (st & smntType::caseStatement)
        newStatement->loadCaseFromNode(st,
                                       allExp,
                                       expPos,
                                       parsingLanguage);

      else if (st & smntType::blockStatement)
        newStatement->loadBlockFromNode(st,
                                        allExp,
                                        expPos,
                                        parsingLanguage);

      else if (st & smntType::namespaceStatement)
        newStatement->loadNamespaceFromNode(st,
                                            allExp,
                                            expPos,
                                            parsingLanguage);

      else if (st & smntType::functionStatement) {
        if (st & smntType::functionDefinition)
          newStatement->loadFunctionDefinitionFromNode(st,
                                                       allExp,
                                                       expPos,
                                                       parsingLanguage);

        else if (st & smntType::functionPrototype)
          newStatement->loadFunctionPrototypeFromNode(st,
                                                      allExp,
                                                      expPos,
                                                      parsingLanguage);
      }

      else if (st & smntType::structStatement)
        newStatement->loadStructFromNode(st,
                                         allExp,
                                         expPos,
                                         parsingLanguage);

      else if (st & smntType::blankStatement)
        newStatement->loadBlankFromNode(st,
                                        allExp,
                                        expPos,
                                        parsingLanguage);

      else if (st & smntType::macroStatement)
        newStatement->loadMacroFromNode(st,
                                        allExp,
                                        expPos,
                                        parsingLanguage);
    }

    expNode statement::createExpNodeFrom(const std::string &source) {
      pushLanguage(parserInfo::parsingC);

      expNode ret = parserNS::splitAndLabelContent(source, parserInfo::parsingC);

      ret.changeExpTypes();

      popLanguage();

      return ret;
    }

    expNode statement::createPlainExpNodeFrom(const std::string &source) {
      pushLanguage(parserInfo::parsingC);

      expNode ret = parserNS::splitAndLabelContent(source, parserInfo::parsingC);

      ret.setNestedSInfo(*this);

      ret.changeExpTypes();

      popLanguage();

      return ret;
    }

    void statement::reloadFromSource(const std::string &source) {
      pushLanguage(parserInfo::parsingC);

      expNode newExpRoot = parserNS::splitAndLabelContent(source, parserInfo::parsingC);

      int expPos = 0;

      expRoot.free();
      expRoot.loadFromNode(newExpRoot, expPos, parserInfo::parsingC);

      popLanguage();
    }

    expNode statement::createOrganizedExpNodeFrom(const std::string &source) {
      statement &tmpS = *(makeSubStatement());

      tmpS.reloadFromSource(source);

      return tmpS.expRoot;
    }

    expNode statement::createOrganizedExpNodeFrom(expNode &allExp,
                                                  const int expPos,
                                                  const int leafCount) {
      pushLanguage(parserInfo::parsingC);

      expNode ret(*(allExp.sInfo));
      ret.useExpLeaves(allExp, expPos, leafCount);

      ret.initOrganization();
      ret.organize();

      popLanguage();

      return ret;
    }

    void statement::loadSimpleFromNode(const info_t st,
                                       expNode &allExp,
                                       int &expPos,
                                       const int parsingLanguage) {
    }

    void statement::loadOneStatementFromNode(const info_t st,
                                             expNode &allExp,
                                             int &expPos,
                                             const int parsingLanguage) {

      if (allExp.leafCount <= expPos)
        return;

      if (parsingLanguage & parserInfo::parsingC) {
        if ((allExp[expPos].info  & expType::C) &&
           (allExp[expPos].value == "{")) {

          loadAllFromNode(allExp[expPos]);
          ++expPos;
        }
        else {
          loadFromNode(allExp, expPos, parsingLanguage);
        }
      }
      else {
        loadFromNode(allExp, expPos, parsingLanguage);
      }
    }

    void statement::loadForFromNode(const info_t st,
                                    expNode &allExp,
                                    int &expPos,
                                    const int parsingLanguage) {

      if (parsingLanguage & parserInfo::parsingC)
        loadOneStatementFromNode(st, allExp, expPos, parsingLanguage);
      else
        loadUntilFortranEnd(allExp, expPos);
    }

    void statement::loadWhileFromNode(const info_t st,
                                      expNode &allExp,
                                      int &expPos,
                                      const int parsingLanguage) {

      if (parsingLanguage & parserInfo::parsingC) {
        loadOneStatementFromNode(st, allExp, expPos, parsingLanguage);

        if (st != smntType::whileStatement) {
          bool skipSemicolon = false;

          // Re-use the while-loop load
          info = smntType::whileStatement;

          if (expPos < allExp.leafCount)
            skipSemicolon = (((allExp[expPos].info & expType::C) == 0) ||
                             (allExp[expPos].value               != "{"));

          expRoot.loadFromNode(allExp, expPos, parsingLanguage);

          info = smntType::doWhileStatement;

          if (skipSemicolon)
            skipAfterStatement(allExp, expPos);
        }
      }
      else {
        loadUntilFortranEnd(allExp, expPos);
      }
    }

    void statement::loadIfFromNode(const info_t st_,
                                   expNode &allExp,
                                   int &expPos,
                                   const int parsingLanguage) {

      statement *newStatement = statementEnd->value;

      if (parsingLanguage & parserInfo::parsingC) {
        newStatement->loadOneStatementFromNode(st_,
                                               allExp,
                                               expPos,
                                               parsingLanguage);

        int tmpPos     = expPos;
        info_t st      = findStatementType(allExp, tmpPos, parsingLanguage);
        info_t stCheck = smntType::elseIfStatement;

        while(true) {
          if (st != stCheck) {
            if (stCheck == smntType::elseIfStatement)
              stCheck = smntType::elseStatement;
            else
              break;
          }
          else if (allExp.leafCount <= expPos) {
            break;
          }
          else {
            newStatement = makeSubStatement();
            newStatement->expRoot.loadFromNode(allExp, expPos, parsingLanguage);

            OCCA_ERROR("Not a valid statement",
                       (st & smntType::invalidStatement) == 0);

            addStatement(newStatement);

            newStatement->loadOneStatementFromNode(st,
                                                   allExp,
                                                   expPos,
                                                   parsingLanguage);

            if (expPos < allExp.leafCount) {
              tmpPos = expPos;
              st     = findStatementType(allExp, tmpPos);
            }
          }
        }
      }
      else {
        if (newStatement->info != smntType::ifStatement) {
          newStatement->loadUntilFortranEnd(allExp, expPos);
          return;
        }

        if ((expPos < allExp.leafCount) &&
           (allExp[expPos].value == "THEN")) {

          newStatement->loadUntilFortranEnd(allExp, expPos);
        }
        else {
          newStatement->loadFromNode(allExp, expPos, parserInfo::parsingFortran);
        }
      }
    }

    // [-] Missing Fortran
    void statement::loadSwitchFromNode(const info_t st,
                                       expNode &allExp,
                                       int &expPos,
                                       const int parsingLanguage) {

      if (parsingLanguage & parserInfo::parsingC) {
        loadOneStatementFromNode(st,
                                 allExp,
                                 expPos,
                                 parsingLanguage);
      }
      else {
        loadUntilFortranEnd(allExp, expPos);
      }
    }

    // [-] Missing Fortran
    void statement::loadCaseFromNode(const info_t st,
                                     expNode &allExp,
                                     int &expPos,
                                     const int parsingLanguage) {

      if (up)
        up->loadOneStatementFromNode(up->info,
                                     allExp,
                                     expPos,
                                     parsingLanguage);
    }

    // [-] Missing Fortran
    void statement::loadGotoFromNode(const info_t st,
                                     expNode &allExp,
                                     int &expPos,
                                     const int parsingLanguage) {

    }

    void statement::loadFunctionDefinitionFromNode(const info_t st,
                                                   expNode &allExp,
                                                   int &expPos,
                                                   const int parsingLanguage) {
      if (parsingLanguage & parserInfo::parsingC) {
        if (expPos < allExp.leafCount) {
          // Most cases
          if (!isInlinedASM(allExp, expPos)) {
            loadAllFromNode(allExp[expPos], parsingLanguage);
            ++expPos;
          }
          // Case where __asm breaks the traditional {} syntax
          else {
            loadFromNode(allExp, expPos, parsingLanguage);
          }
        }
      }
      else
        return loadUntilFortranEnd(allExp, expPos);
    }

    // [-] Missing Fortran
    void statement::loadFunctionPrototypeFromNode(const info_t st,
                                                  expNode &allExp,
                                                  int &expPos,
                                                  const int parsingLanguage) {
    }

    // [-] Missing Fortran
    void statement::loadBlockFromNode(const info_t st,
                                      expNode &allExp,
                                      int &expPos,
                                      const int parsingLanguage) {

      if (expPos < allExp.leafCount) {
        loadAllFromNode(allExp[expPos], parsingLanguage);
        ++expPos;
      }
    }

    void statement::loadNamespaceFromNode(const info_t st,
                                          expNode &allExp,
                                          int &expPos,
                                          const int parsingLanguage) {

      // [namespace] [A::] [B::] [C]
      // 1            2     2     1    =   6 -> 3 (A,B,C)
      const int namespaceCount = (expRoot.leafCount / 2);
      statement *s = this;

      if (0 < namespaceCount) {
        delete scope;
        scope = getNamespace()->addNamespace(expRoot[1].value);
      }

      if (1 < namespaceCount) {
        for (int i = 1; i < namespaceCount; ++i) {
          scopeInfo *upScope = s->scope;

          s->addStatement(s->makeSubStatement());
          s = s->statementStart->value;

          s->scope = upScope->addNamespace(expRoot[1 + 2*i].value);
        }
      }

      expRoot.free();

      if (expPos < allExp.leafCount) {
        s->loadAllFromNode(allExp[expPos], parsingLanguage);
        ++expPos;

        if ((expPos < allExp.leafCount)                   &&
           (allExp[expPos].info & expType::endStatement) &&
           (allExp[expPos].value == ";")) {

          ++expPos;
        }
      }
    }

    // [-] Missing Fortran
    void statement::loadStructFromNode(const info_t st,
                                       expNode &allExp,
                                       int &expPos,
                                       const int parsingLanguage) {
    }

    // [-] Missing
    void statement::loadBlankFromNode(const info_t st,
                                      expNode &allExp,
                                      int &expPos,
                                      const int parsingLanguage) {
    }

    // [-] Missing
    void statement::loadMacroFromNode(const info_t st,
                                      expNode &allExp,
                                      int &expPos,
                                      const int parsingLanguage) {
    }

    //  ---[ Fortran ]--------
    // [+] Missing
    info_t statement::checkFortranStructStatementType(expNode &allExp, int &expPos) {
      skipUntilFortranStatementEnd(allExp, expPos);

      return smntType::structStatement;
    }

    info_t statement::checkFortranUpdateStatementType(expNode &allExp, int &expPos) {
      skipUntilFortranStatementEnd(allExp, expPos);

      return smntType::updateStatement;
    }

    info_t statement::checkFortranDescriptorStatementType(expNode &allExp, int &expPos) {
      if (((expPos + 1) < allExp.leafCount)        &&
         (allExp[expPos].value     == "IMPLICIT") &&
         (allExp[expPos + 1].value == "NONE")) {

        skipUntilFortranStatementEnd(allExp, expPos);

        return smntType::skipStatement;
      }

      varInfo var;
      var.loadFromFortran(*this, allExp, expPos);

      if ( !(var.info & varType::functionDef) )
        skipUntilFortranStatementEnd(allExp, expPos);

      if (var.info & varType::var)
        return smntType::declareStatement;
      else
        return smntType::functionDefinition;
    }

    info_t statement::checkFortranFlowStatementType(expNode &allExp, int &expPos) {
      if (expPos < allExp.leafCount)
        allExp[expPos].info = expType::checkSInfo;

      std::string &expValue = allExp[expPos].value;

      int st = 0;

      if (expValue == "DO")
        st = smntType::forStatement;
      else if (expValue == "DO WHILE")
        st = smntType::whileStatement;
      else if (expValue == "IF")
        st = smntType::ifStatement;
      else if (expValue == "ELSE IF")
        st = smntType::elseIfStatement;
      else if (expValue == "ELSE")
        st = smntType::elseStatement;
      else if (expValue == "SWITCH")
        st = smntType::switchStatement;

      // [-] Missing one-line case
      while((expPos < allExp.leafCount)     &&
            (allExp[expPos].value != "\\n") &&
            (allExp[expPos].value != ";")) {

        ++expPos;
      }

      if (expPos < allExp.leafCount)
        ++expPos;

      if (st)
        return st;

      OCCA_ERROR("You found the [Waldo 3] error in:\n"
                 << expRoot.toString("  "),
                 false);

      return 0;
    }

    info_t statement::checkFortranSpecialStatementType(expNode &allExp, int &expPos) {
      info_t retType = smntType::blankStatement;

      if (expPos < allExp.leafCount) {
        if (allExp[expPos].value == "CALL") {
          retType = smntType::updateStatement;
        }
        else if ((allExp[expPos].value == "FUNCTION") ||
                (allExp[expPos].value == "SUBROUTINE")) {

          retType = checkFortranDescriptorStatementType(allExp, expPos);
        }

        skipUntilFortranStatementEnd(allExp, expPos);
      }

      return retType;
    }

    bool statement::isFortranEnd(expNode &allExp, int &expPos) {
      if (allExp.leafCount <= expPos)
        return true;

      std::string expValue = allExp[expPos].value;

      if (info & smntType::functionDefinition) {
        const std::string &typeName = (getFunctionVar()->baseType->name);

        if (typeName == "void")
          return (expValue == "ENDSUBROUTINE");
        else
          return (expValue == "ENDFUNCTION");
      }
      else if (info & (smntType::forStatement |
                      smntType::whileStatement)) {

        return (expValue == "ENDDO");
      }
      else if (info & smntType::ifStatement) {
        if (info != smntType::elseStatement) {

          if ((expValue == "ENDIF")   ||
             (expValue == "ELSE IF") ||
             (expValue == "ELSE")) {

            return true;
          }
        }
        else
          return (expValue == "ENDIF");
      }

      return false;
    }

    void statement::loadUntilFortranEnd(expNode &allExp, int &expPos) {

      while(!isFortranEnd(allExp, expPos))
        loadFromNode(allExp, expPos, parserInfo::parsingFortran);

      // Don't skip [ELSE IF] and [ELSE]
      if ((expPos < allExp.leafCount) &&
         (allExp[expPos].value.substr(0,3) == "END")) {

        skipUntilFortranStatementEnd(allExp, expPos);
      }
    }

    void statement::skipAfterStatement(expNode &allExp, int &expPos) {
      skipUntilStatementEnd(allExp, expPos);

      if (expPos < allExp.leafCount)
        ++expPos;
    }

    void statement::skipUntilStatementEnd(expNode &allExp, int &expPos) {
      while((expPos < allExp.leafCount) &&
            !(allExp[expPos].info & expType::endStatement)) {

        ++expPos;
      }
    }

    void statement::skipUntilFortranStatementEnd(expNode &allExp, int &expPos) {
      while(expPos < allExp.leafCount) {
        if ((allExp[expPos].value == "\\n") ||
           (allExp[expPos].value == ";")) {

          break;
        }

        ++expPos;
      }
    }
    //==================================

    statement* statement::getGlobalScope() {
      statement *globalScope = this;

      while(globalScope->up)
        globalScope = globalScope->up;

      return globalScope;
    }

    scopeInfo* statement::getNamespace() {
      if (info & smntType::namespaceStatement)
        return scope;

      if (up != NULL)
        return up->getNamespace();

      return NULL;
    }

    statementNode* statement::getStatementNode() {
      if (up != NULL) {
        statementNode *ret = up->statementStart;

        while(ret) {
          if (ret->value == this)
            return ret;

          ret = ret->right;
        }
      }

      return NULL;
    }

    void statement::pushLastStatementLeftOf(statement *target) {
      if (target == NULL)
        return;

      statementNode *lastSN   = statementEnd;
      statementNode *targetSN = target->getStatementNode();

      statementEnd = statementEnd->left;

      // Only one statement
      if (statementEnd == NULL) {
        statementEnd        = statementStart;
        statementEnd->right = NULL;

        return;
      }

      statementEnd->right = NULL;

      if (statementStart->value == target)
        statementStart = lastSN;

      // Set lastSN neighbors
      lastSN->left  = targetSN->left;
      lastSN->right = targetSN;

      // Set lastSN neighbors' neighbors
      if (targetSN->left)
        targetSN->left->right = lastSN;

      targetSN->left = lastSN;
    }

    void statement::pushLastStatementRightOf(statement *target) {
      if (target == NULL)
        return;

      statementNode *lastSN   = statementEnd;
      statementNode *targetSN = target->getStatementNode();

      if (targetSN == statementEnd->left)
        return;

      statementEnd = statementEnd->left;

      // Only one statement
      if (statementEnd == NULL) {
        statementEnd        = statementStart;
        statementEnd->right = NULL;

        return;
      }

      statementEnd->right = NULL;

      // Set lastSN neighbors
      lastSN->left  = targetSN;
      lastSN->right = targetSN->right;

      // Set lastSN neighbors' neighbors
      if (targetSN->right)
        targetSN->right->left = lastSN;

      targetSN->right = lastSN;
    }

    void statement::pushLeftOf(statement *target, statement *s) {
      addStatement(s);

      pushLastStatementLeftOf(target);
    }

    void statement::pushRightOf(statement *target, statement *s) {
      addStatement(s);

      pushLastStatementRightOf(target);
    }

    statement& statement::pushNewStatementLeft(const info_t info_) {
      statement &newS = *(up->makeSubStatement());
      newS.info = info_;

      up->addStatement(&newS);

      pushLastStatementLeftOf(this);

      return newS;
    }

    statement& statement::pushNewStatementRight(const info_t info_) {
      statement &newS = *(up->makeSubStatement());
      newS.info = info_;

      up->addStatement(&newS);

      pushLastStatementRightOf(this);

      return newS;
    }

    statement& statement::createStatementFromSource(const std::string &source) {
      statementNode sn;

      pushSourceRightOf(&sn, source);

      statement &ret = *(sn.right->value);

      delete sn.right;

      return ret;
    }

    void statement::addStatementFromSource(const std::string &source) {
      pushLanguage(parserInfo::parsingC);

      expNode allExp = splitAndLabelContent(source, parserInfo::parsingC);

      loadFromNode(allExp);

      popLanguage();
    }

    void statement::addStatementsFromSource(const std::string &source) {
      pushLanguage(parserInfo::parsingC);

      loadAllFromNode(splitAndLabelContent(source, parserInfo::parsingC));

      popLanguage();
    }

    void statement::pushSourceLeftOf(statementNode *target,
                                     const std::string &source) {
      addStatementFromSource(source);

      pushLastStatementLeftOf((target == NULL) ? NULL : target->value);
    }

    void statement::pushSourceRightOf(statementNode *target,
                                      const std::string &source) {
      addStatementFromSource(source);

      pushLastStatementRightOf((target == NULL) ? NULL : target->value);
    }

    //---[ Misc ]---------------------
    bool statement::hasBarrier() {
      expNode &flatRoot = *(expRoot.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        if (flatRoot[i].value == "occaBarrier")
          return true;
      }

      expNode::freeFlatHandle(flatRoot);

      return false;
    }

    bool statement::hasStatementWithBarrier() {
      if (hasBarrier())
        return true;

      statementNode *statementPos = statementStart;

      while(statementPos) {
        statement &s = *(statementPos->value);

        if (s.hasBarrier())
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    // Guaranteed to work with statements under a globalScope
    statement& statement::greatestCommonStatement(statement &s) {
      std::vector<statement*> path[2];

      for (int pass = 0; pass < 2; ++pass) {
        statement *cs = ((pass == 0) ? this : &s);

        while(cs) {
          path[pass].push_back(cs);
          cs = cs->up;
        }
      }

      const int dist0   = (int) path[0].size();
      const int dist1   = (int) path[1].size();
      const int minDist = ((dist0 < dist1) ? dist0 : dist1);

      for (int i = 1; i <= minDist; ++i) {
        if (path[0][dist0 - i] != path[1][dist1 - i])
          return *(path[0][dist0 - i + 1]);
      }

      return *(path[0][dist0 - minDist]);
    }

    unsigned int statement::distToForLoop() {
      return distToStatementType(smntType::forStatement);
    }

    unsigned int statement::distToOccaForLoop() {
      statement *s = this;

      unsigned int dist = 0;

      while(s) {
        if (s->info == smntType::occaFor)
          return dist;

        s = s->up;
        ++dist;
      }

      return -1; // Maximum distance
    }

    unsigned int statement::distToStatementType(const info_t info_) {
      statement *s = this;

      unsigned int dist = 0;

      while(s) {
        if (s->info == info_)
          return dist;

        s = s->up;
        ++dist;
      }

      return -1; // Maximum distance
    }

    bool statement::insideOf(statement &s) {
      statement *up_ = up;

      while(up_ != NULL) {
        if (up_ == &s)
          return true;

        up_ = up_->up;
      }

      return false;
    }

    void statement::setStatementIdMap(statementIdMap_t &idMap) {
      int startID = 0;

      setStatementIdMap(idMap, startID);
    }

    void statement::setStatementIdMap(statementIdMap_t &idMap,
                                      int &startID) {

      statementNode *nodePos = statementStart;

      while(nodePos) {
        statement &s = *(nodePos->value);
        idMap[&s] = startID++;

        s.setStatementIdMap(idMap, startID);

        nodePos = nodePos->right;
      }
    }

    void statement::setStatementVector(statementVector_t &vec,
                                       const bool init) {

      statementNode *nodePos = statementStart;

      if (init)
        vec.clear();

      while(nodePos) {
        statement &s = *(nodePos->value);

        vec.push_back(&s);

        s.setStatementVector(vec, false);

        nodePos = nodePos->right;
      }
    }

    void statement::setStatementVector(statementIdMap_t &idMap,
                                       statementVector_t &vec) {

      statementIdMapIterator it = idMap.begin();

      const int statementCount_ = idMap.size();

      vec.clear();
      vec.resize(statementCount_);

      for (int i = 0; i < statementCount_; ++i) {
        vec[ it->second ] = (it->first);

        ++it;
      }
    }
    //================================

    void statement::checkIfVariableIsDefined(varInfo &var,
                                             statement *origin) {
      if ((var.name.size() == 0)     ||
         var.hasQualifier("extern") ||
         (var.info & varType::functionDef)) {

        return;
      }

      varInfo *scopeVar = scope->hasLocalVariable(var.name);

      OCCA_ERROR("Variable [" << var.name << "] defined in:\n"
                 << *origin
                 << "is already defined in:\n"
                 << *this,
                 scopeVar == NULL);
    }

    varInfo& statement::addVariable(varInfo &var,
                                    statement *origin) {
      varInfo &newVar = *(new varInfo);
      newVar = var.clone();

      addVariable(&newVar, origin);

      return newVar;
    }

    void statement::addVariable(varInfo *var,
                                statement *origin_) {
      if (var->name.size() == 0)
        return;

      statement *origin = (origin_ == NULL ? this : origin_);

      checkIfVariableIsDefined(*var, origin);

      scope->add(*var);

      parser.varOriginMap[var] = origin;
    }

    // Swap variable varInfo*
    void statement::replaceVarInfos(varToVarMap_t &v2v) {
      expNode &flatRoot = *(expRoot.makeDumbFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        if (leaf.info & expType::varInfo) {
          varToVarMapIterator it = v2v.find(&leaf.getVarInfo());

          if (it != v2v.end())
            leaf.setVarInfo(*(it->second));
        }
      }

      expNode::freeFlatHandle(flatRoot);

      statementNode *sn = statementStart;

      while(sn) {
        statement &s = *(sn->value);

        s.replaceVarInfos(v2v);

        sn = sn->right;
      }
    }

    void statement::addStatement(statement *newStatement) {
      newStatement->up = this;

      if (statementStart != NULL) {
        statementEnd = statementEnd->push(newStatement);
      }
      else {
        statementStart = new statementNode(newStatement);
        statementEnd   = statementStart;
      }
    }

    void statement::removeStatement(statement &s) {
      statementNode *sn = statementStart;

      if (sn == NULL)
        return;

      if (sn->value == &s) {
        statementStart = statementStart->right;

        if (sn == statementEnd)
          statementEnd = NULL;

        delete sn;

        return;
      }

      while(sn) {
        if (sn->value == &s) {
          if (sn == statementEnd)
            statementEnd = statementEnd->left;

          delete sn->pop();

          return;
        }

        sn = sn->right;
      }
    }

    void statement::swap(statement &a, statement &b) {
      swapValues(a.info, b.info);

      expNode::swap(a.expRoot, b.expRoot);

      swapValues(a.scope, b.scope);

      a.attributeMap.swap(b.attributeMap);
    }

    void statement::swapPlaces(statement &a, statement &b) {
      statementNode *aSN = b.getStatementNode();
      statementNode *bSN = a.getStatementNode();

      if (aSN != NULL) aSN->value = &a;
      if (bSN != NULL) bSN->value = &b;

      swapValues(a.up, b.up);

      for (int pass = 0; pass < 2; ++pass) {
        statementNode *sn = ((pass == 0)      ?
                             a.statementStart :
                             b.statementStart);

        statementNode *upSN = ((pass == 0) ?
                               aSN         :
                               bSN);

        while(sn) {
          sn->up = upSN;
          sn     = sn->right;
        }
      }
    }

    void statement::swapStatementNodesFor(statement &a, statement &b) {
      statementNode *aSN = b.getStatementNode();
      statementNode *bSN = a.getStatementNode();

      if (aSN != NULL) aSN->value = &a;
      if (bSN != NULL) bSN->value = &b;

      swapValues(a.up, b.up);

      for (int pass = 0; pass < 2; ++pass) {
        statement *up_ = ((pass == 0) ?
                          &b          :
                          &a);

        statementNode *sn = ((pass == 0)      ?
                             a.statementStart :
                             b.statementStart);

        statementNode *upSN = ((pass == 0) ?
                               aSN         :
                               bSN);

        while(sn) {
          sn->up        = upSN;
          sn->value->up = up_;

          sn = sn->right;
        }
      }

      swapValues(a.statementStart, b.statementStart);
      swapValues(a.statementEnd  , b.statementEnd);
    }

    statement* statement::clone(statement *up_) {
      statement *newStatement;

      if (up_) {
        newStatement = new statement(info, up_);
      }
      else if (up) {
        newStatement = new statement(info, up);
      }
      else {
        newStatement = new statement(parser);
      }

      expRoot.cloneTo(newStatement->expRoot);

      newStatement->statementStart = NULL;
      newStatement->statementEnd   = NULL;

      statementNode *sn = statementStart;

      while(sn) {
        newStatement->addStatement( sn->value->clone(newStatement) );
        sn = sn->right;
      }

      newStatement->attributeMap = attributeMap;

      return newStatement;
    }

    void statement::printVariablesInScope() {
      if (up)
        up->printVariablesInScope();

      printVariablesInLocalScope();
    }

    void statement::printVariablesInLocalScope() {
      varMapIterator it = scope->varMap.begin();

      while(it != scope->varMap.end()) {
        std::cout << "  " << *(it->second) << '\n';

        ++it;
      }
    }

    void statement::printTypesInScope() {
      if (up)
        up->printTypesInScope();

      printTypesInStatement();
    }

    void statement::printTypesInStatement() {
      typeMapIterator it = scope->typeMap.begin();

      while(it != scope->typeMap.end()) {
        std::cout << (it->first) << '\n';

        ++it;
      }
    }

    //---[ Statement Info ]-----------
    void statement::createUniqueVariables(std::vector<std::string> &names,
                                          const info_t flags) {
      std::stringstream ss;

      const int nameCount = names.size();
      int iterCount = 0;

      while(true) {
        if (flags & statementFlag::updateByNumber)
          ss << iterCount++;
        else
          ss << "_";

        const std::string &suffix = ss.str();

        for (int i = 0; i < nameCount; ++i) {
          if (hasVariableInScope(names[i] + suffix))
            break;

          if ((i + 1) == nameCount) {
            for (int j = 0; j < nameCount; ++j)
              names[j] += suffix;

            return;
          }
        }

        if (flags & statementFlag::updateByNumber)
          ss.str("");
      }
    }

    void statement::createUniqueSequentialVariables(std::string &varName,
                                                    const int varCount) {
      std::stringstream ss;

      // Find unique baseName
      while(true) {
        int v;

        for (v = 0; v < varCount; ++v) {
          ss << v;

          if (hasVariableInLocalScope(varName + ss.str()))
            break;

          ss.str("");
        }

        if (v == varCount)
          break;

        varName += '_';
      }
    }

    void statement::swapExpWith(statement &s) {
      expNode::swap(expRoot, s.expRoot);
    }

    bool statement::hasQualifier(const std::string &qualifier) {
      if (info & smntType::declareStatement) {
        varInfo &var = getDeclarationVarInfo(0);
        return var.hasQualifier(qualifier);
      }
      else if (info & smntType::functionStatement) {
        varInfo &var = expRoot.getVarInfo(0);
        return var.hasQualifier(qualifier);
      }
      else if (info & smntType::forStatement) {
        if (expRoot.leafCount) {
          expNode &node1 = *(expRoot.leaves[0]);

          if ((node1.leafCount) &&
             (node1.leaves[0]->info & expType::type)) {

            return node1.leaves[0]->hasQualifier(qualifier);
          }
        }

        return false;
      }

      return false;
    }

    void statement::addQualifier(const std::string &qualifier,
                                 const int pos) {
      if (hasQualifier(qualifier))
        return;

      if (info & smntType::declareStatement) {
        varInfo &var = getDeclarationVarInfo(0);
        var.addQualifier(qualifier);
      }
      else if (info & smntType::functionStatement) {
        varInfo &var = expRoot.getVarInfo(0);
        var.addQualifier(qualifier, pos);
      }
    }

    void statement::removeQualifier(const std::string &qualifier) {
      if (!hasQualifier(qualifier))
        return;

      if (info & smntType::declareStatement) {
        varInfo &var = getDeclarationVarInfo(0);
        var.removeQualifier(qualifier);
      }
      else if (info & smntType::functionStatement) {
      }
      else if (info & smntType::forStatement) {
      }
    }

    varInfo& statement::getDeclarationVarInfo(const int pos) {
      expNode *varNode = expRoot.getVariableInfoNode(pos);
      return varNode->getVarInfo();
    }

    expNode* statement::getDeclarationVarNode(const int pos) {
      if (info & smntType::declareStatement)
        return expRoot.leaves[pos];

      return NULL;
    }

    std::string statement::getDeclarationVarName(const int pos) {
      if (info & smntType::declareStatement) {
        varInfo &var = getDeclarationVarInfo(pos);
        return var.name;
      }

      return "";
    }

    expNode* statement::getDeclarationVarInitNode(const int pos) {
      if (info & smntType::declareStatement)
        return expRoot.getVariableInitNode(pos);

      return NULL;
    }

    int statement::getDeclarationVarCount() {
      if (info & smntType::declareStatement)
        return expRoot.leafCount;

      return 0;
    }

    varInfo* statement::getFunctionVar() {
      if (info & smntType::functionStatement) {
        return &(expRoot.getVarInfo(0));
      }
      else if (info & smntType::updateStatement) {
        statement *s = up;

        while(s &&
              !(s->info & smntType::functionStatement)) {
          s = s->up;
        }

        if (s)
          return s->getFunctionVar();

        return NULL;
      }

      OCCA_ERROR(false, "Not added yet");

      return NULL;
    }

    void statement::setFunctionVar(varInfo &var) {
      if (info & smntType::functionStatement) {
        expRoot.setVarInfo(0, var);
      }
      else if (info & smntType::updateStatement) {
        statement *s = up;

        while(s &&
              !(s->info & smntType::functionStatement)) {
          s = s->up;
        }

        if (s)
          s->setFunctionVar(var);
      }
    }

    std::string statement::getFunctionName() {
      if (info & smntType::functionStatement) {
        return getFunctionVar()->name;
      }

      OCCA_ERROR(false, "Not added yet");

      return "";
    }

    void statement::setFunctionName(const std::string &newName) {
      if (info & smntType::functionStatement) {
        getFunctionVar()->name = newName;
        return;
      }

      OCCA_ERROR(false, "Not added yet");
    }

    bool statement::functionHasQualifier(const std::string &qName) {
      if (info & smntType::functionStatement) {
        return getFunctionVar()->hasQualifier(qName);
      }

      OCCA_ERROR(false, "Not added yet");

      return false;
    }

    int statement::getFunctionArgCount() {
      if (info & smntType::functionStatement) {
        return getFunctionVar()->argumentCount;
      }

      return 0;
    }

    std::string statement::getFunctionArgType(const int pos) {
      if (info & smntType::functionDefinition) {
        return getFunctionVar()->baseType->name;
      }

      return "";
    }

    std::string statement::getFunctionArgName(const int pos) {
      if (info & smntType::functionDefinition) {
        return getFunctionVar()->getArgument(pos).name;
      }

      return "";
    }

    varInfo* statement::getFunctionArgVar(const int pos) {
      if (info & smntType::functionDefinition) {
        return &(getFunctionVar()->getArgument(pos));
      }

      return NULL;
    }

    bool statement::hasFunctionArgVar(varInfo &var) {
      if (info & smntType::functionDefinition) {
        const int argc = getFunctionArgCount();

        for (int i = 0; i < argc; ++i) {
          if (&var == getFunctionArgVar(i))
            return true;
        }

        return false;
      }

      return false;
    }

    void statement::addFunctionArg(const int pos, varInfo &var) {
      if ( !(info & smntType::functionStatement) )
        return;

      getFunctionVar()->addArgument(pos, var);
    }

    expNode* statement::getForStatement(const int pos) {
      if (info & smntType::forStatement)
        return expRoot.leaves[pos];

      return NULL;
    }

    void statement::addForStatement() {
      expRoot.addNode();
    }

    int statement::getForStatementCount() {
      if (info & smntType::forStatement)
        return expRoot.leafCount;

      return 0;
    }
    //================================

    void statement::printDebugInfo() {
      std::cout << "[" << getBits(info) << "] s = " << expRoot
                << ' ' << attributeMapToString() << '\n';

      expRoot.print();
    }

    void statement::printOnString(std::string &str,
                                  const info_t flags) {
      std::string tab;

      if (flags & statementFlag::printSubStatements)
        tab = getTab();

      // OCCA For's
      if ((info == smntType::occaFor) &&
         (expRoot.leafCount == 0)) {

        if ( !(flags & statementFlag::printSubStatements) ) {
          str += expRoot.value;
          return;
        }

        str += tab + expRoot.value + " {\n";

        printSubsOnString(str);

        str += tab;
        str += "}\n";
      }

      else if (info & smntType::declareStatement) {
        expRoot.printOnString(str, tab);
      }

      else if (info & (smntType::simpleStatement | smntType::gotoStatement)) {
        if (flags & statementFlag::printSubStatements) {
          expRoot.printOnString(str, tab);
          str += '\n';
        }
        else {
          expRoot.printOnString(str);
        }
      }

      else if (info & smntType::flowStatement) {
        if ( !(flags & statementFlag::printSubStatements) ) {
          expRoot.printOnString(str);
          return;
        }

        if (info != smntType::doWhileStatement) {
          str += expRoot.toString(tab);

          if (statementStart != NULL) {
            str += " {";
          }
          else {
            str += '\n';
            str += tab;
            str += "  ;";
          }

          str += '\n';
        }
        else {
          str += tab;
          str += "do {\n";
        }

        printSubsOnString(str);

        if (statementStart != NULL) {
          str += tab;
          str += '}';
        }

        if (info != smntType::doWhileStatement) {
          str += '\n';
        }
        else {
          str += ' ';
          expRoot.printOnString(str);
          str += ";\n";
        }
      }

      else if (info & smntType::caseStatement) {
        if (flags & statementFlag::printSubStatements) {
          expRoot.printOnString(str, tab);
          str += '\n';
        }
        else {
          expRoot.printOnString(str);
        }
      }

      else if (info & smntType::functionStatement) {
        if (info & smntType::functionDefinition) {
          if ( !(flags & statementFlag::printSubStatements) ) {
            expRoot.printOnString(str);
            return;
          }

          expRoot.printOnString(str, tab);

          if (statementCount() == 1) {
            expNode &e = statementStart->value->expRoot;

            if (e.info & expType::asm_) {
              str += ' ';
              e.printOnString(str);
              str += ";\n";

              return;
            }
          }

          str += " {\n";

          printSubsOnString(str);

          str += tab;

          if (back(str) != '\n')
            str += "\n}\n\n";
          else
            str += "}\n\n";
        }
        else if (info & smntType::functionPrototype) {
          expRoot.printOnString(str, tab);
          str += '\n';
        }
      }
      else if (info & smntType::blockStatement) {
        if ( !(flags & statementFlag::printSubStatements) ) {
          str += "{}";
          return;
        }

        str += tab;
        str += "{\n";

        printSubsOnString(str);

        if (back(str) != '\n')
          str += '\n';

        str += tab;
        str += "}\n";
      }
      else if (info & smntType::structStatement) {
        if (flags & statementFlag::printSubStatements) {
          expRoot.printOnString(str, tab);
          str += '\n';
        }
        else {
          expRoot.printOnString(str);
        }
      }
      else if (info & smntType::namespaceStatement) {
        if (scope->isTheGlobalScope()) {
          printSubsOnString(str);
          return;
        }

        if (back(str) != '\n')
          str += '\n';

        str += tab;
        str += "namespace ";

        if (0 < scope->name.size()) {
          str += scope->name;
          str += ' ';
        }

        str += "{\n";

        printSubsOnString(str);

        str += tab;
        str += "}\n";
      }
      else if (info & smntType::macroStatement) {
        str += expRoot.value;

        if (flags & statementFlag::printSubStatements)
          str += '\n';
      }
      else {
        expRoot.printOnString(str, tab);
      }
    }

    void statement::printSubsOnString(std::string &str) {
      statementNode *statementPos = statementStart;

      while(statementPos) {
        statementPos->value->printOnString(str);
        statementPos = statementPos->right;
      }
    }

    std::ostream& operator << (std::ostream &out, statement &s) {
      out << (std::string) s;

      return out;
    }
    //============================================

    bool isAnOccaTag(const std::string &tag) {
      return (isAnOccaInnerTag(tag) ||
              isAnOccaOuterTag(tag));
    }

    bool isAnOccaInnerTag(const std::string &tag) {
      if ( (tag.find("inner") == std::string::npos) ||
          ((tag != "inner")  &&
           (tag != "inner0") &&
           (tag != "inner1") &&
           (tag != "inner2")) ) {

        return false;
      }

      return true;
    }

    bool isAnOccaOuterTag(const std::string &tag) {
      if ( (tag.find("outer") == std::string::npos) ||
          ((tag != "outer")  &&
           (tag != "outer0") &&
           (tag != "outer1") &&
           (tag != "outer2")) ) {

        return false;
      }

      return true;
    }
  }
}
