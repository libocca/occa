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

#include "occa/parser/types.hpp"
#include "occa/parser/parser.hpp"
#include "occa/tools/misc.hpp"
#include "occa/tools/string.hpp"

namespace occa {
  namespace parserNS {
    bool isInlinedASM(const std::string &attrName) {
      return (attrName == "__asm");
    }

    bool isInlinedASM(expNode &expRoot, int leafPos) {
      if (expRoot.leafCount <= leafPos)
        return false;

      return isInlinedASM(expRoot[leafPos].value);
    }


    //---[ Scope Info Class ]---------------------
    scopeInfo::scopeInfo() :
      up(NULL) {}

    void scopeInfo::appendVariablesFrom(scopeInfo *scope) {
      if (scope == NULL)
        return;

      varMap.insert(scope->varMap.begin(),
                    scope->varMap.end());
    }

    void scopeInfo::add(scopeInfo &scope) {
      scopeMap[scope.name] = &scope;
    }

    void scopeInfo::add(typeInfo &type) {
      typeMap[type.name] = &type;
    }

    void scopeInfo::add(varInfo &var) {
      varMap[var.name] = &var;
    }

    scopeInfo* scopeInfo::addNamespace(const std::string &namespaceName) {
      scopeInfo *&scope = scopeMap[namespaceName];

      if (scope != NULL)
        return scope;

      scope       = new scopeInfo();
      scope->up   = this;
      scope->name = namespaceName;

      return scope;
    }

    typeInfo* scopeInfo::hasLocalType(const std::string &typeName) {
      typeMapIterator it = typeMap.find(typeName);

      if (it != typeMap.end())
        return it->second;

      return NULL;
    }

    varInfo* scopeInfo::hasLocalVariable(const std::string &varName) {
      varMapIterator it = varMap.find(varName);

      if (it != varMap.end())
        return it->second;

      return NULL;
    }

    bool scopeInfo::removeLocalType(const std::string &typeName) {
      // For readability
      const bool removedType = true;

      typeMapIterator it = typeMap.find(typeName);

      if (it != typeMap.end()) {
        typeMap.erase(it);
        return removedType;
      }

      return !removedType;
    }

    bool scopeInfo::removeLocalVariable(const std::string &varName) {
      // For readability
      const bool removedVar = true;

      varMapIterator it = varMap.find(varName);

      if (it != varMap.end()) {
        varMap.erase(it);
        return removedVar;
      }

      return !removedVar;
    }

    bool scopeInfo::removeLocalType(typeInfo &type) {
      return removeLocalType(type.name);
    }

    bool scopeInfo::removeLocalVariable(varInfo &var) {
      return removeLocalVariable(var.name);
    }

    void scopeInfo::printOnString(std::string &str) {
      if (up) {
        up->printOnString(str);
        str += "::";
      }

      str += name;
    }
    //============================================


    //---[ Attribute Class ]----------------------
    bool isAnAttribute(const std::string &attrName) {
      return ((attrName == "@") ||
              (attrName == "__attribute__"));
    }

    bool isAnAttribute(expNode &expRoot, int leafPos) {
      if (expRoot.leafCount <= leafPos)
        return false;

      return isAnAttribute(expRoot[leafPos].value);
    }

    int skipAttribute(expNode &expRoot, int leafPos) {
      if (!isAnAttribute(expRoot, leafPos))
        return leafPos;

      const std::string &attrTag = expRoot[leafPos].value;
      ++leafPos;

      if ((attrTag == "@")              &&
         (leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value != "(")) {

        ++leafPos;
      }

      if ((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "(")) {

        ++leafPos;
      }

      return leafPos;
    }

    attribute_t::attribute_t() :
      argCount(0),
      args(NULL),

      value(NULL) {}

    attribute_t::attribute_t(expNode &e) :
      argCount(0),
      args(NULL),

      value(NULL) {

      load(e);
    }

    attribute_t::attribute_t(const attribute_t &attr) :
      name(attr.name),

      argCount(attr.argCount),
      args(attr.args),

      value(attr.value) {}

    attribute_t& attribute_t::operator = (const attribute_t &attr) {
      name = attr.name;

      argCount = attr.argCount;
      args     = attr.args;

      value = attr.value;

      return *this;
    }

    void attribute_t::load(expNode &e) {
      const int attrIsSet = (e.value == "=");

      expNode &attrNode = (attrIsSet ? e[0] : e);

      if (startsSection(attrNode.value))
        loadVariable(attrNode);
      else
        name = attrNode.value;

      if (!attrIsSet)
        return;

      value = e[1].clonePtr();
    }

    void attribute_t::loadVariable(expNode &e) {
      name = e[0].value;

      expNode &csvFlatRoot = *(e[1].makeCsvFlatHandle());

      argCount = csvFlatRoot.leafCount;

      if (argCount) {
        args = new expNode*[argCount];

        for (int i = 0; i < argCount; ++i)
          args[i] = csvFlatRoot[i].clonePtr();
      }

      expNode::freeFlatHandle(csvFlatRoot);
    }

    expNode& attribute_t::operator [] (const int pos) {
      return *(args[pos]);
    }

    std::string attribute_t::argStr(const int pos) {
      return args[pos]->toString();
    }

    expNode& attribute_t::valueExp() {
      return *value;
    }

    std::string attribute_t::valueStr() {
      return value->toString();
    }

    attribute_t::operator std::string() {
      std::string ret;

      ret += name;

      if (argCount) {
        ret += "(";

        for (int i = 0; i < argCount; ++i) {
          if (i)
            ret += ", ";

          ret += argStr(i);
        }

        ret += ")";
      }

      if (value) {
        ret += " = ";
        ret += valueStr();
      }

      return ret;
    }

    void updateAttributeMap(attributeMap_t &attributeMap,
                            const std::string &attrName) {

      attribute_t &attr = *(new attribute_t());
      attr.name = attrName;

      attributeMap[attrName] = &attr;
    }

    int updateAttributeMap(attributeMap_t &attributeMap,
                           expNode &expRoot,
                           int leafPos) {

      while(true) {
        const int leafPos2 = leafPos;

        leafPos = updateAttributeMapR(attributeMap, expRoot, leafPos);

        if (leafPos == leafPos2)
          break;
      }

      return leafPos;
    }

    int updateAttributeMapR(attributeMap_t &attributeMap,
                            expNode &expRoot,
                            int leafPos) {

      if (!isAnAttribute(expRoot, leafPos))
        return leafPos;

      if (expRoot[leafPos].value == "__attribute__") {
        ++leafPos;

        if (leafPos < expRoot.leafCount) {
          expNode &tmp = *(expRoot[leafPos].clonePtr());
          tmp.changeExpTypes();
          tmp.organize();

          attribute_t &attr = *(new attribute_t());
          attr.name = tmp.toString();

          attributeMap["__attribute__"] = &attr;

          tmp.free();

          ++leafPos;
        }

        return leafPos;
      }

      ++leafPos;

      // Only one attribute
      if ((expRoot[leafPos].info & expType::C) == 0) {
        expNode attrNode;

        expRoot[leafPos].info |= expType::attribute;

        const int leafStart = leafPos;

        if (((leafPos + 1) < expRoot.leafCount) &&
           (expRoot[leafPos + 1].info & expType::C)) {

          ++leafPos;
        }

        ++leafPos;

        attrNode.copyAndUseExpLeaves(expRoot,
                                     leafStart, (leafPos - leafStart));
        attrNode.organize();

        attribute_t &attr = *(new attribute_t(attrNode[0]));

        attributeMap[attr.name] = &attr;

        attrNode.free();

        return leafPos;
      }
      else {
        expNode attrRoot = expRoot[leafPos].clone();

        for (int i = 0; i < attrRoot.leafCount; ++i) {
          if (attrRoot[i].info & expType::unknown)
            attrRoot[i].info |= expType::attribute;
        }

        attrRoot.organize();

        expNode &csvFlatRoot = *(attrRoot[0].makeCsvFlatHandle());

        const int attributeCount = csvFlatRoot.leafCount;

        for (int i = 0; i < attributeCount; ++i) {
          expNode &attrNode = csvFlatRoot[i];

          attribute_t &attr = *(new attribute_t(attrNode));

          attributeMap[attr.name] = &attr;
        }

        attrRoot.free();
        expNode::freeFlatHandle(csvFlatRoot);
      }

      return (leafPos + 1);
    }

    void printAttributeMap(attributeMap_t &attributeMap) {
      if (attributeMap.size())
        std::cout << attributeMapToString(attributeMap) << '\n';
    }

    std::string attributeMapToString(attributeMap_t &attributeMap) {
      std::string ret;

      if (attributeMap.size() == 0)
        return ret;

      attributeMapIterator it = attributeMap.begin();

      const bool putParentheses = ((1 < attributeMap.size()) ||
                                   (it->second->value != NULL));

      ret += '@';

      if (putParentheses)
        ret += '(';

      bool oneAttrSet = false;

      while(it != attributeMap.end()) {
        if (oneAttrSet)
          ret += ", ";
        else
          oneAttrSet = true;

        ret += (std::string) *(it->second);

        ++it;
      }

      if (putParentheses)
        ret += ')';

      return ret;
    }
    //==================================


    //---[ Qualifier Info Class ]-----------------
    qualifierInfo::qualifierInfo() :
      qualifierCount(0),
      qualifiers(NULL) {}

    qualifierInfo::qualifierInfo(const qualifierInfo &q) :
      qualifierCount(q.qualifierCount),
      qualifiers(q.qualifiers) {}

    qualifierInfo& qualifierInfo::operator = (const qualifierInfo &q) {
      qualifierCount = q.qualifierCount;
      qualifiers     = q.qualifiers;

      return *this;
    }

    void qualifierInfo::free() {
      if (qualifiers) {
        qualifierCount = 0;

        delete [] qualifiers;
        qualifiers = NULL;
      }
    }

    qualifierInfo qualifierInfo::clone() {
      qualifierInfo q;

      q.qualifierCount = qualifierCount;

      if (qualifierCount) {
        q.qualifiers = new std::string[qualifierCount];

        for (int i = 0; i < qualifierCount; ++i)
          q.qualifiers[i] = qualifiers[i];
      }

      return q;
    }

    int qualifierInfo::loadFrom(expNode &expRoot,
                                int leafPos) {

      OCCA_ERROR("Cannot load [qualifierInfo] without a statement",
                 expRoot.sInfo != NULL);

      varInfo var;

      return loadFrom(*(expRoot.sInfo), var, expRoot, leafPos);
    }

    int qualifierInfo::loadFrom(varInfo &var,
                                expNode &expRoot,
                                int leafPos) {

      OCCA_ERROR("Cannot load [qualifierInfo] without a statement",
                 expRoot.sInfo != NULL);

      return loadFrom(*(expRoot.sInfo), var, expRoot, leafPos);
    }

    int qualifierInfo::loadFrom(statement &s,
                                expNode &expRoot,
                                int leafPos) {

      varInfo var;

      return loadFrom(s, var, expRoot, leafPos);
    }

    int qualifierInfo::loadFrom(statement &s,
                                varInfo &var,
                                expNode &expRoot,
                                int leafPos) {

      if (expRoot.leafCount <= leafPos)
        return leafPos;

      qualifierCount = 0;

      const int leafRoot = leafPos;

      for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
          if (qualifierCount)
            qualifiers = new std::string[qualifierCount];
          else
            break;
        }

        qualifierCount = 0;
        leafPos = leafRoot;

        while(leafPos < expRoot.leafCount) {
          if (expHasQualifier(expRoot, leafPos)) {
            if (pass == 1) {
              if ((expRoot[leafPos].value == "*") &&
                 hasImplicitInt()) {
                break;
              }

              qualifiers[qualifierCount] = expRoot[leafPos].value;
            }

            ++qualifierCount;
            ++leafPos;
          }
          else if (isAnAttribute(expRoot, leafPos)) {
            const bool is__attribute__ = (expRoot[leafPos].value == "__attribute__");

            if (pass == 0) {
              leafPos = skipAttribute(expRoot, leafPos);
            }
            else {
              leafPos = updateAttributeMap(var.attributeMap, expRoot, leafPos);

              if (is__attribute__) {
                attributeMapIterator it = var.attributeMap.find("__attribute__");
                attribute_t &attr       = *(it->second);

                qualifiers[qualifierCount] = ("__attribute__" + attr.name);

                var.attributeMap.erase(it);
              }
            }

            if (is__attribute__)
              ++qualifierCount;
          }
          else
            break;
        }
      }

      return leafPos;
    }

    int qualifierInfo::loadFromFortran(statement &s,
                                       varInfo &var,
                                       expNode &expRoot,
                                       int leafPos) {
      if (expRoot.leafCount <= leafPos)
        return leafPos;

      while(true) {
        int newLeafPos = updateFortranVar(s, var, expRoot, leafPos);

        if (newLeafPos == leafPos)
          break;

        if (leafPos < expRoot.leafCount) {
          if (expRoot[newLeafPos].value == ",") {
            ++newLeafPos;
          }
          else if (expRoot[newLeafPos].value == "::") {
            leafPos = newLeafPos + 1;
            break;
          }
        }

        leafPos = newLeafPos;
      }

      return leafPos;
    }

    bool qualifierInfo::fortranVarNeedsUpdate(varInfo &var,
                                              const std::string &fortranQualifier) {
      // Normal Fortran
      if (fortranQualifier == "POINTER") {
        ++(var.pointerCount);
        return true;
      }
      else if (fortranQualifier == "VOLATILE") {
        add("volatile");
        return true;
      }
      else if (fortranQualifier == "PARAMETER") {
        add("occaConst", 0);
        return true;
      }
      // OFL Keywords
      if (fortranQualifier == "KERNEL") {
        add("occaKernel");
        return true;
      }
      if (fortranQualifier == "DEVICE") {
        add("occaFunction");
        return true;
      }
      if (fortranQualifier == "SHARED") {
        add("occaShared");
        return true;
      }
      if (fortranQualifier == "EXCLUSIVE") {
        add("exclusive");
        return true;
      }

      return false;
    }

    int qualifierInfo::updateFortranVar(statement &s,
                                        varInfo &var,
                                        expNode &expPos,
                                        const int leafPos) {
      if (fortranVarNeedsUpdate(var, expPos[leafPos].value))
        return (leafPos + 1);

      if (expPos[leafPos].info & expType::type) {
        int nextLeafPos = leafPos;

        std::string typeName = varInfo::getFullFortranType(expPos, nextLeafPos);
        var.baseType = s.hasTypeInScope(typeName);

        return nextLeafPos;
      }
      else {
        const std::string &value = expPos[leafPos].value;

        if (value == "INTENT") {
          expNode *leaf = expPos.leaves[leafPos + 1];

          if (leaf && (leaf->leafCount)) {
            leaf = leaf->leaves[0];

            var.leftQualifiers.add("INTENT" + upString(leaf->value));
            var.rightQualifiers.add("&", 0);

            if (upStringCheck(leaf->value, "IN"))
              add("occaConst", 0);

            return (leafPos + 2);
          }

          return (leafPos + 1);
        }
        else if (value == "DIMENSION") {
          var.leftQualifiers.add("DIMENSION");
          return var.loadStackPointersFromFortran(expPos, leafPos + 1);
        }
      }

      return leafPos;
    }

    //---[ Qualifier Info ]-------------
    int qualifierInfo::has(const std::string &qName) {
      int count = 0;

      for (int i = 0; i < qualifierCount; ++i) {
        if (qualifiers[i] == qName)
          ++count;
      }

      return count;
    }

    std::string& qualifierInfo::get(const int pos) {
      OCCA_ERROR("There are only ["
                 << qualifierCount << "] qualifiers (asking for ["
                 << pos << "])",
                 (0 <= pos) && (pos < qualifierCount));

      return qualifiers[pos];
    }

    void qualifierInfo::add(const std::string &qName,
                            int pos) {
      std::string *newQualifiers = new std::string[qualifierCount + 1];

      if (pos < 0)
        pos = qualifierCount;

      for (int i = 0; i < pos; ++i)
        newQualifiers[i] = qualifiers[i];

      newQualifiers[pos] = qName;

      for (int i = pos; i < qualifierCount; ++i)
        newQualifiers[i + 1] = qualifiers[i];

      delete [] qualifiers;

      qualifiers = newQualifiers;
      ++qualifierCount;
    }

    void qualifierInfo::remove(const std::string &qName) {
      for (int i = 0; i < qualifierCount; ++i) {
        if (qualifiers[i] == qName) {
          remove(i);
          return;
        }
      }
    }

    void qualifierInfo::remove(const int pos,
                               const int count) {
      for (int i = (pos + count); i < qualifierCount; ++i)
        qualifiers[i - count] = qualifiers[i];

      qualifierCount -= count;

      if ((qualifierCount == 0) &&
         (count != 0)) {

        delete [] qualifiers;
        qualifiers = NULL;
      }
    }

    void qualifierInfo::clear() {
      if (qualifierCount) {
        qualifierCount = 0;

        delete [] qualifiers;
        qualifiers = NULL;
      }
    }

    bool qualifierInfo::hasImplicitInt() {
      return (has("unsigned") ||
              has("signed")   ||
              has("short")    ||
              has("long"));
    }
    //==================================

    void qualifierInfo::printOnString(std::string &str,
                                      varInfo *var) {

      for (int i = 0; i < qualifierCount; ++i) {
        str += qualifiers[i];

        // Don't put the space if the variable doesn't have a name
        if ((i == (qualifierCount - 1)) &&
           (var != NULL)               &&
           (var->name.size() == 0)) {

          return;
        }

        const bool referenceType = ((qualifiers[i][0] == '*') ||
                                    (qualifiers[i][0] == '&'));

        if (!referenceType) {
          str += ' ';
        }
        else if ((i + 1) < qualifierCount) {
          const bool referenceType2 = ((qualifiers[i + 1][0] == '*') ||
                                       (qualifiers[i + 1][0] == '&'));

          if (!referenceType2)
            str += ' ';
        }
      }
    }

    bool expHasQualifier(expNode &allExp, int expPos) {
      return (allExp[expPos].info & expType::qualifier);
    }
    //============================================


    //---[ Type Info Class ]----------------------
    typeInfo::typeInfo() :
      typeScope(NULL),

      leftQualifiers(),

      name(""),

      thType(noType),

      nestedInfoCount(0),
      nestedExps(NULL),

      typedefHasDefinition(false),
      typedefing(NULL),
      baseType(NULL),

      typedefVar(NULL) {}

    typeInfo::typeInfo(const typeInfo &type) :
      typeScope(type.typeScope),

      leftQualifiers(type.leftQualifiers),

      name(type.name),

      thType(type.thType),

      nestedInfoCount(type.nestedInfoCount),
      nestedExps(type.nestedExps),

      typedefHasDefinition(type.typedefHasDefinition),
      typedefing(type.typedefing),
      baseType(type.baseType),

      typedefVar(type.typedefVar),

      opOverloadMaps(type.opOverloadMaps) {}

    typeInfo& typeInfo::operator = (const typeInfo &type) {
      typeScope = type.typeScope;

      leftQualifiers = type.leftQualifiers;

      name = type.name;

      thType = type.thType;

      nestedInfoCount = type.nestedInfoCount;
      nestedExps      = type.nestedExps;

      typedefHasDefinition = type.typedefHasDefinition;
      typedefing           = type.typedefing;
      baseType             = type.baseType;

      typedefVar = type.typedefVar;

      opOverloadMaps = type.opOverloadMaps;

      return *this;
    }

    typeInfo typeInfo::clone() {
      typeInfo c = *this;

      c.leftQualifiers = leftQualifiers.clone();

      if (nestedInfoCount) {
        c.nestedExps = new expNode[nestedInfoCount];

        for (int i = 0; i < nestedInfoCount; ++i)
          nestedExps[i].cloneTo(c.nestedExps[i]);
      }

      if (typedefVar) {
        c.typedefVar  = new varInfo;
        *c.typedefVar = typedefVar->clone();
      }

      return c;
    }

    //---[ Load Info ]------------------
    int typeInfo::loadFrom(expNode &expRoot,
                           int leafPos,
                           bool addTypeToScope) {

      OCCA_ERROR("Cannot load [typeInfo] without a statement",
                 expRoot.sInfo != NULL);

      return loadFrom(*(expRoot.sInfo),
                      expRoot,
                      leafPos,
                      addTypeToScope);
    }

    int typeInfo::loadFrom(statement &s,
                           expNode &expRoot,
                           int leafPos,
                           bool addTypeToScope) {

      if (expRoot.leafCount <= leafPos)
        return leafPos;

      // Typedefs are pre-loaded with qualifiers
      if (leftQualifiers.size() == 0)
        leafPos = leftQualifiers.loadFrom(s, expRoot, leafPos);

      if (leftQualifiers.has("typedef")) {
        leafPos = loadTypedefFrom(s, expRoot, leafPos);

        if (addTypeToScope &&
           (s.up != NULL)) {

          s.up->addType(*this);
        }

        return leafPos;
      }

      baseType = this;

      if ((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].info & expType::unknown)) {

        name = expRoot[leafPos++].value;

        if (addTypeToScope &&
           (s.up != NULL)) {

          s.up->addType(*this);
        }

        updateThType();
      }
      else if (hasImplicitInt()) {
        name     = "int";
        baseType = s.hasTypeInScope("int");
      }

      if ((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "{")) {

        expNode &leaf = expRoot[leafPos++];

        if (leftQualifiers.has("enum")) {
          nestedInfoCount = 1;
          nestedExps      = new expNode[nestedInfoCount];

          nestedExps[0] = leaf.clone();
          nestedExps[0].organizeDeclareStatement(expFlag::none);

          return leafPos;
        }

        nestedInfoCount = delimiterCount(leaf, ";");
        nestedExps      = new expNode[nestedInfoCount];

        int sLeafPos = 0;

        for (int i = 0; i < nestedInfoCount; ++i) {
          int sNextLeafPos = nextDelimiter(leaf, sLeafPos, ";");

          // Empty statements
          if (sNextLeafPos != sLeafPos) {
            const bool loadType = typeInfo::statementIsATypeInfo(s,
                                                                 leaf,
                                                                 sLeafPos);

            sNextLeafPos = leaf.mergeRange(expType::root,
                                           sLeafPos,
                                           sNextLeafPos);

            expNode::swap(nestedExps[i], leaf[sLeafPos]);

            // For nested types, types are still
            //   labeled as expType::unknown
            nestedExps[i].labelReferenceQualifiers();

            if (!loadType)
              nestedExps[i].organizeDeclareStatement(expFlag::none);
            else
              nestedExps[i].organizeStructStatement();

            leaf.setLeaf(nestedExps[i], sLeafPos);
          }
          else {
            --i;
            --nestedInfoCount;
            ++sNextLeafPos;
          }

          sLeafPos = sNextLeafPos;
        }
      }

      return leafPos;
    }

    int typeInfo::loadTypedefFrom(statement &s,
                                  expNode &expRoot,
                                  int leafPos) {
      leftQualifiers.remove("typedef");

      if ((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value != "{")) {

        typeInfo *leafType = s.hasTypeInScope(expRoot[leafPos].value);

        if (leafType) {
          typedefing = leafType;

          ++leafPos;
        }
        else {
          if (!leftQualifiers.hasImplicitInt()) {
            typedefing           = new typeInfo;
            typedefing->name     = expRoot[leafPos].value;
            typedefing->baseType = typedefing;

            ++leafPos;
          }
          else {
            typedefing = s.hasTypeInScope("int");
          }
        }
      }

      if ((leafPos < expRoot.leafCount) &&
         (expRoot[leafPos].value == "{")) {

        // Anonymous type
        if (typedefing == NULL) {
          typedefing           = new typeInfo;
          typedefing->baseType = typedefing;

          std::string transferQuals[4] = {"class", "enum", "union", "struct"};

          for (int i = 0; i < 4; ++i) {
            if (leftQualifiers.has(transferQuals[i])) {
              leftQualifiers.remove(transferQuals[i]);
              typedefing->leftQualifiers.add(transferQuals[i]);
            }
          }
        }

        typedefing->loadFrom(s, expRoot, leafPos);
        ++leafPos;

        typedefHasDefinition = true;
      }

      baseType = typedefing->baseType;

      varInfo typedefVarInfo;
      typedefVarInfo.baseType = typedefing;

      typedefVar = new varInfo;
      leafPos = typedefVar->loadFrom(s, expRoot, leafPos, &typedefVarInfo);

      name = typedefVar->name;

      updateThType();

      return leafPos;
    }

    void typeInfo::updateThType() {
      if (name == "bool")
        thType = boolType;
      else if (name == "char")
        thType = charType;
      else if (name == "float")
        thType = floatType;
      else if (name == "double")
        thType = doubleType;
      else {
        if (name == "short") {
          const bool unsigned_ = hasQualifier("unsigned");

          thType = (unsigned_ ? ushortType : shortType);
        }
        else if ((name == "int") ||
                (name == "long")) {

          const bool unsigned_ = hasQualifier("unsigned");
          const int longs_     = hasQualifier("long");

          switch(longs_) {
          case 0:
            thType = (unsigned_ ? uintType      : intType);
          case 1:
            thType = (unsigned_ ? ulongType     : longType);
          default:
            thType = (unsigned_ ? ulonglongType : longlongType);
          }
        }
        else
          thType = noType;
      }
    }

    bool typeInfo::statementIsATypeInfo(statement &s,
                                        expNode &expRoot,
                                        int leafPos) {
      if (expRoot.leafCount == 0)
        return false;

      qualifierInfo qualifiers;

      leafPos = qualifiers.loadFrom(s, expRoot, leafPos);

      if (qualifiers.has("typedef"))
        return true;

      if (qualifiers.hasImplicitInt())
        return false;

      if (leafPos < expRoot.leafCount) {
        if ((expRoot[leafPos].info & expType::unknown) &&
           (!s.hasTypeInScope(expRoot[leafPos].value))) {

          return true;
        }

        if (expRoot[leafPos].value == "{")
          return true;
      }

      return false;
    }

    int typeInfo::delimiterCount(expNode &expRoot,
                                 const char *delimiter) {
      int count = 0;

      for (int i = 0; i < expRoot.leafCount; ++i) {
        if (expRoot[i].value == delimiter)
          ++count;
      }

      return count;
    }

    int typeInfo::nextDelimiter(expNode &expRoot,
                                int leafPos,
                                const char *delimiter) {
      for (int i = leafPos; i < expRoot.leafCount; ++i) {
        if (expRoot[i].value == delimiter)
          return i;
      }

      return expRoot.leafCount;
    }
    //==================================


    //---[ Type Info ]------------------
    int typeInfo::hasQualifier(const std::string &qName) {
      return leftQualifiers.has(qName);
    }

    void typeInfo::addQualifier(const std::string &qName,
                                int pos) {
      leftQualifiers.add(qName, pos);
    }

    bool typeInfo::hasImplicitInt() {
      return leftQualifiers.hasImplicitInt();
    }

    int typeInfo::pointerDepth() {
      if (typedefing)
        return typedefVar->pointerDepth();

      return 0;
    }
    //==================================


    //---[ Class Info ]---------------
    varInfo* typeInfo::hasOperator(const std::string &name_) {
      return NULL;
    }
    //================================

    void typeInfo::printOnString(std::string &str,
                                 const std::string &tab) {

      if (typedefing) {
        str += tab;
        str += "typedef ";
        leftQualifiers.printOnString(str);

        if (typedefHasDefinition)
          typedefing->printOnString(str);
        else
          str += typedefing->name;

        str += ' ';
        typedefVar->printOnString(str, false);
      }
      else {
        const bool isAnEnum = leftQualifiers.has("enum");

        str += tab;
        leftQualifiers.printOnString(str);
        str += name;

        if (nestedInfoCount) {
          if (name.size())
            str += ' ';

          str += '{';
          str += '\n';

          for (int i = 0; i < nestedInfoCount; ++i) {
            if (!isAnEnum) {
              str += nestedExps[i].toString(tab + "  ");
            }
            else {
              if (i < (nestedInfoCount - 1)) {
                str += nestedExps[i].toString(tab + "  ", (expFlag::noSemicolon |
                                                           expFlag::endWithComma));
              }
              else {
                str += nestedExps[i].toString(tab + "  ", expFlag::noSemicolon);
              }
            }

            if (back(str) != '\n')
              str += '\n';
          }

          str += tab;
          str += '}';
        }
      }
    }
    //============================================


    //---[ Variable Info Class ]------------------
    varInfo::varInfo() :
      scope(NULL),

      info(0),

      leftQualifiers(),
      rightQualifiers(),

      baseType(NULL),

      name(""),

      pointerCount(0),

      stackPointerCount(0),
      stackPointersUsed(0),
      stackExpRoots(NULL),

      bitfieldSize(-1),

      usesTemplate(false),
      tArgCount(0),
      tArgs(NULL),

      argumentCount(0),
      argumentVarInfos(NULL),

      functionNestCount(0),
      functionNests(NULL) {}

    varInfo::varInfo(const varInfo &var) :
      scope(var.scope),

      info(var.info),

      attributeMap(var.attributeMap),
      leftQualifiers(var.leftQualifiers),
      rightQualifiers(var.rightQualifiers),

      baseType(var.baseType),

      name(var.name),

      pointerCount(var.pointerCount),

      stackPointerCount(var.stackPointerCount),
      stackPointersUsed(var.stackPointersUsed),
      stackExpRoots(var.stackExpRoots),

      bitfieldSize(var.bitfieldSize),

      dimAttr(var.dimAttr),
      idxOrdering(var.idxOrdering),

      usesTemplate(var.usesTemplate),
      tArgCount(var.tArgCount),
      tArgs(var.tArgs),

      argumentCount(var.argumentCount),
      argumentVarInfos(var.argumentVarInfos),

      functionNestCount(var.functionNestCount),
      functionNests(var.functionNests) {}

    varInfo& varInfo::operator = (const varInfo &var) {
      scope = var.scope;

      info = var.info;

      attributeMap    = var.attributeMap;
      leftQualifiers  = var.leftQualifiers;
      rightQualifiers = var.rightQualifiers;

      baseType = var.baseType;

      name = var.name;

      pointerCount = var.pointerCount;

      stackPointerCount  = var.stackPointerCount;
      stackPointersUsed  = var.stackPointersUsed;
      stackExpRoots      = var.stackExpRoots;

      bitfieldSize = var.bitfieldSize;

      dimAttr     = var.dimAttr;
      idxOrdering = var.idxOrdering;

      usesTemplate = var.usesTemplate;
      tArgCount    = var.tArgCount;
      tArgs        = var.tArgs;

      argumentCount    = var.argumentCount;
      argumentVarInfos = var.argumentVarInfos;

      functionNestCount = var.functionNestCount;
      functionNests     = var.functionNests;

      return *this;
    }

    varInfo varInfo::clone() {
      varInfo v = *this;

      v.attributeMap    = attributeMap;
      v.leftQualifiers  = leftQualifiers.clone();
      v.rightQualifiers = rightQualifiers.clone();

      if (stackPointerCount) {
        v.stackExpRoots = new expNode[stackPointerCount];

        for (int i = 0; i < stackPointerCount; ++i)
          stackExpRoots[i].cloneTo(v.stackExpRoots[i]);
      }

      if (tArgCount) {
        v.tArgs = new typeInfo*[tArgCount];

        for (int i = 0; i < tArgCount; ++i)
          v.tArgs[i] = new typeInfo(tArgs[i]->clone());
      }

      if (argumentCount) {
        v.argumentVarInfos = new varInfo*[argumentCount];

        for (int i = 0; i < argumentCount; ++i)
          v.argumentVarInfos[i] = new varInfo(argumentVarInfos[i]->clone());
      }

      if (functionNestCount) {
        v.functionNests = new varInfo[functionNestCount];

        for (int i = 0; i < functionNestCount; ++i)
          v.functionNests[i] = functionNests[i].clone();
      }

      return v;
    }

    varInfo* varInfo::clonePtr() {
      return new varInfo(clone());
    }

    int varInfo::variablesInStatement(expNode &expRoot) {
      int argc = 0;

      for (int i = 0; i < expRoot.leafCount; ++i) {
        if ((expRoot[i].value == ",") ||
           (expRoot[i].value == ";")) {

          ++argc;
        }
        else if (i == (expRoot.leafCount - 1))
          ++argc;
      }

      return argc;
    }

    //---[ Load Info ]------------------
    int varInfo::loadFrom(expNode &expRoot,
                          int leafPos,
                          varInfo *varHasType) {

      OCCA_ERROR("Cannot load [varInfo] without a statement",
                 expRoot.sInfo != NULL);

      return loadFrom(*(expRoot.sInfo), expRoot, leafPos, varHasType);
    }

    int varInfo::loadFrom(statement &s,
                          expNode &expRoot,
                          int leafPos,
                          varInfo *varHasType) {

      if (expRoot.leafCount <= leafPos)
        return leafPos;

      leafPos = loadTypeFrom(s, expRoot, leafPos, varHasType);

      info = getVarInfoFrom(s, expRoot, leafPos);

      if (info & varType::functionPointer) {
        functionNestCount = getNestCountFrom(expRoot, leafPos);
        functionNests     = new varInfo[functionNestCount];
      }

      leafPos = loadNameFrom(s, expRoot, leafPos);
      leafPos = loadArgsFrom(s, expRoot, leafPos);

      if ((leafPos < expRoot.leafCount) &&
         expRoot[leafPos].value == ":") {

        ++leafPos;

        if (leafPos < expRoot.leafCount) {
          typeHolder th(expRoot[leafPos].value);

          OCCA_ERROR("Bitfield is not known at compile-time",
                     th.type != noType);

          bitfieldSize = th.to<int>();

          ++leafPos;
        }
      }

      leafPos = updateAttributeMap(attributeMap, expRoot, leafPos);

      setupAttributes();

      organizeExpNodes();

      return leafPos;
    }

    int varInfo::loadTypeFrom(statement &s,
                              expNode &expRoot,
                              int leafPos,
                              varInfo *varHasType) {

      if (expRoot.leafCount <= leafPos)
        return leafPos;

      if (varHasType == NULL) {
        leafPos = leftQualifiers.loadFrom(s, expRoot, leafPos);

        if (leafPos < expRoot.leafCount) {
          baseType = s.hasTypeInScope(expRoot[leafPos].value);

          if (baseType)
            ++leafPos;
        }
        else if (leftQualifiers.has("unsigned"))
          baseType = s.hasTypeInScope("int");
      }
      else {
        leftQualifiers = varHasType->leftQualifiers.clone();
        baseType       = varHasType->baseType;
      }

      leafPos = rightQualifiers.loadFrom(s, expRoot, leafPos);

      for (int i = 0; i < rightQualifiers.qualifierCount; ++i) {
        if (rightQualifiers[i] == "*")
          ++pointerCount;
      }

      return leafPos;
    }

    int varInfo::getVarInfoFrom(statement &s,
                                expNode &expRoot,
                                int leafPos) {
      // No name var (argument for function)
      if (expRoot.leafCount <= leafPos)
        return varType::var;

      const int nestCount = getNestCountFrom(expRoot, leafPos);

      if (nestCount)
        return varType::functionPointer;

      ++leafPos;

      if (expRoot.leafCount <= leafPos)
        return varType::var;

      if (expRoot[leafPos].value == "(") {
        ++leafPos;

        if (expRoot.leafCount <= leafPos)
          return varType::functionDec;

        if ((expRoot[leafPos].value == "{") ||
           isInlinedASM(expRoot, leafPos)) {

          return varType::functionDef;
        }

        return varType::functionDec;
      }

      return varType::var;
    }

    int varInfo::getNestCountFrom(expNode &expRoot,
                                  int leafPos) {
      if (expRoot.leafCount <= leafPos)
        return 0;

      int nestCount = 0;

      expNode *leaf = expRoot.leaves[leafPos];

      while((leaf->value == "(") &&
            (leaf->leafCount != 0)) {

        if ((leaf->leaves[0]->value == "*") ||
           (leaf->leaves[0]->value == "^")) {

          ++nestCount;

          if (1 < leaf->leafCount)
            leaf = leaf->leaves[1];
          else
            break;
        }
        else
          leaf = leaf->leaves[0];
      }

      return nestCount;
    }

    int varInfo::loadNameFrom(statement &s,
                              expNode &expRoot,
                              int leafPos) {
      if (expRoot.leafCount <= leafPos)
        return leafPos;

      if (nodeHasName(expRoot, leafPos))
        return loadNameFromNode(expRoot, leafPos);

      expNode *expRoot2 = &expRoot;
      int leafPos2      = leafPos;
      expNode *leaf     = expRoot2->leaves[leafPos2];

      int nestPos = 0;

      while((leaf != NULL)            &&
            (leaf->info & expType::C) &&
            (0 < leaf->leafCount)     &&
            (leaf->value == "(")) {

        if (((*leaf)[0].value == "*") ||
           ((*leaf)[0].value == "^")) {

          if ((*leaf)[0].value == "^")
            info |= varType::block;

          if ((leafPos2 + 1) < (expRoot2->leafCount)) {
            leaf = expRoot2->leaves[leafPos2 + 1];

            if ((leaf->info & expType::C) &&
               (leaf->value == "(")) {

              functionNests[nestPos].info = varType::function;
              functionNests[nestPos].loadArgsFrom(s, *expRoot2, leafPos2 + 1);
            }
          }

          expRoot2 = expRoot2->leaves[leafPos2];
          leafPos2 = 1;

          leaf = ((leafPos2 < expRoot.leafCount) ?
                  expRoot2->leaves[leafPos2] :
                  NULL);

          ++nestPos;
        }
        else {
          break;
        }
      }

      if ((expRoot2 != &expRoot) &&
         (nodeHasName(*expRoot2, leafPos2))) {

        leafPos2 = loadNameFromNode(*expRoot2, leafPos2);

        if ((leafPos2 < expRoot2->leafCount) &&
           expRoot2->leaves[leafPos2]->value == "(") {

          info = varType::function;
          loadArgsFrom(s, *expRoot2, leafPos2);
        }

        // Skip the name and function-pointer arguments
        leafPos += 2;
      }

      return leafPos;
    }

    bool varInfo::nodeHasName(expNode &expRoot,
                              int leafPos) {
      if (expRoot.leafCount <= leafPos)
        return false;

      return (expRoot[leafPos].info & (expType::unknown  |
                                       expType::varInfo  |
                                       expType::function));
    }

    int varInfo::loadNameFromNode(expNode &expRoot,
                                  int leafPos) {
      if (expRoot.leafCount <= leafPos)
        return leafPos;

      expNode *leaf = expRoot.leaves[leafPos];

      if (leaf->info & (expType::unknown  |
                       expType::varInfo  |
                       expType::function)) {

        if (leaf->info & expType::varInfo)
          name = leaf->getVarInfo().name;
        else
          name = leaf->value;

        return loadStackPointersFrom(expRoot, leafPos + 1);
      }

      return leafPos;
    }

    int varInfo::loadStackPointersFrom(expNode &expRoot,
                                       int leafPos) {
      if (expRoot.leafCount <= leafPos)
        return leafPos;

      stackPointerCount = 0;

      for (int i = leafPos; i < expRoot.leafCount; ++i) {
        if (expRoot[i].value == "[")
          ++stackPointerCount;
        else
          break;
      }

      if (stackPointerCount) {
        stackExpRoots = new expNode[stackPointerCount];

        for (int i = 0; i < stackPointerCount; ++i) {
          if (expRoot[leafPos + i].leafCount) {
            expRoot[leafPos + i].cloneTo(stackExpRoots[i]);

            stackExpRoots[i].info  = expType::root;
            stackExpRoots[i].value = "";
          }
        }
      }

      stackPointersUsed = stackPointerCount;

      return (leafPos + stackPointerCount);
    }

    int varInfo::loadArgsFrom(statement &s,
                              expNode &expRoot,
                              int leafPos) {
      if ( !(info & varType::function) )
        return leafPos;

      OCCA_ERROR("Missing arguments from function variable",
                 leafPos < expRoot.leafCount);

      if (expRoot[leafPos].leafCount) {
        expNode &leaf = expRoot[leafPos];
        int sLeafPos  = 0;

        argumentCount    = 1 + typeInfo::delimiterCount(leaf, ",");
        argumentVarInfos = new varInfo*[argumentCount];

        for (int i = 0; i < argumentCount; ++i) {
          if (leaf[sLeafPos].value == "...") {
            OCCA_ERROR("Variadic argument [...] has to be the last argument",
                       i == (argumentCount - 1));

            info |= varType::variadic;

            --argumentCount;
            break;
          }

          argumentVarInfos[i] = new varInfo();
          varInfo &argVar = *(argumentVarInfos[i]);

          sLeafPos = argVar.loadFrom(s, leaf, sLeafPos);

          sLeafPos = typeInfo::nextDelimiter(leaf, sLeafPos, ",") + 1;
        }

        setupArrayArguments(s);
      }

      return (leafPos + 1);
    }

    void varInfo::setupArrayArguments(statement &s) {
      int arrayArgs = 0;

      for (int i = 0; i < argumentCount; ++i) {
        if (argumentVarInfos[i]->hasAttribute("arrayArg"))
          ++arrayArgs;
      }

      if (0 < arrayArgs) {
        varInfo **args = new varInfo*[argumentCount + arrayArgs];
        swapValues(argumentVarInfos, args);

        int argPos = 0;

        for (int i = 0; i < argumentCount; ++i) {
          argumentVarInfos[argPos++] = args[i];

          if (args[i]->hasAttribute("arrayArg")) {
            std::string arrayArgName = "__occaAutoKernelArg";
            arrayArgName            += occa::toString(argPos + 1);

            // Don't add if it's already added
            if ((argumentCount <= (i + 1)) ||
               (args[i + 1]->name != arrayArgName)) {

              varInfo &arrayArg = getArrayArgument(s,
                                                   *(args[i]),
                                                   arrayArgName);

              argumentVarInfos[argPos++] = &arrayArg;
            }
          }
        }

        argumentCount = argPos;

        delete [] args;
      }
    }

    varInfo& varInfo::getArrayArgument(statement &s,
                                       varInfo &argVar,
                                       const std::string &arrayArgName) {

      attribute_t &argDimAttr = *(argVar.hasAttribute("dim"));

      varInfo &arrayArg = *(new varInfo());
      const int dims    = argDimAttr.argCount;

      const std::string dims2 = ((1 < dims)                     ?
                                 occa::toString(maxBase2(dims)) :
                                 "");

      // Setup new argument
      arrayArg.addQualifier("occaConst");
      arrayArg.baseType = s.hasTypeInScope("int" + dims2);

      arrayArg.name = arrayArgName;

      // Auto-generated from arrayArg, might need to parse this ...
      for (int i = 0; i < dims; ++i) {
        std::string &dimName = argDimAttr[i].value;

        dimName = arrayArgName;
        dimName += '.';

        if (i < 4) {
          dimName += (char) ('w' + ((i + 1) % 4));
        }
        else {
          dimName += 's';
          dimName += (char) ('0' + i);
        }
      }

      return arrayArg;
    }

    void varInfo::setupAttributes() {
      setupArrayArgAttribute();
      setupDimAttribute();
      setupIdxOrderAttribute();
    }

    void varInfo::setupDimAttribute() {
      attribute_t *attr_ = hasAttribute("dim");

      if (attr_ == NULL)
        return;

      dimAttr = *attr_;
    }

    void varInfo::setupArrayArgAttribute() {
      attribute_t *attr_ = hasAttribute("arrayArg");

      if (attr_ == NULL)
        return;

      attribute_t &attr = *(attr_);

      OCCA_ERROR("@arrayArg has only one argument:\n"
                 "  dims = X",
                 attr.argCount == 1);

      expNode &dimsArg = attr[0];

      if (!dimsArg.isOrganized())
        dimsArg.changeExpTypes();

      typeHolder thDims;

      bool isValid = ((dimsArg.value     == "=")   &&
                      (dimsArg.leafCount == 2)     &&
                      (dimsArg[0].value == "dims") &&
                      (dimsArg[1].info & expType::presetValue));

      if (isValid) {
        thDims = dimsArg[1].calculateValue();

        if (thDims.type & noType)
          isValid = false;
      }

      OCCA_ERROR("@arrayArg must have an argument:\n"
                 "  dims = X, where X is known at compile-time\n",
                 isValid);

      const int dims = thDims.to<int>();

      OCCA_ERROR("@arrayArg only supports dims [1-6]",
                 (0 < dims) && (dims < 7));

      // Add argument dims attribute
      std::string dimAttrStr = "@dim(";

      for (int i = 0; i < dims; ++i) {
        if (0 < i)
          dimAttrStr += ',';

        // Dummy args, the function will fill up the
        //   proper auto-generated dims
        dimAttrStr += '0';
      }

      dimAttrStr += ")";

      expNode attrNode = createExpNodeFrom(dimAttrStr);

      updateAttributeMap(attributeMap,
                         attrNode,
                         0);

      attrNode.free();
    }

    void varInfo::setupIdxOrderAttribute() {
      attribute_t *attr_ = hasAttribute("idxOrder");

      if (attr_ == NULL)
        return;

      attribute_t &idxOrderAttr = *(attr_);

      OCCA_ERROR("Variable [" << *this << "] has attributes dim(...) and idxOrder(...)"
                 " with different dimensions (dim: " << dimAttr.argCount
                 << ", idx: " << idxOrderAttr.argCount << ')',
                 idxOrderAttr.argCount == dimAttr.argCount);

      const int dims = dimAttr.argCount;

      bool *idxFound = new bool[dims];

      idxOrdering.clear();

      for (int i = 0; i < dims; ++i) {
        idxFound[i] = false;
        idxOrdering.push_back(0);
      }

      for (int i = 0; i < dims; ++i) {
        typeHolder th;

        bool foundIdx = false;

        if ((idxOrderAttr[i].leafCount    == 0) &&
           (idxOrderAttr[i].value.size() == 1)) {

          const char c = idxOrderAttr[i].value[0];

          if (('w' <= c) && (c <= 'z')) {
            th = (int) (((c - 'w') + 3) % 4); // [w,x,y,z] -> [x,y,z,w]
            foundIdx = true;
          }
          else if (('W' <= c) && (c <= 'Z')) {
            th = (int) (((c - 'W') + 3) % 4); // [W,X,Y,Z] -> [X,Y,Z,W]
            foundIdx = true;
          }
        }

        if (!foundIdx) {
          OCCA_ERROR("Variable [" << *this << "] has the attribute [" << idxOrderAttr << "] with ordering not known at compile time",
                     idxOrderAttr[i].valueIsKnown());

          th = idxOrderAttr[i].calculateValue();

          OCCA_ERROR("Variable [" << *this << "] has the attribute [" << idxOrderAttr << "] with a non-integer ordering",
                     !th.isAFloat());
        }

        const int idxOrder = th.to<int>();

        idxOrdering[idxOrder] = i;

        OCCA_ERROR("Variable [" << *this << "] has the attribute [" << idxOrderAttr << "] with a repeating index",
                   idxFound[idxOrder] == false);

        OCCA_ERROR("Variable [" << *this << "] has the attribute [" << idxOrderAttr << "] with an index [" << idxOrder << "] outside the range [0," << (dims - 1) << "]",
                   (0 <= idxOrder) && (idxOrder < dims));

        idxFound[idxOrder] = true;
      }
    }

    //   ---[ Fortran ]-------
    int varInfo::loadFromFortran(expNode &expRoot,
                                 int leafPos,
                                 varInfo *varHasType) {

      OCCA_ERROR("Cannot load [varInfo] without a statement",
                 expRoot.sInfo != NULL);

      return loadFromFortran(*(expRoot.sInfo), expRoot, leafPos, varHasType);
    }

    int varInfo::loadFromFortran(statement &s,
                                 expNode &expRoot,
                                 int leafPos,
                                 varInfo *varHasType) {
      // Load Type
      leafPos = loadTypeFromFortran(s, expRoot, leafPos, varHasType);

      // Load Name
      if (expRoot.leafCount <= leafPos)
        return leafPos;

      name = expRoot[leafPos++].value;

      // Load Args
      if (expRoot.leafCount <= leafPos)
        return leafPos;

      if (expRoot[leafPos].leafCount) {
        expNode &leaf = *(expRoot.leaves[leafPos]);

        if (info & varType::function) {
          argumentCount = (leaf.leafCount + 1)/2;

          if (argumentCount)
            argumentVarInfos = new varInfo*[argumentCount];

          for (int i = 0; i < argumentCount; ++i) {
            argumentVarInfos[i] = new varInfo();
            argumentVarInfos[i]->name = leaf[2*i].value;
          }

          leafPos = expRoot.leafCount;
        }
        else {
          leafPos = loadStackPointersFromFortran(expRoot, leafPos);
        }
      }

      return leafPos;
    }

    int varInfo::loadTypeFromFortran(expNode &expRoot,
                                     int leafPos,
                                     varInfo *varHasType) {

      OCCA_ERROR("Cannot load [varInfo] without a statement",
                 expRoot.sInfo != NULL);

      return loadTypeFromFortran(*(expRoot.sInfo), expRoot, leafPos, varHasType);
    }

    int varInfo::loadTypeFromFortran(statement &s,
                                     expNode &expRoot,
                                     int leafPos,
                                     varInfo *varHasType) {
      if (expRoot.leafCount <= leafPos)
        return leafPos;

      if (varHasType == NULL) {
        leafPos = leftQualifiers.loadFromFortran(s, *this, expRoot, leafPos);

        if (leafPos < expRoot.leafCount) {
          if (expRoot[leafPos].value == "SUBROUTINE") {
            baseType = s.hasTypeInScope("void");
            info    |= varType::functionDec;
            ++leafPos;
          }
          else if (expRoot[leafPos].value == "FUNCTION") {
            info |= varType::functionDec;
            ++leafPos;
          }
        }
      }
      else {
        leftQualifiers  = varHasType->leftQualifiers.clone();
        rightQualifiers = varHasType->rightQualifiers.clone();
        baseType        = varHasType->baseType;
      }

      if ( !(info & varType::functionDec) )
        info |= varType::var;

      return leafPos;
    }

    std::string varInfo::getFullFortranType(expNode &expRoot,
                                            int &leafPos) {
      if ( !(expRoot[leafPos].info & expType::type) )
        return "";

      std::string typeNode = expRoot[leafPos++].value;

      if (leafPos < expRoot.leafCount) {
        int bytes = -1;

        // [-] Ignoring complex case
        const bool isFloat = ((typeNode.find("REAL") != std::string::npos) ||
                              (typeNode == "PRECISION")                    ||
                              (typeNode == "COMPLEX"));

        const int typeNodeChars  = typeNode.size();
        const bool typeHasSuffix = isADigit(typeNode[typeNodeChars - 1]);

        std::string suffix = "";

        if (typeHasSuffix) {
          for (int i = 0; i < typeNodeChars; ++i) {
            if (isADigit(typeNode[i]))
              suffix += typeNode[i];
          }
        }

        if (isFloat) {
          if (typeNode.find("REAL") != std::string::npos)
            bytes = 4;
          else if (typeNode == "PRECISION")
            bytes = 8;
        }
        else {
          if (typeNode.find("INTEGER") != std::string::npos)
            bytes = 4;
          else if ((typeNode == "LOGICAL") ||
                  (typeNode == "CHARACTER"))
            bytes = 1;
        }

        if (leafPos < expRoot.leafCount) {
          if (expRoot[leafPos].value == "*") {
            ++leafPos;
            bytes    = atoi(expRoot[leafPos].value.c_str());
            ++leafPos;
          }
          else if ((expRoot[leafPos].value == "(") &&
                  (expRoot[leafPos].leafCount)) {

            bytes = atoi(expRoot[leafPos][0].value.c_str());
            ++leafPos;
          }
        }

        switch(bytes) {
        case 1:
          typeNode = "char" + suffix; break;
        case 2:
          typeNode = "short" + suffix; break;
        case 4:
          if (isFloat)
            typeNode = "float" + suffix;
          else
            typeNode = "int" + suffix;
          break;
        case 8:
          if (isFloat)
            typeNode = "double" + suffix;
          else
            typeNode = "long long" + suffix;
          break;
        default:
          OCCA_ERROR("Error loading " << typeNode << "(" << bytes << ")",
                     false);
        };
      }

      return typeNode;
    }

    int varInfo::loadStackPointersFromFortran(expNode &expRoot,
                                              int leafPos) {
      if (expRoot.leafCount <= leafPos)
        return leafPos;

      if ((expRoot[leafPos].value != "(") ||
         (expRoot[leafPos].leafCount == 0)) {

        if (expRoot[leafPos].value == "(")
          return (leafPos + 1);

        return leafPos;
      }

      // rightQualifiers are copied from [firstVar]
      if (rightQualifiers.has("*"))
        rightQualifiers.remove("*");

      expRoot[leafPos].changeExpTypes();
      expRoot[leafPos].organize(parserInfo::parsingFortran);

      expNode &csvFlatRoot = *(expRoot[leafPos][0].makeCsvFlatHandle());

      for (int i = 0; i < csvFlatRoot.leafCount; ++i) {
        expNode &stackNode = csvFlatRoot[i];

        if (stackNode.value == ":") {
          pointerCount      = csvFlatRoot.leafCount;
          stackPointerCount = 0;

          for (int j = 0; j < pointerCount; ++j)
            rightQualifiers.add("*", 0);

          break;
        }
        else {
          ++stackPointerCount;
        }
      }

      if (stackPointerCount) {
        stackExpRoots = new expNode[stackPointerCount];

        for (int i = 0; i < stackPointerCount; ++i) {
          stackExpRoots[i].info  = expType::root;
          stackExpRoots[i].value = "";

          stackExpRoots[i].addNode();
          csvFlatRoot[i].cloneTo(stackExpRoots[i][0]);
        }
      }

      expNode::freeFlatHandle(csvFlatRoot);

      ++leafPos;

      if (pointerCount &&
         rightQualifiers.has("&")) {

        rightQualifiers.remove("&");
      }

      return leafPos;
    }

    void varInfo::setupFortranStackExp(expNode &stackExp,
                                       expNode &valueExp) {
      stackExp.info  = expType::C;
      stackExp.value = "[";

      stackExp.leaves    = new expNode*[1];
      stackExp.leafCount = 1;

      stackExp.leaves[0] = &valueExp;
    }
    //   =====================

    void varInfo::organizeExpNodes() {
      for (int i = 0; i < stackPointerCount; ++i) {
        if (!stackExpRoots[i].isOrganized()) {
          stackExpRoots[i].changeExpTypes();
          stackExpRoots[i].initOrganization();
          stackExpRoots[i].organize();
        }
      }

      for (int i = 0; i < argumentCount; ++i)
        argumentVarInfos[i]->organizeExpNodes();

      for (int i = 0; i < functionNestCount; ++i)
        functionNests[i].organizeExpNodes();
    }
    //==================================


    //---[ Variable Info ]------------
    attribute_t* varInfo::hasAttribute(const std::string &attr) {
      attributeMapIterator it = attributeMap.find(attr);

      if (it != attributeMap.end())
        return (it->second);

      if ((baseType             != NULL) &&
         (baseType->typedefVar != NULL)) {

        return baseType->typedefVar->hasAttribute(attr);
      }

      return NULL;
    }

    void varInfo::removeAttribute(const std::string &attr) {
      attributeMapIterator it = attributeMap.find(attr);

      if (it != attributeMap.end())
        attributeMap.erase(it);
    }

    int varInfo::leftQualifierCount() {
      return leftQualifiers.qualifierCount;
    }

    int varInfo::rightQualifierCount() {
      return rightQualifiers.qualifierCount;
    }

    int varInfo::hasQualifier(const std::string &qName) {
      return leftQualifiers.has(qName);
    }

    int varInfo::hasRightQualifier(const std::string &qName) {
      return rightQualifiers.has(qName);
    }

    void varInfo::addQualifier(const std::string &qName, int pos) {
      leftQualifiers.add(qName, pos);
    }

    void varInfo::addRightQualifier(const std::string &qName, int pos) {
      rightQualifiers.add(qName, pos);
    }

    void varInfo::removeQualifier(const std::string &qName) {
      leftQualifiers.remove(qName);
    }

    void varInfo::removeRightQualifier(const std::string &qName) {
      rightQualifiers.remove(qName);
    }

    std::string& varInfo::getLeftQualifier(const int pos) {
      return leftQualifiers.get(pos);
    }

    std::string& varInfo::getRightQualifier(const int pos) {
      return rightQualifiers.get(pos);
    }

    std::string& varInfo::getLastLeftQualifier() {
      return leftQualifiers.get(leftQualifiers.qualifierCount - 1);
    }

    std::string& varInfo::getLastRightQualifier() {
      return rightQualifiers.get(rightQualifiers.qualifierCount - 1);
    }

    int varInfo::pointerDepth() {
      if (baseType)
        return (pointerCount + stackPointerCount + baseType->pointerDepth());
      else
        return (pointerCount + stackPointerCount);
    }

    expNode& varInfo::stackSizeExpNode(const int pos) {
      return stackExpRoots[pos];
    }

    void varInfo::removeStackPointers() {
      if (stackPointerCount) {
        stackPointerCount = 0;
        stackPointersUsed = 0;

        delete [] stackExpRoots;
        stackExpRoots = NULL;
      }
    }

    varInfo& varInfo::getArgument(const int pos) {
      return *(argumentVarInfos[pos]);
    }

    void varInfo::setArgument(const int pos, varInfo &var) {
      argumentVarInfos[pos] = &var;
    }

    void varInfo::addArgument(const int pos, varInfo &arg) {
      varInfo **newArgumentVarInfos = new varInfo*[argumentCount + 1];

      for (int i = 0; i < pos; ++i)
        newArgumentVarInfos[i] = argumentVarInfos[i];

      newArgumentVarInfos[pos] = &arg;

      for (int i = pos; i < argumentCount; ++i)
        newArgumentVarInfos[i + 1] = argumentVarInfos[i];

      if (argumentCount)
        delete [] argumentVarInfos;

      argumentVarInfos = newArgumentVarInfos;
      ++argumentCount;
    }
    //================================


    //---[ Class Info ]---------------
    varInfo* varInfo::hasOperator(const std::string &op) {
      if (op.size() == 0)
        return NULL;

      if (pointerDepth())
        return (varInfo*) -1; // Dummy non-zero value

      if (baseType)
        return baseType->hasOperator(op);

      return NULL;
    }

    bool varInfo::canBeCastedTo(varInfo &var) {
      if (((    baseType->thType & noType) == 0) &&
         ((var.baseType->thType & noType) == 0)) {

        return true;
      }

      return false;
    }

    bool varInfo::hasSameTypeAs(varInfo &var) {
      if (baseType != var.baseType)
        return false;

      if (stackPointerCount != var.stackPointerCount)
        return false;

      // [-] Need to check if void* is an exception
      if (pointerCount != var.pointerCount)
        return false;

      return true;
    }
    //================================

    bool varInfo::isConst() {
      const int qCount = leftQualifiers.qualifierCount;

      for (int i = 0; i < qCount; ++i) {
        const std::string &q = leftQualifiers[i];

        if ((q == "const") ||
           (q == "occaConst")) {

          return true;
        }
      }

      return false;
    }

    void varInfo::printDebugInfo() {
      std::cout << toString() << ' ' << attributeMapToString(attributeMap) << '\n';
    }

    void varInfo::printOnString(std::string &str,
                                const bool printType) {

      bool addSpaceBeforeName = false;

      if (printType) {
        leftQualifiers.printOnString(str);

        if (baseType)
          str += baseType->name;

        addSpaceBeforeName = !((rightQualifiers.qualifierCount) ||
                               (name.size()));

        if (!addSpaceBeforeName) {
          if ((info & varType::function)       &&
             (rightQualifiers.qualifierCount) &&
             ((getLastRightQualifier() == "*") ||
              (getLastRightQualifier() == "&"))) {

            addSpaceBeforeName = true;
          }
        }

        if (!addSpaceBeforeName && baseType)
          str += ' ';
      }

      rightQualifiers.printOnString(str, this);

      for (int i = 0; i < (functionNestCount - 1); ++i)
        str += "(*";

      if (functionNestCount) {
        if ((info & varType::block) == 0)
          str += "(*";
        else
          str += "(^";
      }

      if (addSpaceBeforeName &&
         (name.size() != 0))
        str += ' ';

      str += name;

      if (stackPointerCount && stackPointersUsed) {
        if (stackPointersUsed == stackPointerCount) {
          for (int i = 0; i < stackPointerCount; ++i) {
            str += '[';
            str += (std::string) stackExpRoots[i];
            str += ']';
          }
        }
        else {
          str += "[(";
          str += (std::string) stackSizeExpNode(0);
          str += ')';

          for (int i = 1; i < stackPointerCount; ++i) {
            str += "*(";
            str += (std::string) stackSizeExpNode(i);
            str += ")";
          }

          str += ']';
        }
      }

      if (info & varType::function) {
        str += '(';

        const int argTabSpaces = charsBeforeNewline(str);
        const std::string argTab(argTabSpaces, ' ');

        if (argumentCount) {
          argumentVarInfos[0]->printOnString(str);

          for (int i = 1; i < argumentCount; ++i) {
            str += ",\n";
            str += argTab;
            argumentVarInfos[i]->printOnString(str);
          }
        }

        if (info & varType::variadic) {
          if (argumentCount) {
            str += ",\n";
            str += argTab;
          }

          str += "...";
        }

        str += ')';
      }

      for (int i = (functionNestCount - 1); 0 <= i; --i) {
        str += ')';
        functionNests[i].printOnString(str);
      }

      if (0 <= bitfieldSize) {
        str += " : ";
        str += occa::toString(bitfieldSize);
      }

      attribute_t *attr = hasAttribute("__attribute__");

      if (attr != NULL) {
        str += " __attribute__";
        str += attr->name;
      }
    }
    //============================================


    //---[ Overloaded Operator Class ]------------
    void overloadedOp_t::add(varInfo &function) {
      functions.push_back(&function);
    }

    varInfo* overloadedOp_t::getFromArgs(const int argumentCount,
                                         expNode *arguments) {

      varInfo *argumentTypes = new varInfo[argumentCount];

      for (int i = 0; i < argumentCount; ++i)
        argumentTypes[i] = arguments[i].evaluateType();

      varInfo *ret = getFromTypes(argumentCount,
                                  argumentTypes);

      delete [] argumentTypes;

      return ret;
    }

    varInfo* overloadedOp_t::getFromTypes(const int argumentCount,
                                          varInfo *argumentTypes) {

      const int functionCount = (int) functions.size();

      varInfoVector_t candidates;

      for (int i = 0; i < functionCount; ++i) {
        varInfo &f = argumentTypes[i];
        int arg;

        if (f.argumentCount != argumentCount)
          continue;

        for (arg = 0; arg < argumentCount; ++arg) {
          if (!argumentTypes[arg].canBeCastedTo(f.getArgument(arg)))
            break;
        }

        if (arg == argumentCount)
          candidates.push_back(&f);
      }

      return bestFitFor(argumentCount,
                        argumentTypes,
                        candidates);
    }

    varInfo* overloadedOp_t::bestFitFor(const int argumentCount,
                                        varInfo *argumentTypes,
                                        varInfoVector_t &candidates) {

      const int candidateCount = (int) candidates.size();

      if (candidateCount == 0)
        return NULL;
      else if (candidateCount == 1)
        return candidates[0];

      int nonAmbiguousCount = candidateCount;
      bool *ambiguous       = new bool[candidateCount];

      for (int i = 0; i < candidateCount; ++i)
        ambiguous[i] = false;

      for (int arg = 0; arg < argumentCount; ++arg) {
        varInfo &argType = argumentTypes[arg];

        for (int i = 0; i < candidateCount; ++i) {
          if (!ambiguous[i])
            continue;

          if (candidates[i]->getArgument(arg).hasSameTypeAs(argType)) {
            for (int i2 = 0; i2 < i; ++i) {
              if (!ambiguous[i2]) {
                --nonAmbiguousCount;
                ambiguous[i2] = true;
              }
            }

            for (int i2 = (i + 1); i2 < candidateCount; ++i2) {
              if (!candidates[i2]->getArgument(arg).hasSameTypeAs(argType)) {
                if (!ambiguous[i2]) {
                  --nonAmbiguousCount;
                  ambiguous[i2] = true;
                }
              }
            }
          }
        }

        // [-] Clean the error message
        OCCA_ERROR("Ambiguous Function",
                   0 < nonAmbiguousCount);
      }

      // [-] Clean the error message
      OCCA_ERROR("Ambiguous Function",
                 1 < nonAmbiguousCount);

      for (int i = 0; i < candidateCount; ++i) {
        if (!ambiguous[i])
          return candidates[i];
      }

      return NULL;
    }
    //============================================


    //---[ Kernel Info ]--------------------------
    argumentInfo::argumentInfo() :
      pos(0),
      isConst(false) {}

    argumentInfo::argumentInfo(const argumentInfo &info) :
      pos(info.pos),
      isConst(info.isConst) {}

    argumentInfo& argumentInfo::operator = (const argumentInfo &info) {
      pos     = info.pos;
      isConst = info.isConst;

      return *this;
    }

    argumentInfo argumentInfo::fromJson(const json &j) {
      argumentInfo info;
      info.pos     = j["pos"];
      info.isConst = j["isConst"];
      return info;
    }

    json argumentInfo::toJson() const {
      json j;
      j["pos"] = pos;
      j["isConst"] = isConst;
      return j;
    }

    kernelInfo::kernelInfo() :
      name(),
      baseName() {}

    kernelInfo::kernelInfo(const kernelInfo &info) :
      name(info.name),
      baseName(info.baseName),
      nestedKernels(info.nestedKernels),
      argumentInfos(info.argumentInfos) {}

    kernelInfo& kernelInfo::operator = (const kernelInfo &info) {
      name     = info.name;
      baseName = info.baseName;

      nestedKernels = info.nestedKernels;
      argumentInfos = info.argumentInfos;

      return *this;
    }

    occa::kernelMetadata kernelInfo::metadata() {
      occa::kernelMetadata kInfo;

      kInfo.name     = name;
      kInfo.baseName = baseName;

      kInfo.nestedKernels = nestedKernels.size();
      kInfo.argumentInfos = argumentInfos;

      return kInfo;
    }
    //==============================================
  }

  //---[ Parsed Kernel Info ]---------------------
  kernelMetadata::kernelMetadata() :
    name(""),
    baseName(""),
    nestedKernels(0) {}

  kernelMetadata::kernelMetadata(const kernelMetadata &kInfo) :
    name(kInfo.name),
    baseName(kInfo.baseName),
    nestedKernels(kInfo.nestedKernels),
    argumentInfos(kInfo.argumentInfos) {}

  kernelMetadata& kernelMetadata::operator = (const kernelMetadata &kInfo) {
    name     = kInfo.name;
    baseName = kInfo.baseName;

    nestedKernels = kInfo.nestedKernels;
    argumentInfos = kInfo.argumentInfos;

    return *this;
  }

  void kernelMetadata::removeArg(const int pos) {
    argumentInfos.erase(argumentInfos.begin() + pos);
  }


  kernelMetadata kernelMetadata::getNestedKernelMetadata(const int kIdx) const {
    kernelMetadata meta;
    meta.name     = baseName + toString(kIdx);
    meta.baseName = baseName;

    const int argumentCount = (int) argumentInfos.size();
    for (int i = 0; i < argumentCount; ++i) {
      // Don't pass nestedKernels** argument
      if (i != 1) {
        meta.argumentInfos.push_back(argumentInfos[i]);
      }
    }
    return meta;
  }

  kernelMetadata kernelMetadata::fromJson(const json &j) {
    kernelMetadata meta;

    meta.name          = j["name"].string();
    meta.baseName      = j["baseName"].string();
    meta.nestedKernels = j["nestedKernels"].number();

    const jsonArray_t &argInfos = j["argumentInfos"].array();
    const int argumentCount = (int) argInfos.size();
    for (int i = 0; i < argumentCount; ++i) {
      meta.argumentInfos.push_back(argumentInfo::fromJson(argInfos[i]));
    }

    return meta;
  }

  json kernelMetadata::toJson() const {
    json j;

    j["name"]          = name;
    j["baseName"]      = baseName;
    j["nestedKernels"] = nestedKernels;

    const int argumentCount = (int) argumentInfos.size();
    json &argInfos = j["argumentInfos"].asArray();
    for (int k = 0; k < argumentCount; ++k) {
      argInfos += argumentInfos[k].toJson();
    }

    return j;
  }
  //==============================================
}
