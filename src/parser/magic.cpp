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

#include "occa/parser/magic.hpp"
#include "occa/tools/string.hpp"

#define DBP0 1 // Read/Write/Expand
#define DBP1 0 // Index Sorting/Updating
#define DBP2 0 // Expression Simplification
#define DBP3 0 // Has Stride
#define DBP4 1 // Check Complex Inputs, Access Stride Conflicts, Access Conflicts
#define DBP5 0 // LCD-labeled Statements and GCS Prints, For-loops with LCD
#define DBP6 1 // Outer-Loop/Inner-Loop Posibilities

namespace occa {
  namespace parserNS {
    namespace viType {
      std::string infoToStr(const int info) {
        std::string tag;

        if (info & viType::isUseless)    tag += 'U';
        if (info & viType::isAVariable)  tag += 'V';
        if (info & viType::isAnIterator) tag += 'I';
        if (info & viType::isConstant)   tag += 'C';
        if (info & viType::isComplex)    tag += '@';

        return tag;
      }
    }

    atomInfo_t::atomInfo_t(infoDB_t *db_) :
      db(db_),
      info(viType::isUseless),
      exp(NULL),
      var(NULL) {}

    atomInfo_t::atomInfo_t(const atomInfo_t &ai) :
      db(ai.db),
      info(ai.info),
      constValue(ai.constValue),
      exp(ai.exp),
      var(ai.var) {}

    atomInfo_t& atomInfo_t::operator = (const atomInfo_t &ai) {
      db = ai.db;

      info       = ai.info;
      constValue = ai.constValue;
      exp        = ai.exp;
      var        = ai.var;

      return *this;
    }

    bool atomInfo_t::operator == (const std::string &str) {
      if ( !(info & viType::isConstant) )
        return false;

      return (constValue == typeHolder(str));
    }

    bool atomInfo_t::operator == (expNode &e) {
      if (info & viType::isConstant) {
        if (e.valueIsKnown())
          return (constValue == e.calculateValue());

        return false;
      }

      if (info & viType::isAVariable) {
        if (e.info & expType::varInfo)
          return (var == &(e.getVarInfo()));

        return false;
      }

      return (exp && (*exp == e));
    }

    bool atomInfo_t::operator == (atomInfo_t &ai) {
      if (info & viType::isConstant) {
        if ( !(ai.info & viType::isConstant) )
          return false;

        return (constValue == ai.constValue);
      }
      else if (info & viType::isAVariable) {
        if ( !(ai.info & viType::isAVariable) )
          return false;

        return (var == ai.var);
      }
      else {
        if (info & (viType::isConstant |
                   viType::isAVariable)) {
          return false;
        }

        if ((exp == NULL) || (ai.exp == NULL))
          return false;

        return exp->sameAs(*ai.exp);
      }

      return true;
    }

    bool atomInfo_t::operator != (atomInfo_t &ai) {
      return !(*this == ai);
    }

    void atomInfo_t::setDB(infoDB_t *db_) {
      db = db_;
    }

    void atomInfo_t::load(expNode &e) {
      info = viType::isUseless;

      if (e.info & expType::varInfo) {
        info = viType::isAVariable;
        var  = &(e.getVarInfo());
      }
      else {
        constValue = e.calculateValue();

        if (constValue.type & noType) {
          if (exp == NULL)
            exp = e.clonePtr();
          else
            e.cloneTo(*exp);
        }
        else
          info = viType::isConstant;
      }
    }

    void atomInfo_t::load(varInfo &var_) {
      info = viType::isAVariable;
      var  = &var_;
    }

    void atomInfo_t::load(const std::string &s) {
      info       = viType::isConstant;
      constValue = s;
    }

    bool atomInfo_t::expandValue() {
      if (info & viType::isConstant)
        return !analyzeInfo::changed;

      if (info & viType::isAVariable)
        return expandValue(exp, *var);
      else
        return expandValue(*exp);
    }

    bool atomInfo_t::expandValue(expNode &e) {
      int iterations = 0;

      while(true) {
        bool updated = false;

        expNode &flatRoot = *(e.makeFlatHandle());

        for (int i = 0; i < flatRoot.leafCount; ++i) {
          expNode &leaf = flatRoot[i];

          if (leaf.info & expType::varInfo) {
            expNode *l = &leaf;
            varInfo &v = leaf.getVarInfo();

            expandValue(l, v);

            if (!(leaf.info & expType::varInfo) ||
               &(leaf.getVarInfo()) != &v) {

              updated = true;
            }
          }
        }

        expNode::freeFlatHandle(flatRoot);

        if (!updated)
          break;

        ++iterations;
      }

      if (iterations == 0)
        return !analyzeInfo::changed;

      return analyzeInfo::changed;
    }

    bool atomInfo_t::expandValue(expNode *&expRoot, varInfo &v) {
      if ((0 < v.pointerDepth()) ||
         (v.info & varType::function)) {

        return !analyzeInfo::changed;
      }

      viInfo_t *vi = db->has(v);

      // Can't do anything about variables outside db's scope
      //   and don't want to replace iterator values
      if ((vi == NULL)                ||
         !(vi->hasBeenInitialized()) ||
         (vi->info & (viType::isComplex |
                      viType::isAnIterator))) {

        return !analyzeInfo::changed;
      }

      const bool expRootWasNull = (expRoot == NULL);

      if (expRoot == NULL)
        expRoot = new expNode();

      expNode &e = *expRoot;

      vi->valueInfo.saveTo(e);

      // We're operating on our exp (which was NULL)
      if (expRootWasNull) {
        if (e[0].info & expType::varInfo) {
          if (&(e[0].getVarInfo()) == &v)
            return !analyzeInfo::changed;

          var = &(e[0].getVarInfo());

          e.free();
          expRoot = NULL;
        }
        else {
          if (e.leafCount) {
            expNode *e0 = &(e[0]);
            e.freeThis();
            expRoot = e0;
          }

          info &= ~(viType::isAVariable |
                    viType::isAnIterator);

          constValue = expRoot->calculateValue();

          if ( !(constValue.type & noType) )
            info |= viType::isConstant;
        }
      }
      else { // Root node -> ()
        e.info  = expType::C;
        e.value = "(";
      }

      return analyzeInfo::changed;
    }

    void atomInfo_t::saveTo(expNode &e, const int leafPos) {
      if (e.info & expType::hasInfo)
        e.free();

      if ((e.leafCount <= leafPos) &&
         !(info & viType::isAVariable)) {

        e.addNode(leafPos);
      }

      if (info & viType::isConstant) {
        expNode &leaf = e[leafPos];

        leaf.free();
        leaf.info  = expType::presetValue;
        leaf.value = (std::string) constValue;
      }
      else if (info & viType::isAVariable) {
        e[leafPos].putVarInfo(*var);
      }
      else if (exp &&
              (0 < exp->leafCount)) {

        e[leafPos].free();
        e.leaves[leafPos] = ((exp->info == 0) ? exp->leaves[0] : exp); // [<>] Find why root info is not 0
      }
    }

    bool atomInfo_t::isComplex() {
      if ((info & (viType::isConstant |
                  viType::isAVariable)) ||
         (exp == NULL)) {

        return false;
      }
      else {
        expNode &flatRoot = *(exp->makeFlatHandle());

        for (int i = 0; i < flatRoot.leafCount; ++i) {
          expNode &leaf = flatRoot[i];

          if ((leaf.info & expType::varInfo) &&
             (0 < leaf.getVarInfo().pointerDepth())) {

            return true;
          }
        }

        expNode::freeFlatHandle(flatRoot);
      }

      return false;
    }

    std::string atomInfo_t::getInfoStr() {
      viInfo_t *vi;

      if ((info & viType::isAVariable) &&
         (vi = db->has(*var))) {

        return viType::infoToStr(info | (vi->info & ~viType::isUseless));
      }

      return viType::infoToStr(info);
    }

    std::ostream& operator << (std::ostream &out, atomInfo_t &info) {
      out << info.getInfoStr() << ": ";

      if (info.info & viType::isConstant)
        out << info.constValue;
      else if (info.info & viType::isAVariable)
        out << info.var->name;
      else if (info.exp != NULL)
        out << *(info.exp);

      return out;
    }

    valueInfo_t::valueInfo_t(infoDB_t *db_) :
      db(db_),
      info(0),
      indices(0),
      value(db_),
      vars(NULL),
      strides(NULL) {}

    valueInfo_t::valueInfo_t(expNode &e, infoDB_t *db_) :
      db(db_),
      info(0),
      indices(0),
      value(db_),
      vars(NULL),
      strides(NULL) {

      load(e);
    }

    valueInfo_t::valueInfo_t(const valueInfo_t &vi) :
      db(vi.db),
      info(vi.info),
      indices(vi.indices),
      value(vi.value),
      vars(vi.vars),
      strides(vi.strides) {}

    valueInfo_t& valueInfo_t::operator = (const valueInfo_t &vi) {
      db = vi.db;

      info    = vi.info;
      indices = vi.indices;
      value   = vi.value;
      vars    = vi.vars;
      strides = vi.strides;

      return *this;
    }

    bool valueInfo_t::operator == (valueInfo_t &vi) {
      if (indices != vi.indices)
        return false;

      if (indices == 0) {
        return (value == vi.value);
      }
      else {
        for (int i = 0; i < indices; ++i)
          if ((vars[i]    != vi.vars[i]) ||
             (strides[i] != vi.strides[i])) {

            return false;
          }
      }

      return true;
    }

    bool valueInfo_t::operator != (valueInfo_t &vi) {
      return !(*this == vi);
    }

    void valueInfo_t::setDB(infoDB_t *db_) {
      db = db_;

      value.setDB(db);

      for (int i = 0; i < indices; ++i) {
        vars[i].setDB(db);
        strides[i].setDB(db);
      }
    }

    void valueInfo_t::allocVS(const int count) {
      if (count <= 0)
        return;

      vars    = new atomInfo_t[count];
      strides = new atomInfo_t[count];

      for (int i = 0; i < count; ++i) {
        vars[i].setDB(db);
        strides[i].setDB(db);
      }
    }

    bool valueInfo_t::isConstant() {
      if (info & viType::isConstant)
        return true;

      for (int i = 0; i < indices; ++i) {
        if (!(vars[i].info    & viType::isConstant) ||
           !(strides[i].info & viType::isConstant)) {

          return false;
        }
      }

      return true;
    }

    typeHolder valueInfo_t::constValue() {
      typeHolder ret;

      if (!isConstant())
        return ret;

      if (indices == 0)
        return value.constValue;

      ret = (vars[0].constValue * strides[0].constValue);

      for (int i = 1; i < indices; ++i)
        ret += (vars[0].constValue * strides[0].constValue);

      return ret;
    }

    bool valueInfo_t::isUseless() {
      if (info & viType::isConstant)
        return false;

      for (int i = 0; i < indices; ++i) {
        if (vars[i].info & viType::isUseless)
          return true;
      }

      return false;
    }

    bool valueInfo_t::isComplex() {
      if (indices == 0) {
        return value.isComplex();
      }
      else {
        for (int i = 0; i < indices; ++i) {
          if (vars[i].isComplex() || strides[i].isComplex())
            return true;
        }

        return false;
      }
    }

    void valueInfo_t::load(expNode &e) {
      info = 0;
      expNode e2 = e.clone();

#if DBP2
      std::cout << "SIMP1: e2 = " << e2 << '\n';
      // e2.print();
#endif
      magician::simplify(*db, e2);
#if DBP2
      std::cout << "SIMP2: e2 = " << e2 << '\n';
      // e2.print();
#endif

      expVector_t strideNodes;
      magician::placeAddedExps(*db, e2, strideNodes);

      indices = strideNodes.size();

      allocVS(indices);

      const int snc = strideNodes.size();

      for (int i = 0; i < snc; ++i)
        loadVS(*(strideNodes[i]), i);

      const bool changed = expandValues();

      if (changed)
        reEvaluateStrides();

      // sortIndices();
    }

    void valueInfo_t::load(varInfo &var_) {
      value.load(var_);
    }

    void valueInfo_t::load(const std::string &s) {
      value.load(s);
    }

    void valueInfo_t::loadVS(expNode &e, const int pos) {
      if (e.info & expType::varInfo) {
        vars[pos].load(e);
        strides[pos].load("1");
        return;
      }
      else if (e.info & expType::presetValue) {
        vars[pos].load(e.value);
        strides[pos].load("1");
        return;
      }
      else if ((e.info & expType::LR) &&
         (e.value == "*")) {

        const bool varIn0 = (e[0].info & expType::varInfo);
        const bool varIn1 = (e[1].info & expType::varInfo);

        if (varIn0 || varIn1) {
          vars[pos].load(e[varIn1]);
          strides[pos].load(e[!varIn1]);
          return;
        }
      }
      /*
      else if (iteratorsExprIn(e)) { // Wrong
        const bool hasIter0 = isAnIteratorExp(e[0]);
        const bool hasIter1 = isAnIteratorExp(e[1]);

        if (hasIter0 ^ hasIter1) {
          vars[pos].load(e[hasIter1]);
          strides[pos].load(e[!hasIter1]);
          return;
        }
      }
      */

      vars[pos].info    = viType::isUseless;
      strides[pos].info = viType::isUseless;

      vars[pos].exp = e.clonePtr();
      strides[pos].load("1");
    }

    int valueInfo_t::iteratorsIn(expNode &e) {
      if (e.info & expType::varInfo)
        return db->varIsAnIterator(e.getVarInfo());

      int count = 0;

      expNode &flatRoot = *(e.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        if ( (leaf.info & expType::varInfo) &&
            db->varIsAnIterator(leaf.getVarInfo()) ) {

          ++count;
        }
      }

      expNode::freeFlatHandle(flatRoot);

      return count;
    }

    bool valueInfo_t::hasAnIterator(expNode &e) {
      return (0 < iteratorsIn(e));
    }

    bool valueInfo_t::isAnIteratorExp(expNode &e) {
      if (e.info & expType::varInfo) {
          return db->varIsAnIterator(e.getVarInfo());
      }

      int iterCount = 0;
      bool ret      = true;

      expNode &flatRoot = *(e.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        if (leaf.info & expType::varInfo) {
          ret = db->varIsAnIterator(leaf.getVarInfo());

          if (ret)
            ++iterCount;
        }

        if (ret == false)
          break;
      }

      expNode::freeFlatHandle(flatRoot);

      return (ret && (0 < iterCount));
    }

    int valueInfo_t::iteratorExpsIn(expNode &e) {
      return 0;
    }

    bool valueInfo_t::expandValues() {
      if (indices == 0) {
        return value.expandValue();
      }
      else {
        bool changed = false;

        for (int i = 0; i < indices; ++i) {
          changed |= vars[i].expandValue();
          changed |= strides[i].expandValue();
        }

        if (!changed)
          return !analyzeInfo::changed;

        // [<>] Memory Leak
        expNode e;
        saveTo(e);
        load(e[0]); // Skip [0] root
        e.freeThis();

        return analyzeInfo::changed;
      }
    }

    void valueInfo_t::reEvaluateStrides() {
      if (indices == 0)
        return;

      for (int i = 0; i < indices; ++i) {
        if ( !(vars[i].info & viType::isUseless) ||
            (vars[i].exp == NULL) ) {

          continue;
        }

        if ( hasAnIterator(*(vars[i].exp)) ) {
          valueInfo_t value2(db);
          value2.load( *(vars[i].exp) );
        }
      }
    }

    void valueInfo_t::sortIndices() {
      if (indices <= 1)
        return;

#if DBP1
      std::cout << "SI 1: " << *this << '\n';
#endif

      int *vi = new int[2*indices];

      for (int i = 0; i < indices; ++i) {
        vi[2*i + 0] = 0; // Value
        vi[2*i + 1] = i; // Index
      }

      // Add values

      // Sort based by value
      qsort(vi, indices, 2*sizeof(int), valueInfo_t::qSortIndices);

      // Re-order
      for (int i = 0; i < indices; ++i) {
        int i2 = vi[2*i + 1];

        if (i == i2)
          continue;

        if (i2 < i)
          i2 = vi[2*i2 + 1];

        atomInfo_t tmpV = vars[i];
        atomInfo_t tmpS = strides[i];

        vars[i]    = vars[i2];
        strides[i] = strides[i2];

        vars[i2]    = tmpV;
        strides[i2] = tmpS;
      }

#if DBP1
      std::cout << "SI 2: " << *this << '\n';
#endif
    }

    int valueInfo_t::qSortIndices(const void *a, const void *b) {
      return ((*((int*) a)) - (*((int*) b)));
    }

    void valueInfo_t::mergeIndices() {
    }

    void valueInfo_t::saveTo(expNode &e, const int leafPos) {
      if (indices == 0) { // Adding value
        value.saveTo(e, leafPos);
      }
      else {            // Merging vars and strides
        if (e.info & expType::hasInfo)
          e.free();

        if (e.leafCount <= leafPos)
          e.addNode(leafPos);

        if (indices == 1) {
          saveIndexTo(0, e, leafPos);
        }
        else if (1 < indices) {
          expNode *cNode = &(e[leafPos]);

          for (int i = 0; i < (indices - 1); ++i) {
            cNode->info  = expType::LR;
            cNode->value = "+";

            cNode->addNodes(2);

            saveIndexTo(i, *cNode, 0);

            if (i == (indices - 2))
              saveIndexTo(i + 1, *cNode, 1);

            cNode = cNode->leaves[1];
          }
        }
      }
    }

    void valueInfo_t::saveIndexTo(const int index,
                                  expNode &e, const int leafPos) {

      if (strides[index] == "1") {
        vars[index].saveTo(e, leafPos);
      }
      else {
        e[leafPos].free();

        expNode &leaf = e[leafPos];

        leaf.info  = expType::LR;
        leaf.value = "*";

        leaf.addNodes(2);

        vars[index].saveTo(leaf, 0);
        strides[index].saveTo(leaf, 1);
      }
    }

    void valueInfo_t::update(expNode &op, expNode &e) {
      if (op.value == "=") {
        load(e);
        return;
      }

      if (info & viType::isUseless)
        return;

      if ((op.value.size() == 2) &&
         (op.value == "++" ||
          op.value == "--" ||
          op.value == "+=" ||
          op.value == "-=" ||
          op.value == "*=" ||
          op.value == "/=")) {

        std::string opValue;
        opValue += op.value[0];

        if (op.value[1] == '=')
          insertOp(opValue, e);
        else
          insertOp(opValue, "1");
      }
      else {
        info = viType::isUseless;
      }
    }

    int valueInfo_t::hasStride(expNode &e) {
      for (int i = 0; i < indices; ++i) {
#if DBP3
        std::cout << "  strides[i] = " << strides[i] << '\n'
                  << "  e          = " << e.toString() << '\n';
#endif
        if (strides[i] == e)
          return i;
      }

      return -1;
    }

    int valueInfo_t::hasStride(const std::string &str) {
      for (int i = 0; i < indices; ++i) {
#if DBP1
        std::cout << "  strides[i] = " << strides[i] << '\n'
                  << "  str        = " << str << '\n';
#endif

        if (strides[i] == str)
          return i;
      }

      return -1;
    }

    int valueInfo_t::hasStride(atomInfo_t &stride_) {
      for (int i = 0; i < indices; ++i) {
        // if (strides[i] == stride) // [<>]
        //   return true;
      }

      return false;
    }

    bool valueInfo_t::hasComplexStride() {
      for (int i = 0; i < indices; ++i) {
        if (vars[i].info & viType::isAVariable) {
          viInfo_t *vi = db->has( *(vars[i].var) );

          if ((vi != NULL) &&
             (vi->info & viType::isAnIterator)) {

            return false;
          }

          return true;
        }
        else if (vars[i].info & (viType::isComplex |
                                viType::isUseless)) {

          return true;
        }
        if ( !(strides[i].info & viType::isConstant) ) {
          return true;
        }
      }

      return false;
    }

    bool valueInfo_t::stridesConflict() {
#if DBP4
      std::cout << "SC:V = " << *this << '\n';
#endif

      if (indices <= 1)
        return false;

      typeHolder *bounds;
      bool *hasBounds;

      setBoundInfo(bounds, hasBounds);

      bool checkLB[2], checkUB[2];
      int idx[2];

      for (int i = 0; i < indices; ++i) {
        idx[0]     = i;
        checkLB[0] = (hasBounds[3*i + analyzeInfo::LB] &&
                      hasBounds[3*i + analyzeInfo::S]);
        checkUB[0] =  hasBounds[3*i + analyzeInfo::UB];

        for (int j = (i + 1); j < indices; ++j) {
          idx[1]     = j;
          checkLB[1] = (hasBounds[3*j + analyzeInfo::LB] &&
                        hasBounds[3*j + analyzeInfo::S]);
          checkUB[1] =  hasBounds[3*j + analyzeInfo::UB];

          int fails = 0;

          for (int pass = 0; pass < 2; ++pass) {
            if (checkUB[pass] && checkLB[(pass + 1) % 2]) {
              int a = idx[pass];
              int b = idx[(pass + 1) % 2];

              typeHolder &aMax = bounds[3*a + analyzeInfo::UB];
              typeHolder  bMin = (bounds[3*b + analyzeInfo::LB] +
                                  bounds[3*b + analyzeInfo::S]);

              // [<>] Assumes for-loop as [<] operator, not [<=]
              fails += (bMin < aMax);
            }

            // Strides overlap
            if (fails == 2) {
              delete [] bounds;
              delete [] hasBounds;
              return true;
            }
          }
        }
      }

      delete [] bounds;
      delete [] hasBounds;

      return false;
    }

    // Assumption that (this and v) are not complex
    //   nor have conflicting strides
    bool valueInfo_t::conflictsWith(valueInfo_t &v) {
      return false; // [<>] test

#if DBP4
      std::cout << "A:V1 = " << *this << '\n'
                << "A:V2 = " << v << '\n';
#endif

      if ((indices == 0) || (v.indices == 0))
        return false;

      typeHolder *boundsA, *boundsB;
      bool *hasBoundsA, *hasBoundsB;

      setBoundInfo(boundsA, hasBoundsA);
      setBoundInfo(boundsB, hasBoundsB);

      delete [] boundsA;
      delete [] boundsB;
      delete [] hasBoundsA;
      delete [] hasBoundsB;

      return false;
    }

    void valueInfo_t::setBoundInfo(typeHolder *&bounds, bool *&hasBounds) {
      if (indices == 0) {
        bounds    = NULL;
        hasBounds = NULL;
        return;
      }

      bounds    = new typeHolder[3*indices];
      hasBounds = new bool[3*indices];

      for (int i = 0; i < 3*indices; ++i)
        hasBounds[i] = false;

      for (int i = 0; i < indices; ++i) {
        if (vars[i].info & viType::isConstant) {
          bounds[3*i + analyzeInfo::LB] = vars[i].constValue;
          bounds[3*i + analyzeInfo::UB] = vars[i].constValue;
          bounds[3*i + analyzeInfo::S]  = typeHolder((int) 0);
          hasBounds[3*i + analyzeInfo::LB] = true;
          hasBounds[3*i + analyzeInfo::UB] = true;
          hasBounds[3*i + analyzeInfo::S]  = true;
        }
        else { // Is an iterator
          viInfo_t &vi         = (*db)[*(vars[i].var)];
          iteratorInfo_t &iter = vi.iteratorInfo;

          valueInfo_t *iterBounds[3];
          iterBounds[analyzeInfo::LB] = &(iter.start);
          iterBounds[analyzeInfo::UB] = &(iter.end);
          iterBounds[analyzeInfo::S]  = &(iter.stride);

          for (int b = 0; b < 3; ++b) {
            if (iterBounds[b]->isConstant()) {
              bounds[3*i + b]    = (iterBounds[b]->constValue() *
                                    strides[i].constValue);
              hasBounds[3*i + b] = true;
            }
          }
        }
      }
    }

    void valueInfo_t::insertOp(const std::string &op,
                               expNode &value_) {
      expNode eOp;
      eOp.info  = expType::LR;
      eOp.value = op;

      eOp.addNodes(2);

      saveTo(eOp[1]);

      eOp.leaves[0] = &value_;

      load(eOp);
    }

    void valueInfo_t::insertOp(const std::string &op,
                               const std::string &value_) {

      expNode eValue;
      eValue.info  = expType::presetValue;
      eValue.value = value_;

      insertOp(op, eValue);
    }

    varInfo& valueInfo_t::varValue() {
      return *(value.var);
    }

    varInfo& valueInfo_t::var(const int pos) {
      return *(vars[pos].var);
    }

    atomInfo_t& valueInfo_t::stride(const int pos) {
      return strides[pos];
    }

    std::ostream& operator << (std::ostream &out, valueInfo_t &info) {
      if (info.indices == 0) {
        out << info.value;
      }
      else {
        for (int i = 0; i < info.indices; ++i) {
          if (i != 0)
            out << " + ";

          out << info.vars[i];

          if ( !(info.vars[i].info & viType::isUseless) )
            out << " (" << info.strides[i] << ')';
        }
      }

      return out;
    }

    accessInfo_t::accessInfo_t(infoDB_t *db_) :
      db(db_),
      s(NULL),
      dim(0),
      value(db_),
      dimIndices(NULL) {}

    accessInfo_t::accessInfo_t(const accessInfo_t &ai) :
      db(ai.db),
      s(ai.s),
      dim(ai.dim),
      value(ai.value),
      dimIndices(ai.dimIndices) {}

    accessInfo_t& accessInfo_t::operator = (const accessInfo_t &ai) {
      db = ai.db;
      s  = ai.s;

      dim        = ai.dim;
      value      = ai.value;
      dimIndices = ai.dimIndices;

      return *this;
    }

    void accessInfo_t::setDB(infoDB_t *db_) {
      db = db_;

      value.setDB(db_);

      for (int i = 0; i < dim; ++i)
        dimIndices[i].setDB(db_);
    }

    void accessInfo_t::load(expNode &varNode) {
      s = varNode.sInfo;

      dim = 0;
      value.load(varNode);
    }

    // [-] Fix
    void accessInfo_t::load(const int brackets, expNode &bracketNode) {
      s = bracketNode.sInfo;

      dim        = brackets;
      dimIndices = new valueInfo_t[dim];

      for (int i = 0; i < dim; ++i) {
        dimIndices[i].setDB(db);
        dimIndices[i].load(bracketNode[i][0]);
      }
    }

    bool accessInfo_t::hasComplexAccess() {
      if (dim <= 0)
        return false;

      for (int i = 0; i < dim; ++i) {
        if (dimIndices[i].hasComplexStride())
          return true;
      }

      return false;
    }

    bool accessInfo_t::stridesConflict() {
      for (int i = 0; i < dim; ++i) {
        if (dimIndices[i].stridesConflict())
          return true;
      }

      return false;
    }

    bool accessInfo_t::conflictsWith(accessInfo_t &ai) {
#if DBP4
      std::cout << "A1 = " << *this << '\n'
                << "A2 = " << ai << '\n';
#endif

      // Only check access infos
      if ((dim != ai.dim) ||
         (dim == 0)) {

        return false;
      }

      for (int i = 0; i < dim; ++i) {
        if (dimIndices[i].conflictsWith(ai.dimIndices[i]))
          return true;
      }

      return false;
    }

    std::ostream& operator << (std::ostream &out, accessInfo_t &info) {
      if (info.dim == 0) {
        out << '[' << info.value << ']';
      }
      else {
        for (int i = 0; i < info.dim; ++i)
          out << '[' << info.dimIndices[i] << ']';
      }

      return out;
    }

    iteratorInfo_t::iteratorInfo_t(infoDB_t *db_) :
      db(db_),
      s(NULL),
      start(db_),
      end(db_),
      stride(db_) {}

    bool iteratorInfo_t::operator == (iteratorInfo_t &iter) {
      return ((start  == iter.start) &&
              (end    == iter.end)   &&
              (stride == iter.stride));
    }

    bool iteratorInfo_t::operator != (iteratorInfo_t &iter) {
      return !(*this == iter);
    }

    void iteratorInfo_t::setDB(infoDB_t *db_) {
      db = db_;

      start.setDB(db);
      end.setDB(db);
      stride.setDB(db);
    }

    std::ostream& operator << (std::ostream &out, iteratorInfo_t &info) {
      out << "[Bounds: ["
          << info.start  << ", "
          << info.end    << "], Stride: "
          << info.stride;

      if (info.s)
        out << ", From: " << info.s->onlyThisToString();

      out << ']';

      return out;
    }

    viInfo_t::viInfo_t(infoDB_t *db_, varInfo *var_) :
      db(db_),
      var(var_),
      info(viType::isUseless),
      valueInfo(db_),
      iteratorInfo(db_) {}

    void viInfo_t::setDB(infoDB_t *db_) {
      db = db_;

      valueInfo.setDB(db);
      iteratorInfo.setDB(db);
    }

    bool viInfo_t::hasBeenInitialized() {
      return (writes.size() != 0);
    }

    void viInfo_t::addWrite(const bool isUpdated,
                            expNode &varNode) {
      if (db->isLocked())
        return;

      if (isUpdated)
        statementHasLCD(varNode.sInfo);

      accessInfo_t ai(db);
      ai.load(varNode);

      writes.push_back(ai);
      writeSetsValue.push_back(!isUpdated);
#if DBP0
      std::cout << "W1. ai = " << ai << '\n';
#endif
    }

    // [-] Fix
    void viInfo_t::addWrite(const bool isUpdated,
                            const int brackets, expNode &bracketNode) {
      if (db->isLocked())
        return;

      // No reduction check for array entries

      accessInfo_t ai(db);
      ai.load(brackets, bracketNode);

      writes.push_back(ai);
      writeSetsValue.push_back(!isUpdated);
#if DBP0
      std::cout << "W2. ai = " << ai << '\n';
#endif

      checkLastInput(ai, writeValue);
    }

    void viInfo_t::addRead(expNode &varNode) {
      if (db->isLocked())
        return;

      accessInfo_t ai(db);
      ai.load(varNode);

      reads.push_back(ai);
#if DBP0
      std::cout << "R1. ai = " << ai << '\n';
#endif
    }

    // [-] Fix
    void viInfo_t::addRead(const int brackets, expNode &bracketNode) {
      if (db->isLocked())
        return;

      accessInfo_t ai(db);
      ai.load(brackets, bracketNode);

      reads.push_back(ai);
#if DBP0
      std::cout << "R2. ai = " << ai << '\n';
#endif

      checkLastInput(ai, readValue);
    }

    void viInfo_t::updateValue(expNode &opNode, expNode &setNode) {
      if (opNode.value == "=") {
        // Fixes recursive issue
        valueInfo_t vi;
        vi.setDB(db);
        vi.load(setNode);

#if DBP0
        std::cout << "X1. valueInfo = " << vi << '\n';
#endif
        vi.expandValues();
#if DBP0
        std::cout << "X2. valueInfo = " << vi << '\n';
#endif
        valueInfo = vi;
      }
      else {
#if DBP0
        std::cout << "Y1. valueInfo = " << valueInfo << '\n';
#endif
        valueInfo.update(opNode, setNode);
#if DBP0
        std::cout << "Y2. valueInfo = " << valueInfo << '\n';
#endif
      }

      checkComplexity();
    }

    void viInfo_t::statementHasLCD(statement *sEnd) {
      if ((sEnd == NULL) ||
         (sEnd->info == smntType::forStatement)) {

        return;
      }

      statement *sStart = lastSetStatement();

      // Variable wasn't set, only label sEnd
      if (sStart == NULL) {
        db->statementsHaveLCD(sEnd);
        return;
      }

      // If the [up] statement is the same, nothing is wrong
      if (sStart->up == sEnd->up)
        return;

      statement *gcs = &(sStart->greatestCommonStatement(*sEnd));

#if DBP5
      if (gcs != NULL)
        std::cout << "GCS s:: " << gcs->onlyThisToString() << '\n';
      else
        std::cout << "GCS s:: NULL\n";
#endif

      db->statementsHaveLCD(gcs);
    }

    void viInfo_t::sharedStatementHaveLCD(statement *a, statement *b) {
      if ((a == NULL) || (b == NULL))
        return;

      statement *gcs = &(a->greatestCommonStatement(*b));

      if (gcs != NULL)
        std::cout << "GCS s:: " << gcs->onlyThisToString() << '\n';
      else
        std::cout << "GCS s:: NULL\n";

      db->statementsHaveLCD(gcs);
    }

    statement* viInfo_t::lastSetStatement() {
      const int writeCount = (int) writes.size();

      for (int i = (writeCount - 1); 0 <= i; --i) {
        if (writeSetsValue[i])
          return writes[i].s;
      }

      return NULL;
    }

    void viInfo_t::checkComplexity() {
      if (info & viType::isAnIterator)
        info &= ~viType::isComplex;
      else if (valueInfo.isComplex())
        info |= viType::isComplex;
      else
        info &= ~viType::isComplex;
    }

    void viInfo_t::checkLastInput(accessInfo_t &ai, const int inputType) {
      if (var->hasQualifier("occaShared") ||
         var->hasQualifier("exclusive")) {

        return;
      }

#if DBP4
      std::cout << "viInfo: " << *var << '\n';
#endif

      bool autoConflicts = false;

      if (info & viType::isComplex)
        autoConflicts = true;

      if (!autoConflicts && ai.hasComplexAccess()) {
#if DBP4
        std::cout << "Complex access: " << ai << '\n';
#endif
        info |= viType::isComplex;
        autoConflicts = true;
      }

      if (!autoConflicts && ai.stridesConflict()) {
#if DBP4
        std::cout << "Access strides overlap: " << ai << '\n';
#endif
        info |= viType::isComplex;
        autoConflicts = true;
      }

      const bool isWriting = (inputType == writeValue);

      std::vector<accessInfo_t> *pInputs[2] = {&writes, &reads};

      const int inputCounts[2] = {(int) writes.size() - isWriting,
                                  (int) reads.size()};

      const int checkInputs = 1 + isWriting;

      for (int i = 0; i < checkInputs; ++i) {
        std::vector<accessInfo_t> &inputs = *(pInputs[i]);
        const int inputCount = inputCounts[i];

        for (int j = 0; j < inputCount; ++j) {
          if (autoConflicts || inputs[j].conflictsWith(ai)) {
#if DBP4
            std::cout << "Access conflicts:\n"
                      << "  A1: " << ai        << '\n'
                      << "  A2: " << inputs[j] << '\n';
#endif
            sharedStatementHaveLCD(ai.s, inputs[j].s);
          }
        }
      }
    }

    std::ostream& operator << (std::ostream &out, viInfo_t &info) {
      if (info.info & viType::isAnIterator)
        out << info.iteratorInfo;
      else if (info.info & viType::isAVariable)
        out << info.valueInfo;

      return out;
    }

    viInfoMap_t::viInfoMap_t(infoDB_t *db_) :
      db(db_),
      anonVar(NULL) {}

    void viInfoMap_t::setDB(infoDB_t *db_) {
      db = db_;
    }

    void viInfoMap_t::free() {
      viInfoIterator it = viMap.begin();

      bool freedAnonVar = false;

      while(it != viMap.end()) {
        if ((it->second) != anonVar) {
          delete (it->second);
        }
        else {
          if (!freedAnonVar) {
            delete anonVar;
            freedAnonVar = true;
          }
        }

        ++it;
      }

      viMap.clear();
      anonVar = NULL;
    }

    void viInfoMap_t::add(varInfo &var) {
      viInfoIterator it = viMap.find(&var);

      if (it == viMap.end()) {
        if (var.hasQualifier("restrict")) {
          viMap[&var] = new viInfo_t(db, &var);
        }
        else {
          if (anonVar != NULL)
            viMap[&var] = anonVar;
          else
            viMap[&var] = new viInfo_t(db, &var);
        }
      }
    }

    viInfo_t* viInfoMap_t::has(varInfo &var) {
      viInfoIterator it = viMap.find(&var);

      return ((it == viMap.end()) ?
              NULL : it->second);
    }

    viInfo_t& viInfoMap_t::operator [] (varInfo &var) {
      viInfoIterator it = viMap.find(&var);

      if (it != viMap.end())
        return *(it->second);


      viInfo_t &vi = *(viMap[&var]);
      vi.setDB(db);

      return vi;
    }

    infoDB_t::infoDB_t() :
      locked(false),
      viInfoMap(this) {

      smntInfoStack.push(analyzeInfo::isExecuted);
    }

    void infoDB_t::lock() {
      locked = true;
    }

    void infoDB_t::unlock() {
      locked = false;
    }

    bool infoDB_t::isLocked() {
      return locked;
    }

    int& infoDB_t::getSmntInfo() {
      return smntInfoStack.top();
    }

    void infoDB_t::add(varInfo &var) {
      viInfoMap.add(var);
    }

    viInfo_t* infoDB_t::has(varInfo &var) {
      return viInfoMap.has(var);
    }

    void infoDB_t::enteringStatement(statement &s) {
      smntInfoStack.push(getSmntInfo());
    }

    void infoDB_t::leavingStatement() {
      smntInfoStack.pop();
    }

    viInfo_t& infoDB_t::operator [] (varInfo &var) {
      return *(viInfoMap.has(var));
    }

    bool infoDB_t::varIsAnIterator(varInfo &var) {
      viInfo_t *vi = viInfoMap.has(var);

      return ((vi != NULL) && (vi->info & viType::isAnIterator));
    }

    void infoDB_t::statementsHaveLCD(statement *s) {
      if (s == NULL)
        return;

#if DBP5
      std::cout << "LCD s:: " << s->onlyThisToString() << '\n';
#endif

      smntInfoMap[s] |= analyzeInfo::hasLCD;

      statementNode *sn = s->statementStart;

      while(sn) {
        statementsHaveLCD(sn->value);
        sn = sn->right;
      }
    }

    bool infoDB_t::statementHasLCD(statement &s) {
      return (smntInfoMap[&s] & analyzeInfo::hasLCD);
    }

    magician::magician(parserBase &parser_) :
      parser(parser_),
      globalScope( *(parser_.globalScope) ) {

      testedTileSizes.push_back(8);
      testedTileSizes.push_back(16);
      testedTileSizes.push_back(32);
      testedTileSizes.push_back(64);
      testedTileSizes.push_back(128);
      testedTileSizes.push_back(256);
    }

    void magician::castMagicOn(parserBase &parser_) {
      magician mickey(parser_);
      mickey.castMagic();
    }

    void magician::castMagic() {
      statementNode *sn = globalScope.statementStart;

      while(sn) {
        statement &s = *(sn->value);

        if (parser.statementIsAKernel(s))
          sn = analyzeFunction(s);
        else
          sn = sn->right;
      }
    }

    statementNode* magician::analyzeFunction(statement &fs) {
      varInfo &func = *(fs.getFunctionVar());

      db.enteringStatement(fs);

      // Place function arguments (if any)
      if (func.argumentCount) {
        for (int arg = 0; arg < func.argumentCount; ++arg) {
          varInfo &varg = *(func.argumentVarInfos[arg]);

          db.add(varg);
        }
      }

      statementNode *statementPos = fs.statementStart;

      while(statementPos) {
        analyzeStatement( *(statementPos->value) );

        statementPos = statementPos->right;
      }

      db.leavingStatement();

      db.lock();
      updateLoopBounds(fs);
      statementNode *ret = generatePossibleKernels(fs);
      db.unlock();

      return ret;
    }

    void magician::analyzeStatement(statement &s) {
      db.enteringStatement(s);

      if (s.info & smntType::declareStatement) {
        analyzeDeclareStatement(s.expRoot);
      }

      else if (s.info & smntType::updateStatement) {
        analyzeUpdateStatement(s.expRoot);
      }

      else if (s.info & smntType::forStatement) {
        if (parser.parsingLanguage & parserInfo::parsingC)
          analyzeForStatement(s);
        else
          analyzeFortranForStatement(s);
      }

      else if (s.info & smntType::whileStatement) {
        analyzeWhileStatement(s);
      }

      else if (s.info & smntType::doWhileStatement) {
        // do-while guarantees at least one run
        analyzeEmbeddedStatements(s);
        analyzeWhileStatement(s);
      }

      else if (s.info & smntType::ifStatement) {
        statementNode *snStart = s.getStatementNode();
        statementNode *snEnd   = snStart->right;

        while(snEnd                                         &&
              (snEnd->value->info &  smntType::ifStatement) &&
              (snEnd->value->info != smntType::ifStatement)) {

          snEnd = snEnd->right;
        }

        analyzeIfStatement(snStart, snEnd);
      }

      else if (s.info & smntType::switchStatement) {
        analyzeSwitchStatement(s);
      }

      else if (s.info & (smntType::typedefStatement   |
                        smntType::blankStatement     |
                        smntType::blockStatement     |
                        smntType::structStatement    |
                        smntType::functionStatement  |
                        smntType::functionDefinition |
                        smntType::functionPrototype)) {
        // Ignore this statement
      }

      else if (s.expRoot.info & expType::goto_) {
        printf("[Magic Analyzer] Goto statements are not supported\n");
      }

      db.smntInfoMap[&s] = db.getSmntInfo();

      if (db.getSmntInfo() & analyzeInfo::isExecuted)
        analyzeEmbeddedStatements(s);

      db.leavingStatement();
    }

    void magician::analyzeEmbeddedStatements(statement &s) {
      if (s.statementStart != NULL) {
        statementNode *statementPos = s.statementStart;

        while(statementPos) {
          analyzeStatement( *(statementPos->value) );

          statementPos = statementPos->right;
        }
      }
    }

    void magician::analyzeDeclareStatement(expNode &e) {
      const int varCount = e.getVariableCount();

      for (int i = 0; i < varCount; ++i) {
        // Add variable to the varInfo map
        varInfo &var = e.getVariableInfoNode(i)->getVarInfo();
        db.add(var);

        analyzeDeclareExpression(e, i);
      }
    }

    void magician::analyzeDeclareExpression(expNode &e, const int pos) {
      if (e.variableHasInit(pos)) {
        expNode &varNode  = *(e.getVariableInfoNode(pos));
        expNode &opNode   = *(e.getVariableOpNode(pos));
        expNode &initNode = *(e.getVariableInitNode(pos));

        addVariableWrite(varNode, opNode, initNode);
        addExpressionRead(initNode);

        viInfo_t &viInfo = db[ varNode.getVarInfo() ];
        viInfo.valueInfo.load(initNode);
      }
    }

    void magician::analyzeUpdateStatement(expNode &e) {
      const int upCount = e.getUpdatedVariableCount();

      for (int i = 0; i < upCount; ++i)
        analyzeUpdateExpression(e, i);
    }

    void magician::analyzeUpdateExpression(expNode &e, const int pos) {
      if (e.updatedVariableIsSet(pos)) {
        expNode &varNode = *(e.getUpdatedVariableInfoNode(pos));
        expNode &opNode  = *(e.getUpdatedVariableOpNode(pos));
        expNode &setNode = *(e.getUpdatedVariableSetNode(pos));

        addVariableWrite(varNode, opNode, setNode);
        addExpressionRead(setNode);
      }
      else
        addExpressionRead(e);
    }

    void magician::analyzeForStatement(statement &s) {
      expNode &initNode   = s.expRoot[0];
      expNode &checkNode  = s.expRoot[1];
      expNode &updateNode = s.expRoot[2];

      // [1] Add first node
      if (s.expRoot[0].info == expType::declaration)
        analyzeDeclareStatement(initNode);
      else
        analyzeUpdateStatement(initNode);

      // [3] Check update node
      bool wrongFormat = false;

      for (int i = 0; i < updateNode.leafCount; ++i) {
        expNode &leaf = updateNode[i];

        if (!(leaf.info & (expType::L_R |
                          expType::LR))) {
          wrongFormat = true;
          break;
        }
        else if (leaf.info & expType::LR) {
          if ((leaf.value != "+=") && (leaf.value != "-=")) {
            wrongFormat = true;
            break;
          }
        }
        else { // (leaf.info & expType::L_R)
          if ((leaf.value != "++") && (leaf.value != "--")) {
            wrongFormat = true;
            break;
          }
        }
      }

      if (wrongFormat) {
        printf("[Magic Analyzer] For-loop update statement (3rd statement) is not standard, for example:\n"
               "  X op Y where op can be [+=] or [-=]\n"
               "  ++X, X++, --X, X--\n");
        return;
      }

      varInfo *var    = NULL;
      expNode *stride = NULL;
      std::string str;

      for (int i = 0; i < updateNode.leafCount; ++i) {
        expNode &leaf = updateNode[i];

        if (leaf.info & expType::LR) {
          if ((leaf.value == "+=") ||
             (leaf.value == "-=")) {

            const bool varIn0 = (leaf[0].info & expType::varInfo);
            const bool varIn1 = (leaf[1].info & expType::varInfo);

            if (varIn0 ^ varIn1) {
              var = (varIn0 ?
                     &(leaf[0].getVarInfo()) :
                     &(leaf[1].getVarInfo()));

              stride = (varIn0 ? &(leaf[0]) : &(leaf[1]));
            }
          }
        }
        else if (leaf.info & expType::L_R) {
          if ((leaf.value == "++") ||
             (leaf.value == "--")) {

            var = &(leaf[0].getVarInfo());

            str = ((leaf.value == "++") ? "1" : "-1");
          }
        }

        viInfo_t &iter = db[*var];

        iter.info |=  viType::isAnIterator;
        iter.info &= ~viType::isUseless;
        iter.iteratorInfo.s = &s;

        if (stride)
          iter.iteratorInfo.stride.load(*stride);
        else
          iter.iteratorInfo.stride.load(str);
      }

      if (wrongFormat) {
        printf("[Magic Analyzer] For-loop update statement (3rd statement) is not standard, for example:\n"
               "  X op Y where op can be [+=] or [-=]\n"
               "  ++X, X++, --X, X--\n");
        return;
      }

      // [2] Find bounds [<>] Only works on [<] operator
      wrongFormat = true;

      if ((checkNode.leafCount == 1) &&
         isAnInequalityOperator(checkNode[0].value)) {

        const bool varIn0 = (checkNode[0][0].info & expType::varInfo);
        const bool varIn1 = (checkNode[0][1].info & expType::varInfo);

        viInfo_t *vi0 = (varIn0 ? db.has(checkNode[0][0].getVarInfo()) : NULL);
        viInfo_t *vi1 = (varIn1 ? db.has(checkNode[0][1].getVarInfo()) : NULL);

        const bool isIter0 = (vi0 && (vi0->info & viType::isAnIterator));
        const bool isIter1 = (vi1 && (vi1->info & viType::isAnIterator));

        if (isIter0 ^ isIter1) {
          viInfo_t &vi  = (isIter0 ? *vi0 : *vi1);
          expNode  &exp = (isIter0 ? checkNode[0][1] : checkNode[0][0]);

          if (isIter0) {
            vi.iteratorInfo.start = vi.valueInfo;
            vi.iteratorInfo.end.load(exp);
          }
          else {
            vi.iteratorInfo.start.load(exp);
            vi.iteratorInfo.end = vi.valueInfo;
          }

          wrongFormat = false;
        }
      }

      if (wrongFormat) {
        printf("[Magic Analyzer] For-loop update statement (2nd statement) is not standard, for example:\n"
               "  X op Y where:\n"
               "  X or Y is an iterator and op can be [<], [<=], [>=], or [>]\n");
        return;
      }

      expNode &flatRoot = *(checkNode.makeFlatHandle());
      viInfo_t *iter = NULL;

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        if (leaf.info & expType::varInfo) {
          varInfo &var2 = leaf.getVarInfo();

          viInfo_t *iter2 = db.has(var2);

          if (iter2 &&
             (iter2->info & viType::isAnIterator)) {

            if (iter == NULL) {
              iter = iter2;
            }
            else {
              expNode::freeFlatHandle(flatRoot);
              printf("[Magic Analyzer] For-loop update statement (2rd statement) is not standard:\n"
                     "  Multiple iterators were used\n");
              return;
            }
          }
        }
      }

      if (iter == NULL) {
        expNode::freeFlatHandle(flatRoot);
        printf("[Magic Analyzer] For-loop update statement (2rd statement) is not standard:\n"
               "  No iterator\n");
        return;
      }
#if DBP0
      std::cout << "iter = " << *iter << '\n';
#endif

      expNode::freeFlatHandle(flatRoot);
    }

    void magician::analyzeWhileStatement(statement &s) {
      typeHolder th = s.expRoot[0].calculateValue();

      if ( !(th.type & noType) &&
          (th == false) ) {

        db.getSmntInfo() &= ~analyzeInfo::isExecuted;
        return;
      }
    }

    void magician::analyzeFortranForStatement(statement &s) {
      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      expNode &e0 = s.expRoot[0][0];

      //      [0] root
      //       [0] =
      // [0] iter    [1] doStart
      varInfo &iterVar = e0[0].getVarInfo();
      viInfo_t &viInfo = db[iterVar];

      varInfo *start  = &(e0[1].getVarInfo());
      varInfo *end    = NULL;
      varInfo *stride = NULL;

      for (int j = 0; j < flatRoot.leafCount; ++j) {
        expNode &n = flatRoot[j];

        if (n.info & expType::varInfo) {
          varInfo &var = n.getVarInfo();

          if (var.name.find("doEnd") != std::string::npos)
            end = &var;
          else if (var.name.find("doStride") != std::string::npos)
            stride = &var;
        }
      }

      expNode::freeFlatHandle(flatRoot);

      viInfo.iteratorInfo.start.load(*start);
      viInfo.iteratorInfo.end.load(*end);

      if (stride == NULL)
        viInfo.iteratorInfo.stride.load("1");
      else
        viInfo.iteratorInfo.stride.load(*stride);
    }

    void magician::analyzeIfStatement(statementNode *snStart, statementNode *snEnd) {
      statementNode *sn = snStart;

      while(sn != snEnd) {
        statement &s = *(sn->value);

        if (s.expRoot.leafCount != 0) {
          typeHolder th = s.expRoot[0].calculateValue();

          if ( !(th.type & noType) &&
              (th == true) ) {

            analyzeEmbeddedStatements(s);

            return;
          }
        }

        sn = sn->right;
      }

      sn = snStart;

      while(sn != snEnd) {
        statement &s = *(sn->value);

        analyzeEmbeddedStatements(s);

        sn = sn->right;
      }
    }

    void magician::analyzeSwitchStatement(statement &s) {
      typeHolder th = s.expRoot[0].calculateValue();

      if (th.type & noType) {
        analyzeEmbeddedStatements(s);
        return;
      }

      statementNode *sn = s.statementStart;
      statementNode *calculateSN = NULL;

      while(sn) {
        statement &s2 = *(sn->value);

        if (s2.info & smntType::caseStatement) {
          if (s2.expRoot.leafCount) { // Not default
            if (th == s2.expRoot[0].calculateValue()) {
              calculateSN = sn;
              break;
            }
          }
          else {                    // Default case
            calculateSN = sn;
          }
        }

        sn = sn->right;
      }

      sn = calculateSN;

      // Analyize until break
      while(sn) {
        statement &s2 = *(sn->value);

        analyzeEmbeddedStatements(s2);

        if (statementGuaranteesBreak(s2))
          break;

        sn = sn->right;
      }
    }

    bool magician::statementGuaranteesBreak(statement &s) {
      return false;
    }

    statementNode* magician::generatePossibleKernels(statement &kernel) {
#if DBP5
      printIfLoopsHaveLCD(kernel);
#endif

      statementVector_t loopsVec;
      intVector_t depthVec;
      intVector_t outerLoopVec;
      intVecVector_t innerLoopVec;
      int depth = 0;

      // Initial vector
      innerLoopVec.push_back(intVector_t());

      storeLoopsAndDepths(kernel, loopsVec, depthVec, depth);

      const int loopCount = (int) loopsVec.size();

      for (int o = 0; o < loopCount; ++o) {
        const int oDepth = depthVec[o];

        // The deepest for-loop branch has all of
        //   the possible bounds
        int innerMostLoop  = 0;
        int innerMostDepth = 0;

        for (int i = (o + 1); i < loopCount; ++i) {
          int iDepth = depthVec[i];

          if (iDepth <= oDepth)
            break;

          if (innerMostDepth < iDepth) {
            innerMostDepth = iDepth;
            innerMostLoop  = i;
          }
        }

        // Always setup tiled kernels
        const int tileKernels = (int) testedTileSizes.size();

        for (int tk = 0; tk < tileKernels; ++tk) {
          outerLoopVec.push_back(o);
          innerLoopVec.push_back(intVector_t());
        }

        if (innerMostLoop == 0)
          continue;

        intVector_t orderVec;

        // Get the correct order
        for (int i = innerMostLoop; o < i; --i) {
          int iDepth = depthVec[i];

          // Going down the loop path
          if (innerMostDepth < iDepth)
            continue;
          else
            innerMostDepth = (iDepth - 1);

          orderVec.push_back(i);
        }

        const int innerLoopCount = orderVec.size();

        if (innerLoopCount == 0)
          continue;

        for (int i_ = (innerLoopCount - 1); 0 <= i_; --i_) {
          int i = orderVec[i_];

          storeInnerLoopCandidates(loopsVec,
                                   depthVec,
                                   o, i,
                                   innerLoopVec.back());

          if (innerLoopVec.back().size()) {
            outerLoopVec.push_back(o);
            innerLoopVec.push_back(intVector_t());
          }
        }
      }

      return generateKernelsAndLabelLoops(loopsVec,
                                          depthVec,
                                          outerLoopVec,
                                          innerLoopVec);
    }

    void magician::storeInnerLoopCandidates(statementVector_t &loopsVec,
                                            intVector_t &depthVec,
                                            int outerLoopIdx,
                                            int innerLoopIdx,
                                            intVector_t &innerLoopVec) {

      iteratorInfo_t iteratorInfo = iteratorLoopBounds(*(loopsVec[innerLoopIdx]));

      bool foundLoops = nestedLoopHasSameBounds(loopsVec,
                                                depthVec,
                                                outerLoopIdx,
                                                iteratorInfo,
                                                innerLoopVec);

      if (!foundLoops)
        innerLoopVec.clear();
    }

    void magician::storeNextDepthLoops(statementVector_t &loopsVec,
                                       intVector_t &depthVec,
                                       int loopIdx,
                                       intVector_t &ndLoopsVec) {

      const int loopCount = (int) loopsVec.size();
      const int depth    = depthVec[loopIdx];

      for (int i = (loopIdx + 1); i < loopCount; ++i) {
        int iDepth = depthVec[i];

        if (iDepth == depth)
          break;

        if (iDepth == (depth + 1))
          ndLoopsVec.push_back(i);
      }
    }

    bool magician::nestedLoopHasSameBounds(statementVector_t &loopsVec,
                                           intVector_t &depthVec,
                                           int ndLoopIdx,
                                           iteratorInfo_t &iteratorInfo,
                                           intVector_t &innerLoopVec,
                                           const bool isFirstCall) {

      statement &ndS = *(loopsVec[ndLoopIdx]);

      iteratorInfo_t ndIteratorInfo = iteratorLoopBounds(ndS);

      // First call doesn't return in the case that
      //   the inner-loop has the same bounds as the
      //   outer-loop
      if (!isFirstCall) {
        if (ndIteratorInfo == iteratorInfo) {
          innerLoopVec.push_back(ndLoopIdx);
          return true;
        }
      }

      intVector_t ndLoopsVec;

      storeNextDepthLoops(loopsVec,
                          depthVec,
                          ndLoopIdx,
                          ndLoopsVec);

      const int ndLoopCount = (int) ndLoopsVec.size();
      int b;

      for (b = 0; b < ndLoopCount; ++b) {
        if (!nestedLoopHasSameBounds(loopsVec,
                                    depthVec,
                                    ndLoopsVec[b],
                                    iteratorInfo,
                                    innerLoopVec,
                                    false)) {
#if DBP6
          std::cout << "loops have different bounds\n";
#endif
          break;
        }
      }

      if (b != ndLoopCount)
        return false;

      return true;
    }

    statementNode* magician::generateKernelsAndLabelLoops(statementVector_t &loopsVec,
                                                          intVector_t &depthVec,
                                                          intVector_t &outerLoopVec,
                                                          intVecVector_t &innerLoopVec) {
      const int kernelCount = outerLoopVec.size();
      statementVector_t newKernelVec;

      statement &kernel  = *(parserBase::getStatementKernel(*(loopsVec[0])));
      varInfo &kernelVar = *(kernel.getFunctionVar());
      statementNode *sn  = kernel.getStatementNode();

      if (kernelCount == 0)
        return sn->right;

#if DBP6
      std::cout << "Kernel: " << kernel.onlyThisToString() << '\n';
#endif

      statement &kGlobalScope = *(kernel.getGlobalScope());

      std::string kernelName = kernel.getFunctionName();
      kGlobalScope.createUniqueSequentialVariables(kernelName, kernelCount);

      std::stringstream ss;
      intVector_t path;

      int tileTest = 0;

      for (int k = 0; k < kernelCount; ++k) {
        ss.str("");
        ss << k;
        kernel.setFunctionName(kernelName + ss.str());

        sn = sn->push(kernel.clone());
        statement &newKernel = *(sn->value);
        newKernelVec.push_back(&newKernel);

        kernel.setFunctionName(kernelName);

        const int o   = outerLoopVec[k];
        statement &os = *(loopsVec[o]);

        os.setIndexPath(path, &kernel);
        statement &newOs = newKernel[path];

        if (newOs.hasAttribute("occaTag") == NULL)
          newOs.addAttribute("@(occaTag = outer, occaNest = 0)");

        intVector_t &innerLoopsVec = innerLoopVec[k];
        const int innerLoopCount  = innerLoopsVec.size();

        if (innerLoopCount != 0) {
          tileTest = 0;
        }
        else {
          newOs.removeAttribute("occaTag");
          newOs.removeAttribute("occaNest");

          newOs.addAttribute("@(occaTag = tile("
                             + toString(testedTileSizes[tileTest++])
                             + "))");

          continue;
        }

#if DBP6
        std::cout << "Outer Loop   : " << newOs.onlyThisToString() << '\n';
        std::cout << "Inner Loop(s):\n";
#endif

        for (int i_ = 0; i_ < innerLoopCount; ++i_) {
          const int i   = innerLoopsVec[i_];
          statement &is = *(loopsVec[i]);

          is.setIndexPath(path, &kernel);
          statement &newIs = newKernel[path];

          attribute_t *occaTagAttr = newIs.hasAttribute("occaTag");

          if (occaTagAttr == NULL)
            newIs.addAttribute("@(occaTag = inner, occaNest = 0)");

#if DBP6
          std::cout << "  " << i << ": " << newIs.onlyThisToString() << '\n';
#endif
        }
#if DBP6
        std::cout << '\n';
#endif
      }

      statementNode *retSn = sn->right;

      sn = kernel.statementStart;

      while(sn)
        popAndGoRight(sn);

      kernel.statementStart = NULL;
      kernel.statementEnd   = NULL;

      std::string syncArgsStr, kernelArgs;

      for (int i = 0; i < kernelVar.argumentCount; ++i) {
        varInfo &argVar = kernelVar.getArgument(i);

        if (0 < argVar.pointerCount) {
          ss.str("");
          ss << "occa::setupMagicFor(" << argVar.name << ");";
          kernel.addStatementFromSource(ss.str());

          ss.str("");
          ss << "occa::syncToDevice(" << argVar.name << ");";
          syncArgsStr += ss.str();
        }
      }

      ss.str("");

      ss << '(';

      for (int i = 0; i < kernelVar.argumentCount; ++i) {
        if (i) ss << ',';
        ss << kernelVar.getArgument(i).name;
      }

      ss << ");";

      kernelArgs = ss.str();

      for (int k = 0; k < kernelCount; ++k) {
        ss.str("");
        ss << k;

        std::string kStr = ss.str();

        kernel.addStatementFromSource(kernelName + kStr + kernelArgs);

        if (k < (kernelCount - 1))
          kernel.addStatementsFromSource(syncArgsStr);
      }

      return retSn;
    }

    void magician::storeLoopsAndDepths(statement &s,
                                       statementVector_t &loopsVec,
                                       intVector_t &depthVec, int depth) {

      statementNode *sn = s.statementStart;

      while(sn) {
        statement &s2 = *(sn->value);

        if (s2.info == smntType::forStatement) {
          loopsVec.push_back(&s2);
          depthVec.push_back(depth);
          storeLoopsAndDepths(s2, loopsVec, depthVec, depth + 1);
        }
        else
          storeLoopsAndDepths(s2, loopsVec, depthVec, depth);

        sn = sn->right;
      }
    }

    iteratorInfo_t magician::iteratorLoopBounds(statement &s) {
      const bool dbWasLocked = db.isLocked();
      db.lock();

      iteratorInfo_t iteratorInfo;

      expNode &flatRoot = *(s.expRoot[2].makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        if (leaf.info & expType::varInfo) {
          varInfo &var = leaf.getVarInfo();

          if (db.varIsAnIterator(var)) {
            viInfo_t &vi = db[var];

            iteratorInfo = vi.iteratorInfo;
            break;
          }
        }
      }

      expNode::freeFlatHandle(flatRoot);

      if (!dbWasLocked)
        db.unlock();

      return iteratorInfo;
    }

    void magician::updateLoopBounds(statement &s) {
      statementNode *sn = s.statementStart;

      while(sn) {
        statement &s2 = *(sn->value);

        if (s2.info == smntType::forStatement) {
          if (parser.parsingLanguage & parserInfo::parsingC)
            analyzeForStatement(s2);
          else
            analyzeFortranForStatement(s2);

          updateLoopBounds(s2);
        }
        else
          updateLoopBounds(s2);

        sn = sn->right;
      }
    }

    void magician::printIfLoopsHaveLCD(statement &s) {
      if (s.info == smntType::forStatement)
        std::cout << "LCD(" << db.statementHasLCD(s) << "): " << s.onlyThisToString() << '\n';

      statementNode *sn = s.statementStart;

      while(sn) {
        printIfLoopsHaveLCD(*(sn->value));
        sn = sn->right;
      }
    }

    void magician::addVariableWrite(expNode &varNode, expNode &opNode, expNode &setNode) {
      const bool isUpdated = (isAnUpdateOperator(opNode.value) &&
                              (opNode.value != "="));

      // [-] Fix
      if (varNode.info & expType::variable) {
        const int brackets = varNode.getVariableBracketCount();

        if (brackets)
          addVariableWrite(varNode, opNode, setNode,
                           brackets, *(varNode.getVariableBracket(0)->up));

        return;
      }

      if (isUpdated)
        addVariableRead(varNode);

      viInfo_t &viInfo = db[ varNode.getVarInfo() ];

      viInfo.addWrite(isUpdated, varNode);
      viInfo.updateValue(opNode, setNode);
    }

    // [-] Fix
    void magician::addVariableWrite(expNode &varNode,
                                    expNode &opNode,
                                    expNode &setNode,
                                    const int brackets,
                                    expNode &bracketNode) {

      const bool isUpdated = (isAnUpdateOperator(opNode.value) &&
                              (opNode.value != "="));

      if (isUpdated)
        addVariableRead(varNode, brackets, bracketNode);

      viInfo_t &viInfo = db[ varNode[0].getVarInfo() ];

      viInfo.addWrite(isUpdated, brackets, bracketNode);
    }

    void magician::addVariableRead(expNode &varNode) {
      // [-] Fix
      if (varNode.info & expType::variable) {
        const int brackets = varNode.getVariableBracketCount();

        if (brackets)
          addVariableRead(varNode, brackets, *(varNode.getVariableBracket(0)->up));

        return;
      }

      viInfo_t &viInfo = db[ varNode.getVarInfo() ];

      viInfo.addRead(varNode);
    }

    // [-] Fix
    void magician::addVariableRead(expNode &varNode,
                                   const int brackets,
                                   expNode &bracketNode) {

      viInfo_t &viInfo = db[ varNode[0].getVarInfo() ];

      viInfo.addRead(brackets, bracketNode);
    }

    void magician::addExpressionRead(expNode &e) {
      // [-] Fix
      if (e.info & expType::variable) {
        const int brackets = e.getVariableBracketCount();

        if (brackets)
          addVariableRead(e, brackets, *(e.getVariableBracket(0)->up));
      }
      else if ((e.info & expType::varInfo) &&
              ((e.up == NULL) ||
               !(e.up->info & expType::variable))) {

        addVariableRead(e);
      }
      else {
        for (int i = 0; i < e.leafCount; ++i)
          addExpressionRead(e[i]);
      }
    }

    //---[ Helper Functions ]---------
    void magician::placeAddedExps(infoDB_t &db, expNode &e, expVector_t &addedExps) {
      placeExps(db, e, addedExps, "+-");
    }

    void magician::placeMultExps(infoDB_t &db, expNode &e, expVector_t &multExps) {
      placeExps(db, e, multExps, "*");
    }

    void magician::placeExps(infoDB_t &db, expNode &e, expVector_t &exps, const std::string &delimiters) {
      expVector_t opNodes;

      if (!expHasOp(e, delimiters))
        exps.push_back(&e);
      else
        opNodes.push_back(&e);

      while(opNodes.size()) {
        expVector_t opNodes2;
        const int snc = opNodes.size();

        for (int i = 0; i < snc; ++i) {
          expNode &se = *(opNodes[i]);

          for (int j = 0; j < 2; ++j) {
            if (expHasOp(se[j], delimiters))
              opNodes2.push_back( &(se[j]) );
            else
              exps.push_back( &(se[j]) );
          }
        }

        opNodes.swap(opNodes2);
        opNodes2.clear();
      }
    }

    bool magician::expHasOp(expNode &e, const std::string &delimiters) {
      const std::string &eValue = e.value;

      if ((eValue.size() != 1) ||
         !(e.info & expType::LR)) {

        return false;
      }

      const int dCount = (int) delimiters.size();

      for (int i = 0; i < dCount; ++i) {
        if (eValue[0] == delimiters[i])
          return true;
      }

      return false;
    }

    void magician::simplify(infoDB_t &db, expNode &e) {
      removePermutations(db, e);
      turnMinusIntoNegatives(db, e);
      expandExp(db, e);
      mergeConstants(db, e);
      // mergeVariables(db, e);
    }

    void magician::removePermutations(infoDB_t &db, expNode &e) {
      expNode &flatRoot = *(e.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        // [-] Fix
        if (leaf.info & expType::variable) {
          varInfo &var = leaf[0].getVarInfo();

          if (var.hasAttribute("permutation")) {
            leaf.info = expType::root;
            expNode::swap(leaf, leaf[1][0][0]);
          }
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void magician::turnMinusIntoNegatives(infoDB_t &db, expNode &e) {
      expNode &flatRoot = *(e.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        if (!(leaf.info & expType::LR) &&
           (leaf.value == "-")) {

          leaf.value = "+";

          expNode &minus = *(new expNode(leaf));

          minus.info  = expType::L;
          minus.value = "-";

          minus.reserve(1);
          minus.setLeaf(leaf[1], 0);

          leaf.setLeaf(minus, 1);
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void magician::mergeConstants(infoDB_t &db, expNode &e) {
      expNode &flatRoot = *(e.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        mergeConstantsIn(db, leaf);
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void magician::mergeConstantsIn(infoDB_t &db, expNode &e) {
      if (e.leafCount <= 1)
        return;

      expVector_t sums, sums2;

      typeHolder constValue = 0;
      bool hasConst = false;

      placeAddedExps(db, e, sums);

      const int sumCount = (int) sums.size();

      for (int i = 0; i < sumCount; ++i) {
        expNode &leaf = *(sums[i]);

        typeHolder th = leaf.calculateValue();

        if ( !(th.type & noType) ) {
          // leaf.free(); // [<>]

          if (hasConst) {
            constValue += th;
          }
          else {
            constValue = th;
            hasConst   = true;
          }
        }
        else {
          applyConstantsIn(db, leaf);
          sums2.push_back(e.clonePtr());
        }
      }

      if (hasConst) {
        expNode &leaf = *(new expNode( e.makeFloatingLeaf() ));

        leaf.info  = expType::presetValue;
        leaf.value = (std::string) constValue;

        sums2.push_back(&leaf);
      }

      const int sumCount2 = (int) sums2.size();

      if (sumCount2 == 1) {
        e.freeThis();
        expNode::swap(e, *(sums2[0]));
        return;
      }

      expNode *cNode = &e;
      e.freeThis();

      for (int i = 0; i < sumCount2; ++i) {
        const bool lastI = (i == (sumCount2 - 1));

        expNode &leaf = *cNode;

        if ((sums2[i]->info & expType::presetValue) &&
           (sums2[i]->value == "0")) {

          if (lastI)
            expNode::swap(leaf, leaf[0]);

          continue;
        }

        if (!lastI) {
          leaf.info  = expType::LR;
          leaf.value = "+";

          leaf.addNodes(2);
        }

        leaf.leaves[lastI] = sums2[i];

        if (i < (sumCount2 - 2))
          cNode = &(leaf[1]);
      }
    }

    void magician::applyConstantsIn(infoDB_t &db, expNode &e) {
      if (!(e.info & expType::LR))
        return;

      expVector_t v, v2, constValues;

      v.push_back(&e);

      while(v.size()) {
        const int vCount = (int) v.size();

        for (int i = 0; i < vCount; ++i) {
          expNode &leaf = *(v[i]);

          if ((leaf.info  &  expType::LR) &&
             (leaf.value == "*")) {

            int jConsts = 0;

            for (int j = 0; j < 2; ++j) {
              if ((leaf[j].info  & expType::LR) &&
                 (leaf[j].value == "*")) {

                v2.push_back( &(leaf[j]) );
              }
              else if (leaf[j].info & expType::presetValue) {
                constValues.push_back( &(leaf[j]) );
                ++jConsts;

                if (jConsts == 2)
                  leaf[j].up = NULL;
              }
            }
          }
        }

        v.swap(v2);
        v2.clear();
      }

      const int constCount = (int) constValues.size();

      if (constCount == 0)
        return;

      typeHolder constValue(constValues[0]->value);

      for (int i = 1; i < constCount; ++i) {
        expNode &leaf = *(constValues[i]);

        // The other leaf is taking care of this
        if ((leaf.up == NULL) || (leaf.up == &leaf)) // [<>] ???
          continue;

        expNode &leafUp = *(leaf.up);
        expNode &leaf2  = leafUp[!leaf.whichLeafAmI()];

        constValue *= leaf.value;

        expNode::swap(leafUp, leaf2);

        leaf2.freeThis();
        delete &leaf2;
      }

      if (constValue == typeHolder((int) 0)) {
        e.free();

        e.info  = expType::presetValue;
        e.value = (std::string) constValue;
      }
      else if (1 < constCount) {
        constValues[0]->value = (std::string) constValue;

        if (constValue == typeHolder((int) 1)) {
          expNode &leaf   = *(constValues[0]);
          expNode &leafUp = *(leaf.up);
          expNode &leaf2  = leafUp[!leaf.whichLeafAmI()];

          expNode::swap(leafUp, leaf2);

          leaf2.freeThis();
          delete &leaf2;
        }
      }
    }

    void magician::mergeVariables(infoDB_t &db, expNode &e) {
      std::cout << "mergeVariables\n";

      expVector_t sums, *mults;
      placeAddedExps(db, e, sums);

      const int sumCount = (int) sums.size();

      mults = new expVector_t[sumCount];

      printf("-----------------------------------\n");
      for (int i = 0; i < sumCount; ++i) {
        expNode &leaf = *(sums[i]);

        placeMultExps(db, leaf, mults[i]);
      }

      for (int i1 = 0; i1 < sumCount; ++i1) {
        expVector_t &mult1 = mults[i1];
        expVector_t iter1  = iteratorsIn(db, mult1);

        if (iter1.size() == 0)
          continue;

        for (int i2 = (i1 + 1); i2 < sumCount; ++i2) {
          expVector_t &mult2 = mults[i2];
          expVector_t iter2  = iteratorsIn(db, mult2);

          if (iter2.size() == 0)
            continue;

          if (iteratorsMatch(iter1, iter2)) {
            expVector_t nonIter1 = removeItersFromExpVec(mult1, iter1);
            expVector_t nonIter2 = removeItersFromExpVec(mult2, iter2);

            printf("HERE\n");
            expNode::printVec(mult1);
            printf("  1:\n");
            expNode::printVec(iter1);
            printf("  2:\n");
            expNode::printVec(nonIter1);
            printf("HERE\n");
            expNode::printVec(mult2);
            printf("  1:\n");
            expNode::printVec(iter2);
            printf("  2:\n");
            expNode::printVec(nonIter2);

            nonIter1.insert(nonIter1.end(), nonIter1.begin(), nonIter1.end());

            expNode zipIter, zipNonIter;

            multiplyExpVec(iter1, zipIter);
            sumExpVec(nonIter1, zipNonIter);

            expNode sum2;
            sum2.info  = expType::LR;
            sum2.value = "*";

            sum2.reserve(2);
            sum2.setLeaf(zipIter   , 0);
            sum2.setLeaf(zipNonIter, 1);

            expNode::swap(*(sums[i2]), sum2);

            sums.erase(sums.begin() + i1);
          }
        }
      }
      printf("===================================\n");
    }

    void magician::expandExp(infoDB_t &db, expNode &e) {
      expNode &flatRoot = *(e.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        if ((leaf.info  &  expType::LR) &&
           (leaf.value == "*")) {

          expandMult(db, leaf);
        }
        else if ((leaf.info  & expType::C) &&
                (leaf.value == "(")) {

          removeParentheses(db, leaf);
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void magician::expandMult(infoDB_t &db, expNode &e) {
      const std::string op = e.value;

      expVector_t a, b;
      expNode tmp;

      placeAddedExps(db, e[0], a);

      if (op == "*")
        placeAddedExps(db, e[1], b);
      else // (op == "/")
        b.push_back( &(e[1]) );

      const int aCount  = (int) a.size();
      const int bCount  = (int) b.size();
      const int abCount = (aCount * bCount);

      if (abCount == 1)
        return;

      tmp.addNodes(expType::LR, 0, abCount);

      int pos = 0;

      for (int i = 0; i < aCount; ++i) {
        for (int j = 0; j < bCount; ++j) {
          expNode &leaf = tmp[pos++];

          leaf.value = "+";

          leaf.reserve(2);

          leaf.setLeaf(*(a[i]->clonePtr()), 0);
          leaf.setLeaf(*(b[j]->clonePtr()), 1);
        }
      }

      expNode *cNode = &e;
      e.freeThis();

      for (int i = 0; i < abCount; ++i) {
        const bool lastI = (i == (abCount - 1));

        expNode &e2   = *cNode;
        expNode &leaf = tmp[i];

        if (!lastI) {
          expNode &nextLeaf = tmp[i + 1];

          e2.info  = nextLeaf.info;
          e2.value = nextLeaf.value;

          e2.addNodes(2);
        }

        e2[lastI].info  = expType::LR;
        e2[lastI].value = op;

        e2[lastI].reserve(2);

        e2[lastI].setLeaf(leaf[0], 0);
        e2[lastI].setLeaf(leaf[1], 1);

        leaf.freeThis();

        if (i < (abCount - 2))
          cNode = &(e2[1]);
      }

      tmp.freeThis();
    }

    void magician::removeParentheses(infoDB_t &db, expNode &e) {
      expNode &leaf = e[0];
      e.freeThis();

      expNode::swap(e, leaf);
    }

    expVector_t magician::iteratorsIn(infoDB_t &db, expVector_t &v) {
      expVector_t iterV;

      const int vCount = (int) v.size();

      std::cout << "II: v = \n";
      expNode::printVec(v);

      for (int i = 0; i < vCount; ++i) {
        expNode &leaf = *(v[i]);

        if (leaf.info & expType::varInfo) {
          viInfo_t *vi = db.viInfoMap.has(leaf.getVarInfo());
          if (vi != NULL)
            std::cout << "vi = " << *vi << '\n';
        }

        if ((leaf.info & expType::varInfo) &&
           db.varIsAnIterator(leaf.getVarInfo())) {

          iterV.push_back(&leaf);
        }
      }

      std::cout << "II: iters = \n";
      expNode::printVec(iterV);

      return iterV;
    }

    bool magician::iteratorsMatch(expVector_t &a, expVector_t &b) {
      const int aCount = (int) a.size();
      const int bCount = (int) b.size();

      if (aCount != bCount)
        return false;

      for (int i = 0; i < aCount; ++i) {
        expNode &aLeaf = *(a[i]);
        int clones[2] = {1,0};

        // Pass 0: Finds clones in a
        // Pass 1: Finds clones in b
        for (int pass = 0; pass < 2; ++pass) {
          for (int j = 0; j < aCount; ++j) {
            if ((i == j) && (pass == 0))
              continue;

            expNode &leaf = (pass ? *(b[j]) : *(a[j]));

            if (&(aLeaf.getVarInfo()) ==
               &(leaf.getVarInfo())) {

              // Only the first instance will check
              if ((j < i) && (pass == 0)) {
                clones[pass] = 0;
                break;
              }

              ++clones[pass];
            }
          }

          if (clones[0] == 0)
            break;
        }

        if (clones[0] == 0)
          continue;

        if (clones[0] != clones[1])
          return false;
      }

      return true;
    }

    expVector_t magician::removeItersFromExpVec(expVector_t &v, expVector_t &iters) {
      const int vCount    = (int) v.size();
      const int iterCount = (int) iters.size();

      expVector_t ret;

      if (iters.size() == 0)
        return ret;;

      std::cout << "v     = \n";
      expNode::printVec(v);
      std::cout << "iters = \n";
      expNode::printVec(iters);

      for (int i = 0; i < vCount; ++i) {
        expNode &leaf = *(v[i]);

        if ( !(leaf.info & expType::varInfo) ) {
          ret.push_back(&leaf);
          continue;
        }

        int clones[2] = {1,0};

        // Pass 0: Finds clones in v
        // Pass 1: Finds clones in iters
        for (int pass = 0; pass < 2; ++pass) {
          int count = (pass ? iterCount : vCount);

          for (int j = 0; j < count; ++j) {
            if ((i == j) && (pass == 0))
              continue;

            expNode &leaf2 = (pass ? *(iters[j]) : *(v[j]));

            if ( !(leaf2.info & expType::varInfo) )
              continue;

            if (&(leaf.getVarInfo()) ==
               &(leaf2.getVarInfo())) {

              // Only the first instance will check
              if ((j < i) && (pass == 0)) {
                clones[0] = 0;
                break;
              }

              ++clones[pass];
            }
          }

          if (clones[0] == 0)
            break;
        }

        if (clones[0] == 0)
          continue;

        std::cout << "clones[0] = " << clones[0] << '\n'
                  << "clones[1] = " << clones[1] << '\n';

        for (int j = clones[1]; j < clones[0]; ++j)
          ret.push_back(&leaf);
      }

      return ret;
    }

    void magician::multiplyExpVec(expVector_t &v, expNode &e) {
      applyOpToExpVec(v, e, "*");
    }

    void magician::sumExpVec(expVector_t &v, expNode &e) {
      applyOpToExpVec(v, e, "+");
    }

    void magician::applyOpToExpVec(expVector_t &v, expNode &e, const std::string &op) {
      const int vCount = (int) v.size();
      expNode *cNode = &e;

      if (vCount == 1) {
        expNode::swap(e, *(v[0]->clonePtr()));
        return;
      }

      for (int i = (vCount - 1); 0 <= i; --i) {
        if (0 < i)
          cNode->reserve(2);

        cNode->info  = expType::LR;
        cNode->value = op;

        cNode->setLeaf(*(v[i]), (0 < i));

        if (1 < i)
          cNode = cNode->leaves[1];
      }
    }
  }
}
