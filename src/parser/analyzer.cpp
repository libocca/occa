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

#include "occa/parser/analyzer.hpp"
#include "occa/parser/parser.hpp"

namespace occa {
  namespace parserNS {
    //---[ Variable Dependencies ]----------------
    varDepInfo::varDepInfo() :
      info(depType::none),
      var(NULL),
      myNode(NULL),
      subNode(NULL) {}

    void varDepInfo::setup(int info_,
                           varInfo &var_,
                           smntDepInfo &sdInfo_,
                           varDepInfoNode &myNode_) {
      info = info_;
      var  = &var_;

      sdInfo = &sdInfo_;

      myNode        = &myNode_;
      myNode->value = this;
    }

    int varDepInfo::startInfo() {
      if (subNode == NULL)
        return info;

      return subNode->value->info;
    }

    int varDepInfo::endInfo() {
      if (subNode == NULL)
        return info;

      return lastNode(subNode)->value->info;
    }
    //============================================


    //---[ Statement Dependencies ]---------------
    smntDepInfo::smntDepInfo() :
      s(NULL),

      myNode(NULL),
      subNode(NULL) {}

    void smntDepInfo::setup(statement &s_, smntDepInfoNode &myNode_) {
      s = &s_;

      myNode        = &myNode_;
      myNode->value = this;

      expNode &flatRoot = *(s_.expRoot.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        if ((leaf.info & expType::varInfo) == 0)
          continue;

        int leafDepInfo = getDepTypeFrom(leaf);

        varInfo &var = leaf.getVarInfo();

        varDepInfoNode *&vdNode = v2dMap[&var];

        varDepInfo &newVdInfo = *(new varDepInfo);
        newVdInfo.setup(leafDepInfo, var, *this, *(new varDepInfoNode));

        if ((leafDepInfo & (depType::set |
                           depType::update)) &&
           (1 < leaf.leafCount)) {

          updateDependenciesFor(newVdInfo, leaf[1]);
        }

        if (vdNode == NULL) {
          vdNode = newVdInfo.myNode;

          setupNestedVdInfos(s_, var, vdNode);
        }
        else {
          varDepInfo &vdInfo = *(vdNode->value);

          // Copy vdInfo to the first subNode
          if (vdInfo.subNode == NULL) {
            varDepInfo &vdInfo2 = *(new varDepInfo);
            vdInfo2.setup(vdInfo.info, var, *this, *(new varDepInfoNode));

            vdInfo.info    = depType::none;
            vdInfo.subNode = vdInfo2.myNode;
          }

          varDepInfoNode *endNode = lastNode(vdInfo.subNode);

          endNode->push(newVdInfo.myNode);
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    void smntDepInfo::setupNestedVdInfos(statement &s_,
                                         varInfo &var,
                                         varDepInfoNode *vdNode) {

      statement &sOrigin = *(s_.parser.varOriginMap[&var]);

      smntDepInfoNode *sdNode = myNode;

      while((sdNode           != NULL) &&
            (sdNode->value->s != &sOrigin)) {

        smntDepInfo &sdInfo = *(sdNode->value);
        varDepInfo *vdInfo2 = sdInfo.has(var);

        if (vdInfo2 != NULL) {
          varDepInfoNode &vdNode2 = *(vdInfo2->myNode);

          if (vdNode2.down == NULL)
            vdNode2.pushDown(vdNode);
          else
            lastNode(vdNode2.down)->push(vdNode);

          break;
        }
        else {
          varDepInfoNode *&vdNodeUp = sdInfo.v2dMap[&var];

          varDepInfo &vdInfoUp = *(new varDepInfo);
          vdInfoUp.setup(depType::none, var, sdInfo, *(new varDepInfoNode));

          vdNodeUp = vdInfoUp.myNode;

          vdNodeUp->pushDown(vdNode);

          vdNode = vdNodeUp;
        }

        sdNode = sdNode->up;
      }
    }

    int smntDepInfo::getDepTypeFrom(expNode &e) {
      if ((e.up == NULL) ||
         ((e.up->info & expType::operator_) == 0)) {

        return depType::used;
      }

      expNode &eUp = *(e.up);

      const bool isUpdated = isAnUpdateOperator(eUp.value);

      if (isUpdated) {
        const bool varIsUpdated = (eUp.leaves[0] == &e);

        if (varIsUpdated) {
          const bool isSet = (eUp.value == "=");

          if (isSet)
            return depType::set;
          else
            return depType::update;
        }
        else
          return depType::used;
      }

      return depType::used;
    }

    void smntDepInfo::updateDependenciesFor(varDepInfo &vdInfo,
                                            expNode &updateExp) {

      expNode &flatRoot = *(updateExp.makeFlatHandle());

      for (int i = 0; i < flatRoot.leafCount; ++i) {
        expNode &leaf = flatRoot[i];

        if ((leaf.info & expType::varInfo) == 0)
          continue;

        varInfo &var = leaf.getVarInfo();

        vdInfo.varDeps[&var] = 0;
      }

      expNode::freeFlatHandle(flatRoot);
    }

    varDepInfo* smntDepInfo::has(varInfo &var) {
      varToDepMapIterator it = v2dMap.find(&var);

      if (it == v2dMap.end())
        return NULL;

      return (it->second->value);
    }

    varDepInfo& smntDepInfo::operator () (varInfo &var) {
      return *(has(var));
    }

    int smntDepInfo::startInfo(varInfo &var) {
      varDepInfo *vdInfo = has(var);

      if (vdInfo == NULL)
        return depType::none;

      return vdInfo->startInfo();
    }

    int smntDepInfo::endInfo(varInfo &var) {
      varDepInfo *vdInfo = has(var);

      if (vdInfo == NULL)
        return depType::none;

      return vdInfo->endInfo();
    }
    //============================================


    //---[ Dependency Map ]-----------------------
    depMap_t::depMap_t(statement &s) :
      parser(s.parser) {
      setup(s);
    }

    void depMap_t::setup(statement &s) {
      setup(s, *(new smntDepInfo));
    }

    void depMap_t::setup(statement &s, smntDepInfo &sdInfo) {
      smntDepInfoNode &sdNode = *(new smntDepInfoNode);
      sdNode.value = &sdInfo;

      sdInfo.setup(s, sdNode);

      s2vdMap[&s] = &sdInfo;

      statementNode *sn = s.statementStart;

      while(sn) {
        statement &s2 = *(sn->value);

        setup(s2, *(new smntDepInfo));

        sn = sn->right;
      }
    }

    varDepInfo* depMap_t::has(varInfo &var) {
      statement   &s      = *(parser.varOriginMap[&var]);
      smntDepInfo *sdInfo = has(s);

      if (sdInfo == NULL)
        return NULL;

      return sdInfo->has(var);
    }

    varDepInfo& depMap_t::operator () (varInfo &var) {
      return *(has(var));
    }

    smntDepInfo* depMap_t::has(statement &s) {
      smntToVarDepMapIterator it = s2vdMap.find(&s);

      if (it == s2vdMap.end())
        return NULL;

      return (it->second);
    }

    smntDepInfo& depMap_t::operator () (statement &s) {
      return *(has(s));
    }

    varDepInfo* depMap_t::has(statement &s, varInfo &var) {
      smntToVarDepMapIterator it = s2vdMap.find(&s);

      if (it == s2vdMap.end())
        return NULL;

      return (it->second->has(var));
    }

    varDepInfo& depMap_t::operator () (statement &s, varInfo &var) {
      return *(has(s,var));
    }
    //============================================
  }
}
