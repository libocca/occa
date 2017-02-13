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

#ifndef OCCA_PARSER_ANALYZER_HEADER
#define OCCA_PARSER_ANALYZER_HEADER

#include "occa/defines.hpp"
#include "occa/parser/statement.hpp"

namespace occa {
  namespace parserNS {
    class varDepInfo;
    class smntDepInfo;

    typedef node<varDepInfo*>                  varDepInfoNode;
    typedef node<smntDepInfo*>                 smntDepInfoNode;

    typedef std::map<varInfo*,varDepInfoNode*> varToDepMap;
    typedef varToDepMap::iterator              varToDepMapIterator;

    typedef std::map<statement*,smntDepInfo*>  smntToVarDepMap;
    typedef smntToVarDepMap::iterator          smntToVarDepMapIterator;

    namespace depType {
      static const int none   = 0;
      static const int set    = (1 << 0);
      static const int update = (1 << 1);
      static const int used   = (1 << 2);
    }

    //---[ Variable Dependencies ]----------------
    class varDepInfo {
    public:
      int info;

      varInfo *var;

      smntDepInfo *sdInfo;

      varDepInfoNode *myNode, *subNode;

      varInfoIdMap_t varDeps;

      varDepInfo();

      void setup(int info_,
                 varInfo &var_,
                 smntDepInfo &sdInfo_,
                 varDepInfoNode &myNode_);

      int startInfo();
      int endInfo();
    };
    //============================================


    //---[ Statement Dependencies ]---------------
    class smntDepInfo {
    public:
      varToDepMap v2dMap;

      statement *s;
      smntDepInfoNode *myNode, *subNode;

      smntDepInfo();

      void setup(statement &s_, smntDepInfoNode &myNode_);

      void setupNestedVdInfos(statement &s_,
                              varInfo &var,
                              varDepInfoNode *vdNode);

      int getDepTypeFrom(expNode &e);

      void updateDependenciesFor(varDepInfo &vdInfo,
                                 expNode &updateExp);

      varDepInfo* has(varInfo &var);
      varDepInfo& operator () (varInfo &var);

      int startInfo(varInfo &var);
      int endInfo(varInfo &var);
    };
    //============================================


    //---[ Dependency Map ]-----------------------
    class depMap_t {
    public:
      parserBase &parser;
      smntToVarDepMap s2vdMap;

      depMap_t(statement &s);

      void setup(statement &s);
      void setup(statement &s, smntDepInfo &sdInfo);

      smntDepInfo* has(statement &s);
      smntDepInfo& operator () (statement &s);

      varDepInfo* has(varInfo &var);
      varDepInfo& operator () (varInfo &var);

      varDepInfo* has(statement &s, varInfo &var);
      varDepInfo& operator () (statement &s, varInfo &var);
    };
    //============================================
  }
}

#endif
