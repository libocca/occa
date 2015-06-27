#ifndef OCCA_PARSER_ANALYZER_HEADER
#define OCCA_PARSER_ANALYZER_HEADER

#include "occaParserStatement.hpp"

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
    }

    //---[ Variable Dependencies ]----------------
    class varDepInfo {
    public:
      int info;

      varInfo *var;
      varDepInfoNode *myNode;
      varDepInfoNode *subNode;

      varDepInfo();

      void setup(int info_,
                 varInfo &var_,
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
      smntDepInfoNode *myNode;

      smntDepInfo();

      void setup(statement &s_, smntDepInfoNode &myNode_);

      void setupNestedVdInfos(statement &s_,
                              varInfo &var,
                              varDepInfoNode *vdNode);

      int getDepTypeFrom(expNode &e);

      varDepInfo* has(varInfo &var);
      varDepInfo& operator () (varInfo &var);

      int startInfo(varInfo &var);
      int endInfo(varInfo &var);
    };
    //============================================


    //---[ Dependency Map ]-----------------------
    class depMap_t {
    public:
      smntToVarDepMap s2vdMap;

      void setup(statement &s);
      void setup(statement &s, smntDepInfo &sdInfo);

      varDepInfo* has(statement &s, varInfo &var);
      varDepInfo& operator () (statement &s, varInfo &var);
    };
    //============================================
  };
};

#endif
