#include "occaParserAnalyzer.hpp"

namespace occa {
  namespace parserNS {
    //---[ Variable Dependencies ]----------------
    void varDepInfo::setup(int info_,
                           varInfo &var_,
                           varDepInfoNode &myNode_){
      info   = info_;
      var    = &var_;
      myNode = &myNode_;
    }

    int varDepInfo::startInfo(){
      varDepInfoNode *startNode = myNode->down;

      if(startNode == NULL)
        return info;

      return (startNode ? startNode->value->startInfo() : depType::none);
    }

    int varDepInfo::endInfo(){
      varDepInfoNode *endNode = myNode->down;

      if(endNode == NULL)
        return info;

      return (endNode ? endNode->value->endInfo() : depType::none);
    }
    //============================================


    //---[ Statement Dependencies ]---------------
    void smntDepInfo::setup(statement &s_, smntDepInfoNode &myNode_){
      s      = &s_;
      myNode = &myNode_;

      expNode &flatRoot = *(s_.expRoot.makeFlatHandle());

      for(int i = 0; i < flatRoot.leafCount; ++i){
        expNode &leaf = flatRoot[i];

        if((leaf.info & expType::varInfo) == 0)
          continue;

        int leafDepInfo = getDepTypeFrom(leaf);

        if(leafDepInfo == depType::none)
          continue;

        varInfo &var = leaf.getVarInfo();

        varDepInfoNode *&vdNode = v2dMap[&var];

        varDepInfo &newVdInfo = *(new varDepInfo);
        newVdInfo.setup(leafDepInfo, var, *(new varDepInfoNode));

        if(vdNode == NULL){
          vdNode = newVdInfo.myNode;
        }
        else {
          if(vdNode->down == NULL){
            vdNode->pushDown(vdNode->value);
            vdNode->value = NULL;
          }

          varDepInfoNode &downNode = *(vdNode->down);
          varDepInfoNode &endNode  = *(lastNode(&downNode));

          endNode.push(&newVdInfo);
        }
      }

      expNode::freeFlatHandle(flatRoot);
    }

    int smntDepInfo::getDepTypeFrom(expNode &e){
      if((e.up == NULL) ||
         ((e.up->info & expType::operator_) == 0)){

        return depType::none;
      }

      expNode &eUp = *(e.up);

      if(eUp.value == "=")
        return depType::set;

      if(isAnUpdateOperator(eUp.value))
        return depType::update;

      return depType::none;
    }

    varDepInfo* smntDepInfo::has(varInfo &var){
      varToDepMapIterator it = v2dMap.find(&var);

      if(it == v2dMap.end())
        return NULL;

      return (it->second->value);
    }

    varDepInfo& smntDepInfo::operator () (varInfo &var){
      return *(has(var));
    }

    int smntDepInfo::startInfo(varInfo &var){
      varDepInfo *vdInfo = has(var);

      if(vdInfo == NULL)
        return depType::none;

      return vdInfo->startInfo();
    }

    int smntDepInfo::endInfo(varInfo &var){
      varDepInfo *vdInfo = has(var);

      if(vdInfo == NULL)
        return depType::none;

      return vdInfo->endInfo();
    }
    //============================================


    //---[ Dependency Map ]-----------------------
    void depMap_t::setup(statement &s){
      setup(s, *(new smntDepInfo));
    }

    void depMap_t::setup(statement &s, smntDepInfo &sdInfo){
      smntDepInfoNode &sdNode = *(new smntDepInfoNode);
      sdNode.value = &sdInfo;

      sdInfo.setup(s, sdNode);

      s2vdMap[&s] = &sdInfo;

      statementNode *sn = s.statementStart;

      while(sn){
        statement &s2 = *(sn->value);

        setup(s2, *(new smntDepInfo));

        sn = sn->right;
      }
    }

    varDepInfo* depMap_t::has(statement &s, varInfo &var){
      smntToVarDepMapIterator it = s2vdMap.find(&s);

      if(it == s2vdMap.end())
        return NULL;

      return (it->second->has(var));
    }

    varDepInfo& depMap_t::operator () (statement &s, varInfo &var){
      return *(has(s,var));
    }
    //============================================
  };
};
