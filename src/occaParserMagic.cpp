#include "occaParserMagic.hpp"

namespace occa {
  namespace parserNS {
    strideInfo::strideInfo() :
      isConstant(false),
      varName(NULL) {}

    accessInfo::accessInfo() :
      isUseful(false) {}

    void accessInfo::load(expNode &root){
      root.print();
    }

    int accessInfo::dim(){
      return ((int) strides.size());
    }

    varInfo& accessInfo::var(const int pos){
      return *(strides[pos].varName);
    }

    strideInfo& accessInfo::operator [] (const int pos){
      return strides[pos];
    }

    strideInfo& accessInfo::operator [] (const std::string &varName){
      const int size_ = ((int) strides.size());

      for(int i = 0; i < size_; ++i){
        if(strides[i].varName->name == varName)
          return strides[i];
      }

      return *((strideInfo*) NULL);
    }

    magician::magician(parserBase &parser_) :
      parser(parser_),
      globalScope( *(parser_.globalScope) ),
      varUpdateMap(parser_.varUpdateMap),
      varUsedMap(parser_.varUsedMap) {}

    void magician::castMagicOn(parserBase &parser_){
      magician mickey(parser_);
      mickey.castMagic();
    }

    void magician::castMagic(){
      statementNode *sn = globalScope.statementStart;

      while(sn){
        statement &s = *(sn->value);

        if(parser.statementIsAKernel(s))
          castMagicOn(s);

        sn = sn->right;
      }
    }

    void magician::castMagicOn(statement &kernel){

    }

    void loopCheckStatement(statement *root){
      if((root == NULL) ||
         !(root->info & forStatementType)){

        return;
      }

      root->expRoot.print();
    }
  };
};