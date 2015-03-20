#include "occaParserMagic.hpp"

namespace occa {
  namespace parserNS {
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