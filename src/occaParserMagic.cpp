#include "occaParserMagic.hpp"

namespace occa {
  namespace parserNS {
    magician::magician(parserBase &parser_) :
      parser(parser_),
      globalScope( *(parser_.globalScope) ) {}

    void magician::castAutomatic(){
      statementNode *sn = globalScope.statementStart;

      while(sn){
        statement &s = *(sn->value);

        if(parser.statementIsAKernel(s))
          castAutomagicOn(s);

        sn = sn->right;
      }
    }

    void magician::castAutomaticOn(statement &kernel){

    }

    void loopCheckStatement(statement *root){
      if((root == NULL) ||
         !(root->type & forStatementType)){

        return;
      }

      root->print();
    }
  };
};