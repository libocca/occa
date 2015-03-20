#include "occaParserMagic.hpp"

namespace occa {
  namespace parserNS {
    viInfoMap_t::viInfoMap_t(){}

    void viInfoMap_t::addVariable(varInfo &var){
      viMap[&var];
    }

    magician::magician(parserBase &parser_) :
      parser(parser_),
      globalScope( *(parser_.globalScope) ),
      varUpdateMap(parser_.varUpdateMap),
      varUsedMap(parser_.varUsedMap) {}

    viInfoMap_t* magician::currentViInfoMap(){
      return &(viInfoMapStack.top());
    }

    void magician::pushMapStack(){
      viInfoMapStack.push( viInfoMap_t() );
    }

    void magician::popMapStack(){
      viInfoMapStack.pop();
    }

    void magician::castMagicOn(parserBase &parser_){
      magician mickey(parser_);
      mickey.castMagic();
    }

    void magician::castMagic(){
      statementNode *sn = globalScope.statementStart;

      while(sn){
        statement &s = *(sn->value);

        if(parser.statementIsAKernel(s))
          analyzeFunction(s);

        sn = sn->right;
      }
    }

    void magician::analyzeFunction(statement &fs){
      varInfo &func = *(fs.getFunctionVar());

      pushMapStack();

      viInfoMap_t *viMap = currentViInfoMap();

      // Place function arguments (if any)
      if(func.argumentCount){
        for(int arg = 0; arg < func.argumentCount; ++arg){
          varInfo &varg = *(func.argumentVarInfos[arg]);

          viMap->addVariable(varg);
        }
      }

      statementNode *statementPos = fs.statementStart;

      while(statementPos){
        analyzeStatement( *(statementPos->value) );

        statementPos = statementPos->right;
      }

      popMapStack();
    }

    void magician::analyzeStatement(statement &s){
      bool analyzeEmbedded = true;

      if(s.expRoot.info & expType::goto_){
        printf("[Magic Analyzer] Goto statements are not allowed\n");
      }

      else if(s.info & (typedefStatementType   |
                        blankStatementType     |
                        blockStatementType     |
                        structStatementType    |
                        functionStatementType  |
                        functionDefinitionType |
                        functionPrototypeType)){
        // Ignore this statement
      }

      else if(s.info & (declareStatementType |
                        updateStatementType)){

        const int varCount = s.expRoot.getVariableCount();
        viInfoMap_t *viMap = currentViInfoMap();

        for(int i = 0; i < varCount; ++i){
          expNode &varNode = *(s.expRoot.getVariableNode(i));

          // Add variable to the varInfo map
          if(s.info & declareStatementType){
            varInfo &var = s.expRoot.getVariableInfoNode(i)->getVarInfo();
            viMap->addVariable(var);
          }

          analyzeUpdateExpression(varNode);
        }
      }

      else if(s.info & forStatementType){
        analyzeEmbedded = analyzeForStatement(s);
      }
      else if(s.info & whileStatementType){
        analyzeEmbedded = analyzeWhileStatement(s);
      }
      else if(s.info & doWhileStatementType){
        analyzeEmbedded = false;

        // do-while guarantees at least one run
        analyzeEmbeddedStatements(s);
        analyzeWhileStatement(s);
      }
      else if(s.info & ifStatementType){
        statementNode *snStart = s.getStatementNode();
        statementNode *snEnd   = snStart->right;

        while(snEnd                                   &&
              (snEnd->value->info & ifStatementType)  &&
              (snEnd->value->info != ifStatementType)){

          snEnd = snEnd->right;
        }

        analyzeEmbedded = false;
        analyzeIfStatement(snStart, snEnd);
      }
      else if(s.info & switchStatementType){
        analyzeEmbedded = false;
        analyzeSwitchStatement(s);
      }

      if(analyzeEmbedded)
        analyzeEmbeddedStatements(s);
    }

    void magician::analyzeEmbeddedStatements(statement &s){
      if(s.statementStart != NULL){
        pushMapStack();

        statementNode *statementPos = s.statementStart;

        while(statementPos){
          analyzeStatement( *(statementPos->value) );

          statementPos = statementPos->right;
        }

        popMapStack();
      }
    }

    void magician::analyzeUpdateExpression(expNode &expRoot){
    }

    bool magician::analyzeForStatement(statement &s){
      return true;
    }

    bool magician::analyzeWhileStatement(statement &s){
      return true;
    }

    void magician::analyzeIfStatement(statementNode *snStart, statementNode *snEnd){
    }

    void magician::analyzeSwitchStatement(statement &s){
    }
  };
};