#include "occaParserMagic.hpp"

namespace occa {
  namespace parserNS {
    viInfo_t::viInfo_t() :
      info(0) {}

    viInfoMap_t::viInfoMap_t() :
      anonVar(NULL) {}

    void viInfoMap_t::free(){
      viInfoIterator it = viMap.begin();

      bool freedAnonVar = false;

      while(it != viMap.end()){
        if((it->second) != anonVar){
          delete (it->second);
        }
        else{
          if(!freedAnonVar){
            delete anonVar;
            freedAnonVar = true;
          }
        }

        ++it;
      }
    }

    void viInfoMap_t::addVariable(varInfo &var){
      viInfoIterator it = viMap.find(&var);

      if(it == viMap.end()){
        if(var.hasQualifier("restrict")){
          viMap[&var] = new viInfo_t;
        }
        else{
          if(anonVar != NULL)
            viMap[&var] = anonVar;
          else
            viMap[&var] = new viInfo_t;
        }
      }
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
      viInfoMapStack.top().free();
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

      if(s.info & declareStatementType){
        const int varCount = s.expRoot.getVariableCount();
        viInfoMap_t *viMap = currentViInfoMap();

        for(int i = 0; i < varCount; ++i){
          // Add variable to the varInfo map
          varInfo &var = s.expRoot.getVariableInfoNode(i)->getVarInfo();
          viMap->addVariable(var);

          analyzeDeclareExpression(s.expRoot, i);
        }
      }

      else if(s.info & updateStatementType){
        const int upCount = s.expRoot.getUpdatedVariableCount();

        for(int i = 0; i < upCount; ++i)
            analyzeUpdateExpression(s.expRoot, i);
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
              (snEnd->value->info &  ifStatementType) &&
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

      else if(s.info & (typedefStatementType   |
                        blankStatementType     |
                        blockStatementType     |
                        structStatementType    |
                        functionStatementType  |
                        functionDefinitionType |
                        functionPrototypeType)){
        // Ignore this statement
      }

      else if(s.expRoot.info & expType::goto_){
        printf("[Magic Analyzer] Goto statements are not supported\n");
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

    void magician::analyzeDeclareExpression(expNode &e, const int pos){
      if(e.variableHasInit(pos)){
        addVariableWrite( *(e.getVariableInfoNode(pos)) );
        addExpressionRead( *(e.getVariableInitNode(pos)) );
      }
    }

    void magician::analyzeUpdateExpression(expNode &e, const int pos){
      if(e.updatedVariableIsSet(pos)){
        addVariableWrite( *(e.getUpdatedVariableInfoNode(pos)) );
        addExpressionRead( *(e.getUpdatedVariableSetNode(pos)) );
      }
      else
        addExpressionRead(e);
    }

    bool magician::analyzeForStatement(statement &s){
      if(s.getForStatementCount() < 3){
        printf("[Magic Analyzer] For-loops without 3 statements (4 for okl/ofl loops) are not supported\n");
        return analyzeEmbeddedStatements_f;
      }

      return analyzeEmbeddedStatements_f;
    }

    bool magician::analyzeWhileStatement(statement &s){
      typeHolder th = s.expRoot[0].calculateValue();

      if( !(th.type & noType) &&
          (th.boolValue() == false) ){

        return !analyzeEmbeddedStatements_f;
      }

      return analyzeEmbeddedStatements_f;
    }

    void magician::analyzeIfStatement(statementNode *snStart, statementNode *snEnd){
      statementNode *sn = snStart;

      while(sn != snEnd){
        statement &s  = *(sn->value);
        typeHolder th = s.expRoot[0].calculateValue();

        if( !(th.type & noType) &&
            (th.boolValue() == true) ){

          analyzeEmbeddedStatements(s);

          return;
        }

        sn = sn->right;
      }

      sn = snStart;

      while(sn != snEnd){
        statement &s = *(sn->value);

        analyzeEmbeddedStatements(s);

        sn = sn->right;
      }
    }

    void magician::analyzeSwitchStatement(statement &s){
      typeHolder th = s.expRoot[0].calculateValue();

      if(th.type & noType){
        analyzeEmbeddedStatements(s);
        return;
      }

      statementNode *sn = s.statementStart;
      statementNode *calculateSN;

      while(sn){
        statement &s2 = *(sn->value);

        if(s2.info & caseStatementType){
          if(s2.expRoot.leafCount){ // Not default
            if(th == s2.expRoot[0].calculateValue()){
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
      while(sn){
        statement &s2 = *(sn->value);

        analyzeEmbeddedStatements(s2);

        if(statementGuaranteesBreak(s2))
          break;

        sn = sn->right;
      }
    }

    bool magician::statementGuaranteesBreak(statement &s){
      return false;
    }

    bool magician::variableIsUpdated(expNode &varNode){
      if(!(varNode.info & (expType::varInfo |
                           expType::variable))){

        return false;
      }

      expNode *up = varNode.up;

      if((up != NULL) &&
         (up->info & expType::variable)){

        up = up->up;
      }

      if(up == NULL)
        return false;

      return ((up->info & expType::operator_) &&
              isAnUpdateOperator(up->value));
    }

    void magician::addVariableWrite(expNode &varNode){
      const bool isUpdated = variableIsUpdated(varNode);

      if(varNode.info & expType::variable){
        const int brackets = varNode.getVariableBracketCount();

        if(brackets)
          addVariableWrite(varNode, brackets, varNode.getVariableBracket(0)->up);

        return;
      }

      if(isUpdated)
        addVariableRead(varNode);

      varNode.print();
    }

    void magician::addVariableWrite(expNode &varNode,
                                    const int brackets,
                                    expNode *bracketNode){
      const bool isUpdated = variableIsUpdated(varNode);

      if(isUpdated)
        addVariableRead(varNode, brackets, bracketNode);

      varNode.print();
    }

    void magician::addVariableRead(expNode &varNode){
      if(varNode.info & expType::variable){
        const int brackets = varNode.getVariableBracketCount();

        if(brackets)
          addVariableRead(varNode, brackets, varNode.getVariableBracket(0)->up);

        return;
      }

      varNode.print();
    }

    void magician::addVariableRead(expNode &varNode,
                                   const int brackets,
                                   expNode *bracketNode){
      varNode.print();
    }

    void magician::addExpressionRead(expNode &e){
      if(e.info & expType::variable){
        const int brackets = e.getVariableBracketCount();

        if(brackets)
          addVariableRead(e, brackets, e.getVariableBracket(0)->up);
      }
      else if((e.info & expType::varInfo) &&
              ((e.up == NULL) ||
               !(e.up->info & expType::variable))){

        addVariableRead(e);
      }
      else {
        for(int i = 0; i < e.leafCount; ++i)
          addExpressionRead(e[i]);
      }
    }
  };
};
