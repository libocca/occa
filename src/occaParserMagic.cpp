#include "occaParserMagic.hpp"

namespace occa {
  namespace parserNS {
    atomInfo_t::atomInfo_t() :
      info(viType::isUseless),
      var(NULL) {}

    void atomInfo_t::load(expNode &e){
      info = viType::isUseless;

      if(e.info & expType::varInfo){
        info = viType::isAVariable;
        var  = &(e.getVarInfo());
      }
      else{
        constValue = e.calculateValue();

        if(constValue.type & noType)
          exp = e;
        else
          info = viType::isConstant;
      }
    }

    void atomInfo_t::load(const std::string &s){
      info       = viType::isConstant;
      constValue = s;
    }

    valueInfo_t::valueInfo_t() :
      indices(0),
      vars(NULL),
      strides(NULL) {}

    bool valueInfo_t::isUseless(){
      for(int i = 0; i < indices; ++i){
        if(vars[i].info & viType::isUseless)
          return true;
      }

      return false;
    }

    void valueInfo_t::load(expNode &e){
      expNode *cNode = &e;
      indices = 1;

      // Needs to warn about negative indices
      while((cNode        != NULL)        &&
            (cNode->info  == expType::LR) &&
            (cNode->value == "+")){

        ++indices;
        cNode = cNode->leaves[0];
      }

      allocVS(indices);

      if(indices == 1){
        loadVS(e, 0);
      }
      else {
        cNode = &e;

        for(int i = 0; i < (indices - 1); ++i){
          const int invI = (indices - i - 1);
          loadVS((*cNode)[1], invI);

          if(invI == 1)
            loadVS((*cNode)[0], 0);

          cNode = cNode->leaves[0];
        }
      }
    }

    void valueInfo_t::loadVS(expNode &e, const int pos){
      if((e.info  == expType::LR) &&
         (e.value == "*")){

        const bool varIn0 = (e[0].info & expType::varInfo);
        const bool varIn1 = (e[1].info & expType::varInfo);

        if(varIn0 || varIn1){
          vars[pos].load(e[varIn1]);
          strides[pos].load(e[!varIn1]);
          return;
        }
      }
      else if(e.info == expType::varInfo){
        vars[pos].load(e);
        strides[pos].load("1");
        return;
      }
      else if(e.info == expType::presetValue){
        vars[pos].load(e.value);
        strides[pos].load("1");
        return;
      }

      std::cout << "useless e = " << e << '\n';

      vars[pos].info    = viType::isUseless;
      strides[pos].info = viType::isUseless;
    }

    void valueInfo_t::allocVS(const int count){
      vars    = new atomInfo_t[count];
      strides = new atomInfo_t[count];
    }

    varInfo& valueInfo_t::var(const int pos){
      return *(vars[pos].var);
    }

    atomInfo_t& valueInfo_t::stride(const int pos){
      return strides[pos];
    }

    accessInfo_t::accessInfo_t() :
      dim(0),
      dimIndices(NULL) {}

    void accessInfo_t::load(const int brackets, expNode &bracketNode){
      dim = brackets;
      dimIndices = new valueInfo_t[dim];

      for(int i = 0; i < dim; ++i)
        dimIndices[i].load(bracketNode[i][0]);
    }

    iteratorInfo_t::iteratorInfo_t(){}

    viInfo_t::viInfo_t() :
      info(viType::isUseless) {}

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

      viMap.clear();
      anonVar = NULL;
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

    viInfoMap_t* viInfoDB_t::map(){
      return &(viInfoMapStack.top());
    }

    void viInfoDB_t::enteringStatement(){
      viInfoMapStack.push( viInfoMap_t() );
    }

    void viInfoDB_t::leavingStatement(){
      viInfoMapStack.top().free();
      viInfoMapStack.pop();
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
          analyzeFunction(s);

        sn = sn->right;
      }
    }

    void magician::analyzeFunction(statement &fs){
      varInfo &func = *(fs.getFunctionVar());

      viInfoDB.enteringStatement();

      viInfoMap_t *viMap = viInfoDB.map();

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

      viInfoDB.leavingStatement();
    }

    void magician::analyzeStatement(statement &s){
      int smntInfo = analyzeInfo::analyzeEmbedded;

      if(s.info & declareStatementType){
        const int varCount = s.expRoot.getVariableCount();
        viInfoMap_t *viMap = viInfoDB.map();

        for(int i = 0; i < varCount; ++i){
          // Add variable to the varInfo map
          varInfo &var = s.expRoot.getVariableInfoNode(i)->getVarInfo();
          viMap->addVariable(var);

          analyzeDeclareExpression(smntInfo, s.expRoot, i);
        }
      }

      else if(s.info & updateStatementType){
        const int upCount = s.expRoot.getUpdatedVariableCount();

        for(int i = 0; i < upCount; ++i)
          analyzeUpdateExpression(smntInfo, s.expRoot, i);
      }

      else if(s.info & forStatementType){
        analyzeForStatement(smntInfo, s);
      }

      else if(s.info & whileStatementType){
        analyzeWhileStatement(smntInfo, s);
      }

      else if(s.info & doWhileStatementType){
        // do-while guarantees at least one run
        analyzeEmbeddedStatements(s);
        analyzeWhileStatement(smntInfo, s);

        smntInfo &= ~analyzeInfo::analyzeEmbedded;
      }

      else if(s.info & ifStatementType){
        statementNode *snStart = s.getStatementNode();
        statementNode *snEnd   = snStart->right;

        while(snEnd                                   &&
              (snEnd->value->info &  ifStatementType) &&
              (snEnd->value->info != ifStatementType)){

          snEnd = snEnd->right;
        }

        analyzeIfStatement(smntInfo, snStart, snEnd);
      }

      else if(s.info & switchStatementType){
        analyzeSwitchStatement(smntInfo, s);
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

      if(smntInfo & analyzeInfo::analyzeEmbedded)
        analyzeEmbeddedStatements(s);
    }

    void magician::analyzeEmbeddedStatements(statement &s){
      if(s.statementStart != NULL){
        viInfoDB.enteringStatement();

        statementNode *statementPos = s.statementStart;

        while(statementPos){
          analyzeStatement( *(statementPos->value) );

          statementPos = statementPos->right;
        }

        viInfoDB.leavingStatement();
      }
    }

    void magician::analyzeDeclareExpression(int &smntInfo, expNode &e, const int pos){
      if(e.variableHasInit(pos)){
        expNode &varNode  = *(e.getVariableInfoNode(pos));
        expNode &initNode = *(e.getVariableInitNode(pos));

        addVariableWrite(varNode, initNode);
        addExpressionRead(initNode);
      }
    }

    void magician::analyzeUpdateExpression(int &smntInfo, expNode &e, const int pos){
      if(e.updatedVariableIsSet(pos)){
        expNode &varNode = *(e.getUpdatedVariableInfoNode(pos));
        expNode &setNode = *(e.getUpdatedVariableSetNode(pos));

        addVariableWrite(varNode, setNode);
        addExpressionRead(setNode);
      }
      else
        addExpressionRead(e);
    }

    void magician::analyzeForStatement(int &smntInfo, statement &s){
      if(s.getForStatementCount() < 3){
        printf("[Magic Analyzer] For-loops without 3 statements (4 for okl/ofl loops) are not supported\n");
        smntInfo &= ~analyzeInfo::analyzeEmbedded;
        return;
      }

      expNode &updateNode = s.expRoot[2];

      if(updateNode.leafCount == 0){
        printf("[Magic Analyzer] For-loop update statement (3rd statement) is not standard, for example:\n  X op Y where op can be [+=] or [-=]\n  ++X, X++, --X, X--\n");
        smntInfo &= ~analyzeInfo::analyzeEmbedded;
        return;
      }

      // updateNode.print();
    }

    void magician::analyzeWhileStatement(int &smntInfo, statement &s){
      typeHolder th = s.expRoot[0].calculateValue();

      if( !(th.type & noType) &&
          (th.boolValue() == false) ){

        smntInfo &= ~analyzeInfo::analyzeEmbedded;
        return;
      }
    }

    void magician::analyzeIfStatement(int &smntInfo, statementNode *snStart, statementNode *snEnd){
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

    void magician::analyzeSwitchStatement(int &smntInfo, statement &s){
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

    void magician::addVariableWrite(expNode &varNode, expNode &setNode){
      const bool isUpdated = variableIsUpdated(varNode);

      if(varNode.info & expType::variable){
        const int brackets = varNode.getVariableBracketCount();

        if(brackets)
          addVariableWrite(varNode, setNode, brackets, *(varNode.getVariableBracket(0)->up));

        return;
      }

      if(isUpdated)
        addVariableRead(varNode);
    }

    void magician::addVariableWrite(expNode &varNode,
                                    expNode &setNode,
                                    const int brackets,
                                    expNode &bracketNode){
      const bool isUpdated = variableIsUpdated(varNode);

      if(isUpdated)
        addVariableRead(varNode, brackets, bracketNode);
    }

    void magician::addVariableRead(expNode &varNode){
      if(varNode.info & expType::variable){
        const int brackets = varNode.getVariableBracketCount();

        if(brackets)
          addVariableRead(varNode, brackets, *(varNode.getVariableBracket(0)->up));

        return;
      }
    }

    void magician::addVariableRead(expNode &varNode,
                                   const int brackets,
                                   expNode &bracketNode){
      accessInfo_t access;
      access.load(brackets, bracketNode);
    }

    void magician::addExpressionRead(expNode &e){
      if(e.info & expType::variable){
        const int brackets = e.getVariableBracketCount();

        if(brackets)
          addVariableRead(e, brackets, *(e.getVariableBracket(0)->up));
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
