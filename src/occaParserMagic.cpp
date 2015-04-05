#include "occaParserMagic.hpp"

namespace occa {
  namespace parserNS {
    namespace viType {
      std::string infoToStr(const int info){
        std::string tag;

        if(info & viType::isUseless)    tag += 'U';
        if(info & viType::isAVariable)  tag += 'V';
        if(info & viType::isAnIterator) tag += 'I';
        if(info & viType::isConstant)   tag += 'C';

        return tag;
      }
    };

    atomInfo_t::atomInfo_t(infoDB_t *db_) :
      db(db_),
      info(viType::isUseless),
      var(NULL) {}

    atomInfo_t::atomInfo_t(const atomInfo_t &ai) :
      db(ai.db),
      info(ai.info),
      constValue(ai.constValue),
      // exp(ai.exp), [<>] Missing, needs to be added after refactor
      var(ai.var) {}

    void atomInfo_t::setDB(infoDB_t *db_){
      db = db_;
    }

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

    void atomInfo_t::load(varInfo &var_){
      info = viType::isAVariable;
      var  = &var_;
    }

    void atomInfo_t::load(const std::string &s){
      info       = viType::isConstant;
      constValue = s;
    }

    std::ostream& operator << (std::ostream &out, atomInfo_t &info){
      out << viType::infoToStr(info.info) << ": ";

      if(info.info & viType::isConstant)
        out << info.constValue;
      else if(info.info & viType::isAVariable)
        out << info.var->name;
      else
        out << info.exp;

      return out;
    }

    valueInfo_t::valueInfo_t(infoDB_t *db_) :
      db(db_),
      info(0),
      indices(0),
      value(db_),
      vars(NULL),
      strides(NULL) {}

    valueInfo_t::valueInfo_t(const valueInfo_t &vi, infoDB_t *db_) :
      db(db_),
      info(vi.info),
      indices(vi.indices),
      value(db_),
      vars(vi.vars),
      strides(vi.strides) {}

    valueInfo_t::valueInfo_t(expNode &e, infoDB_t *db_) :
      db(db_),
      info(0),
      indices(0),
      value(db_),
      vars(NULL),
      strides(NULL) {

      load(e);
    }

    void valueInfo_t::setDB(infoDB_t *db_){
      db = db_;

      value.setDB(db);

      for(int i = 0; i < indices; ++i){
        vars[i].setDB(db);
        strides[i].setDB(db);
      }
    }

    valueInfo_t& valueInfo_t::operator = (const valueInfo_t &vi){
      info    = vi.info;
      indices = vi.indices;
      vars    = vi.vars;
      strides = vi.strides;

      return *this;
    }

    void valueInfo_t::allocVS(const int count){
      if(count <= 0)
        return;

      vars    = new atomInfo_t[count];
      strides = new atomInfo_t[count];

      for(int i = 0; i < count; ++i){
        vars[i].setDB(db);
        strides[i].setDB(db);
      }
    }

    bool valueInfo_t::isUseless(){
      if(info & viType::isConstant)
        return false;

      for(int i = 0; i < indices; ++i){
        if(vars[i].info & viType::isUseless)
          return true;
      }

      return false;
    }

    void valueInfo_t::load(expNode &e){
      expNode *cNode = &e;
      indices = 1;

      // [-] Needs to warn about negative indices
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

      sortIndices();
    }

    void valueInfo_t::load(varInfo &var){
      value.load(var);
    }

    void valueInfo_t::load(const std::string &s){
      value.load(s);
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
      else if(e.info & expType::varInfo){
        vars[pos].load(e);
        strides[pos].load("1");
        return;
      }
      else if(e.info == expType::presetValue){
        vars[pos].load(e.value);
        strides[pos].load("1");
        return;
      }

      vars[pos].info    = viType::isUseless;
      strides[pos].info = viType::isUseless;

      vars[pos].exp = e;
    }

    void valueInfo_t::sortIndices(){
      if(indices <= 1)
        return;

      std::cout << "SI 1: " << *this << '\n';

      int *vi = new int[2*indices];

      for(int i = 0; i < indices; ++i){
        vi[2*i + 0] = 0; // Value
        vi[2*i + 1] = i; // Index
      }

      qsort(vi, indices, 2*sizeof(int), valueInfo_t::qSortIndices);



      std::cout << "SI 2: " << *this << '\n';
    }

    int valueInfo_t::qSortIndices(const void *a, const void *b){
      return ((*((int*) a)) - (*((int*) b)));
    }

    void valueInfo_t::merge(expNode &op, expNode &e){
      valueInfo_t evi(e, db);

      if(op.value == "="){
        *this = evi;
      }
      else {
        // [-] Missing merge
      }
    }

    typeHolder valueInfo_t::constValue(){
      return value.constValue;
    }

    varInfo& valueInfo_t::varValue(){
      return *(value.var);
    }

    varInfo& valueInfo_t::var(const int pos){
      return *(vars[pos].var);
    }

    atomInfo_t& valueInfo_t::stride(const int pos){
      return strides[pos];
    }

    std::ostream& operator << (std::ostream &out, valueInfo_t &info){
      if(info.indices == 0){
        out << info.value;
      }
      else{
        for(int i = 0; i < info.indices; ++i){
          if(i != 0)
            out << " + ";

          out << info.vars[i];

          if( !(info.vars[i].info & viType::isUseless) )
            out << " (" << info.strides[i] << ')';
        }
      }

      return out;
    }

    accessInfo_t::accessInfo_t(infoDB_t *db_) :
      db(db_),
      s(NULL),
      dim(0),
      value(db_),
      dimIndices(NULL) {}

    void accessInfo_t::setDB(infoDB_t *db_){
      db = db_;

      value.setDB(db_);

      for(int i = 0; i < dim; ++i)
        dimIndices[i].setDB(db_);
    }

    void accessInfo_t::load(expNode &varNode){
      s = varNode.sInfo;

      dim = 0;
      value.load(varNode);
    }

    void accessInfo_t::load(const int brackets, expNode &bracketNode){
      s = bracketNode.sInfo;

      dim        = brackets;
      dimIndices = new valueInfo_t[dim];

      for(int i = 0; i < dim; ++i){
        dimIndices[i].setDB(db);
        dimIndices[i].load(bracketNode[i][0]);
      }
    }

    bool accessInfo_t::conflictsWith(accessInfo_t &ai){
      return true;
    }

    std::ostream& operator << (std::ostream &out, accessInfo_t &info){
      if(info.dim == 0){
        out << '[' << info.value << ']';
      }
      else{
        for(int i = 0; i < info.dim; ++i)
          out << '[' << info.dimIndices[i] << ']';
      }

      return out;
    }

    iteratorInfo_t::iteratorInfo_t(infoDB_t *db_) :
      db(db_),
      start(db_),
      end(db_),
      stride(db_) {}

    void iteratorInfo_t::setDB(infoDB_t *db_){
      db = db_;

      start.setDB(db);
      end.setDB(db);
      stride.setDB(db);
    }

    std::ostream& operator << (std::ostream &out, iteratorInfo_t &info){
      out << "[Bounds: ["
          << info.start  << ", "
          << info.end    << "], Stride: "
          << info.stride << ']';

      return out;
    }

    viInfo_t::viInfo_t(infoDB_t *db_) :
      db(db_),
      info(viType::isUseless),
      valueInfo(db_),
      dimInfo(db_),
      iteratorInfo(db_) {}

    void viInfo_t::setDB(infoDB_t *db_){
      db = db_;

      valueInfo.setDB(db);
      dimInfo.setDB(db);
      iteratorInfo.setDB(db);
    }

    accessInfo_t& viInfo_t::addWrite(expNode &varNode){
      writes.push_back( accessInfo_t(db) );

      accessInfo_t &ai = writes.back();
      ai.load(varNode);
      std::cout << "W1. ai = " << ai << '\n';

      checkLastInput(ai, writeValue);

      return ai;
    }

    accessInfo_t& viInfo_t::addWrite(const int brackets, expNode &bracketNode){
      writes.push_back( accessInfo_t(db) );

      accessInfo_t &ai = writes.back();
      ai.load(brackets, bracketNode);
      std::cout << "W2. ai = " << ai << '\n';

      checkLastInput(ai, writeValue);

      return ai;
    }

    accessInfo_t& viInfo_t::addRead(expNode &varNode){
      reads.push_back( accessInfo_t(db) );

      accessInfo_t &ai = reads.back();
      ai.load(varNode);
      std::cout << "R1. ai = " << ai << '\n';

      checkLastInput(ai, readValue);

      return ai;
    }

    accessInfo_t& viInfo_t::addRead(const int brackets, expNode &bracketNode){
      reads.push_back( accessInfo_t(db) );

      accessInfo_t &ai = reads.back();
      ai.load(brackets, bracketNode);
      std::cout << "R2. ai = " << ai << '\n';

      checkLastInput(ai, readValue);

      return ai;
    }

    void viInfo_t::checkLastInput(accessInfo_t &ai, const int inputType){
      std::vector<accessInfo_t> &inputs = ((inputType & readValue) ? writes : reads);

      const int inputCount = (int) inputs.size();

      for(int i = 0; i < inputCount; ++i){
        if(inputs[i].conflictsWith(ai)){

          break;
        }
      }
    }

    std::ostream& operator << (std::ostream &out, viInfo_t &info){
      return out;
    }

    viInfoMap_t::viInfoMap_t(infoDB_t *db_) :
      db(db_),
      anonVar(NULL) {}

    void viInfoMap_t::setDB(infoDB_t *db_){
      db = db_;
    }

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

    void viInfoMap_t::add(varInfo &var){
      viInfoIterator it = viMap.find(&var);

      if(it == viMap.end()){
        if(var.hasQualifier("restrict")){
          viMap[&var] = new viInfo_t(db);
        }
        else{
          if(anonVar != NULL)
            viMap[&var] = anonVar;
          else
            viMap[&var] = new viInfo_t(db);
        }
      }
    }

    viInfo_t* viInfoMap_t::has(varInfo &var){
      viInfoIterator it = viMap.find(&var);

      return ((it == viMap.end()) ?
              NULL : it->second);
    }

    viInfo_t& viInfoMap_t::operator [] (varInfo &var){
      return *(viMap[&var]);
    }

    infoDB_t::infoDB_t() :
      viInfoMap(this) {

      smntInfoStack.push(analyzeInfo::isExecuted);
    }

    int& infoDB_t::getSmntInfo(){
      return smntInfoStack.top();
    }

    void infoDB_t::add(varInfo &var){
      viInfoMap.add(var);
    }

    viInfo_t* infoDB_t::has(varInfo &var){
      return viInfoMap.has(var);
    }

    void infoDB_t::enteringStatement(statement &s){
      smntInfoStack.push(getSmntInfo());
    }

    void infoDB_t::leavingStatement(){
      smntInfoStack.pop();
    }

    viInfo_t& infoDB_t::operator [] (varInfo &var){
      return *(viInfoMap.has(var));
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

      db.enteringStatement(fs);

      // Place function arguments (if any)
      if(func.argumentCount){
        for(int arg = 0; arg < func.argumentCount; ++arg){
          varInfo &varg = *(func.argumentVarInfos[arg]);

          db.add(varg);
        }
      }

      statementNode *statementPos = fs.statementStart;

      while(statementPos){
        analyzeStatement( *(statementPos->value) );

        statementPos = statementPos->right;
      }

      db.leavingStatement();
    }

    void magician::analyzeStatement(statement &s){
      db.enteringStatement(s);

      if(s.info & declareStatementType){
        analyzeDeclareStatement(s.expRoot);
      }

      else if(s.info & updateStatementType){
        analyzeUpdateStatement(s.expRoot);
      }

      else if(s.info & forStatementType){
        if(parser.parsingC)
          analyzeForStatement(s);
        else
          analyzeFortranForStatement(s);
      }

      else if(s.info & whileStatementType){
        analyzeWhileStatement(s);
      }

      else if(s.info & doWhileStatementType){
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

        analyzeIfStatement(snStart, snEnd);
      }

      else if(s.info & switchStatementType){
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

      db.smntInfoMap[&s] = db.getSmntInfo();

      if(db.getSmntInfo() & analyzeInfo::isExecuted)
        analyzeEmbeddedStatements(s);

      db.leavingStatement();
    }

    void magician::analyzeEmbeddedStatements(statement &s){
      if(s.statementStart != NULL){
        statementNode *statementPos = s.statementStart;

        while(statementPos){
          analyzeStatement( *(statementPos->value) );

          statementPos = statementPos->right;
        }
      }
    }

    void magician::analyzeDeclareStatement(expNode &e){
      const int varCount = e.getVariableCount();

      for(int i = 0; i < varCount; ++i){
        // Add variable to the varInfo map
        varInfo &var = e.getVariableInfoNode(i)->getVarInfo();
        db.add(var);

        analyzeDeclareExpression(e, i);
      }
    }

    void magician::analyzeDeclareExpression(expNode &e, const int pos){
      if(e.variableHasInit(pos)){
        expNode &varNode  = *(e.getVariableInfoNode(pos));
        expNode &initNode = *(e.getVariableInitNode(pos));

        addVariableWrite(varNode, initNode);
        addExpressionRead(initNode);

        viInfo_t &viInfo = db[ varNode.getVarInfo() ];
        viInfo.valueInfo.load(initNode);
      }
    }

    void magician::analyzeUpdateStatement(expNode &e){
      const int upCount = e.getUpdatedVariableCount();

      for(int i = 0; i < upCount; ++i)
        analyzeUpdateExpression(e, i);
    }

    void magician::analyzeUpdateExpression(expNode &e, const int pos){
      if(e.updatedVariableIsSet(pos)){
        expNode &varNode = *(e.getUpdatedVariableInfoNode(pos));
        expNode &setNode = *(e.getUpdatedVariableSetNode(pos));

        addVariableWrite(varNode, setNode);
        addExpressionRead(setNode);

        viInfo_t &viInfo = db[ varNode.getVarInfo() ];
        viInfo.valueInfo.merge(e, setNode);
      }
      else
        addExpressionRead(e);
    }

    void magician::analyzeForStatement(statement &s){
      if(s.getForStatementCount() < 3){
        printf("[Magic Analyzer] For-loops without 3 statements (4 for okl/ofl loops) are not supported\n");
        db.getSmntInfo() = analyzeInfo::schrodinger;
        return;
      }

      // [1] Add first node
      if(s.expRoot[0].info == expType::declaration)
        analyzeDeclareStatement(s.expRoot[0]);
      else
        analyzeUpdateStatement(s.expRoot[0]);

      // [3] Check update node
      expNode &updateNode = s.expRoot[2];

      bool wrongFormat = false;

      for(int i = 0; i < updateNode.leafCount; ++i){
        expNode &leaf = updateNode[i];

        if(!(leaf.info & expType::LR)){
          wrongFormat = true;
          break;
        }
        else if(leaf.info == expType::LR){
          if((leaf.value != "+=") && (leaf.value != "-=")){
            wrongFormat = true;
            break;
          }
        }
        else{ // (leaf.info & expType::LR)
          if((leaf.value != "++") && (leaf.value != "--")){
            wrongFormat = true;
            break;
          }
        }
      }

      if(wrongFormat){
        printf("[Magic Analyzer] For-loop update statement (3rd statement) is not standard, for example:\n  X op Y where op can be [+=] or [-=]\n  ++X, X++, --X, X--\n");
        return;
      }

      varInfo *var    = NULL;
      expNode *stride = NULL;
      std::string str;

      for(int i = 0; i < updateNode.leafCount; ++i){
        expNode &leaf = updateNode[i];

        if(leaf.info == expType::LR){
          if((leaf.value == "+=") ||
             (leaf.value == "-=")){

            const bool varIn0 = (leaf[0].info & expType::varInfo);
            const bool varIn1 = (leaf[1].info & expType::varInfo);

            if(varIn0 ^ varIn1){
              var = (varIn0 ?
                     &(leaf[0].getVarInfo()) :
                     &(leaf[1].getVarInfo()));

              stride = (varIn0 ? &(leaf[0]) : &(leaf[1]));
            }
          }
        }
        else if(leaf.info & expType::LR){
          if((leaf.value == "++") ||
             (leaf.value == "--")){

            var = &(leaf[0].getVarInfo());

            str = ((leaf.value == "++") ? "1" : "-1");
          }
        }

        viInfo_t &viInfo = db[*var];

        viInfo.info |= viType::isAnIterator;

        if(stride)
          viInfo.iteratorInfo.stride.load(*stride);
        else
          viInfo.iteratorInfo.stride.load(str);
      }

      if(wrongFormat){
        printf("[Magic Analyzer] For-loop update statement (3rd statement) is not standard, for example:\n  X op Y where op can be [+=] or [-=]\n  ++X, X++, --X, X--\n");
        return;
      }
    }

    void magician::analyzeWhileStatement(statement &s){
      typeHolder th = s.expRoot[0].calculateValue();

      if( !(th.type & noType) &&
          (th.boolValue() == false) ){

        db.getSmntInfo() &= ~analyzeInfo::isExecuted;
        return;
      }
    }

    void magician::analyzeFortranForStatement(statement &s){
      expNode &flatRoot = *(s.expRoot.makeFlatHandle());

      expNode &e0 = s.expRoot[0][0];

      //      [0] root
      //       [0] =
      // [0] iter    [1] doStart
      varInfo &iterVar = e0[0].getVarInfo();
      viInfo_t &viInfo = db[iterVar];

      varInfo *start  = &(e0[1].getVarInfo());
      varInfo *end    = NULL;
      varInfo *stride = NULL;

      for(int j = 0; j < flatRoot.leafCount; ++j){
        expNode &n = flatRoot[j];

        if(n.info & expType::varInfo){
          varInfo &var = n.getVarInfo();

          if(var.name.find("doEnd") != std::string::npos)
            end = &var;
          else if(var.name.find("doStride") != std::string::npos)
            stride = &var;
        }
      }

      expNode::freeFlatHandle(flatRoot);

      viInfo.iteratorInfo.start.load(*start);
      viInfo.iteratorInfo.end.load(*end);

      if(stride == NULL)
        viInfo.iteratorInfo.stride.load("1");
      else
        viInfo.iteratorInfo.stride.load(*stride);
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

      viInfo_t &viInfo = db[ varNode.getVarInfo() ];

      viInfo.addWrite(varNode);
    }

    void magician::addVariableWrite(expNode &varNode,
                                    expNode &setNode,
                                    const int brackets,
                                    expNode &bracketNode){
      const bool isUpdated = variableIsUpdated(varNode);

      if(isUpdated)
        addVariableRead(varNode, brackets, bracketNode);

      viInfo_t &viInfo = db[ varNode[0].getVarInfo() ];

      viInfo.addWrite(brackets, bracketNode);
    }

    void magician::addVariableRead(expNode &varNode){
      if(varNode.info & expType::variable){
        const int brackets = varNode.getVariableBracketCount();

        if(brackets)
          addVariableRead(varNode, brackets, *(varNode.getVariableBracket(0)->up));

        return;
      }

      viInfo_t &viInfo = db[ varNode.getVarInfo() ];

      viInfo.addRead(varNode);
    }

    void magician::addVariableRead(expNode &varNode,
                                   const int brackets,
                                   expNode &bracketNode){

      viInfo_t &viInfo = db[ varNode[0].getVarInfo() ];

      viInfo.addRead(brackets, bracketNode);
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
