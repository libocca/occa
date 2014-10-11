#include "occaParserNodes.hpp"

namespace occa {
  namespace parserNamespace {
    //---[ Str Node ]-------------------------------
    strNode::strNode() :
      left(NULL),
      right(NULL),
      up(NULL),
      down(),

      value(""),

      type(0),
      depth(0),

      sideDepth(0) {}

    strNode::strNode(const std::string &value_) :
      left(NULL),
      right(NULL),
      up(NULL),
      down(),

      value(value_),

      type(0),
      depth(0),

      sideDepth(0) {}

    strNode::strNode(const strNode &n) :
      left(n.left),
      right(n.right),
      up(n.up),
      down(n.down),

      value(n.value),

      type(n.type),
      depth(n.depth),

      sideDepth(n.sideDepth) {}

    strNode& strNode::operator = (const strNode &n){
      left  = n.left;
      right = n.right;
      up    = n.up;
      down  = n.down;

      value = n.value;

      type  = n.type;
      depth = n.depth;

      sideDepth = n.sideDepth;

      return *this;
    }

    void strNode::swapWith(strNode *n){
      if(n == NULL)
        return;

      strNode *l1 = (left  == n) ? this : left;
      strNode *c1  = this;
      strNode *r1 = (right == n) ? this : right;

      strNode *l2 = (n->left  == this) ? n : n->left;
      strNode *c2 = right;
      strNode *r2 = (n->right == this) ? n : n->right;

      n->left  = l1;
      n->right = r1;

      left  = l2;
      right = r2;

      if(l1)
        l1->right = n;
      if(r1)
        r1->left  = n;

      if(l2)
        l2->right = this;
      if(r2)
        r2->left  = this;
    }

    void strNode::swapWithRight(){
      swapWith(right);
    }

    void strNode::swapWithLeft(){
      swapWith(left);
    }

    void strNode::moveLeftOf(strNode *n){
      if(n == NULL)
        return;

      if(left)
        left->right = right;
      if(right)
        right->left = left;

      left = n->left;

      if(n->left)
        left->right = this;

      right   = n;
      n->left = this;
    }

    void strNode::moveRightOf(strNode *n){
      if(n == NULL)
        return;

      if(left)
        left->right = right;
      if(right)
        right->left = left;

      right = n->right;

      if(n->right)
        right->left = this;

      left     = n;
      n->right = this;
    }

    strNode* strNode::clone() const {
      strNode *newNode = new strNode();

      newNode->value = value;
      newNode->type  = type;

      if(right){
        newNode->right = right->clone();
        newNode->right->left = newNode;
      }

      newNode->up = up;

      newNode->depth = depth;

      const int downCount = down.size();

      for(int i = 0; i < downCount; ++i)
        newNode->down.push_back( down[i]->clone() );

      return newNode;
    }

    strNode::operator std::string () const {
      return value;
    }

    strNode* strNode::pop(){
      if(left != NULL)
        left->right = right;

      if(right != NULL)
        right->left = left;

      return this;
    }

    strNode* strNode::push(strNode *node){
      strNode *rr = right;

      right = node;

      right->left  = this;
      right->right = rr;
      right->up    = up;

      if(rr)
        rr->left = right;

      return right;
    }

    strNode* strNode::push(const std::string &value_){
      strNode *newNode = new strNode(value_);

      newNode->type      = type;
      newNode->depth     = depth;
      newNode->sideDepth = sideDepth;

      return push(newNode);
    };

    strNode* strNode::pushDown(strNode *node){
      node->up        = this;
      node->sideDepth = down.size();

      down.push_back(node);

      return node;
    };

    strNode* strNode::pushDown(const std::string &value_){
      strNode *newNode = new strNode(value_);

      newNode->type  = type;
      newNode->depth = depth + 1;

      return pushDown(newNode);
    };

    bool strNode::hasType(const int type_){
      if(type & type_)
        return true;

      const int downCount = down.size();

      for(int i = 0; i < downCount; ++i)
        if(down[i]->hasType(type_))
          return true;

      return false;
    }

    node<strNode*> strNode::getStrNodesWith(const std::string &name_,
                                            const int type_){
      node<strNode*> nRootNode;
      node<strNode*> *nNodePos = nRootNode.push(new strNode());

      strNode *nodePos = this;

      while(nodePos){
        if((nodePos->type & everythingType) &&
           (nodePos->value == name_)){

          nNodePos->value = nodePos;
          nNodePos = nNodePos->push(new strNode());
        }

        const int downCount = nodePos->down.size();

        for(int i = 0; i < downCount; ++i){
          node<strNode*> downRootNode = down[i]->getStrNodesWith(name_, type_);

          if(downRootNode.value != NULL){
            node<strNode*> *lastDownNode = (node<strNode*>*) downRootNode.value;

            node<strNode*> *nnpRight = nNodePos;
            nNodePos = nNodePos->left;

            nNodePos->right       = downRootNode.right;
            nNodePos->right->left = nNodePos;

            nNodePos = lastDownNode;
          }
        }

        nodePos = nodePos->right;
      }

      nNodePos = nNodePos->left;

      if(nNodePos == &nRootNode)
        nRootNode.value = NULL;
      else
        nRootNode.value = (strNode*) nNodePos;

      delete nNodePos->right;
      nNodePos->right = NULL;

      return nRootNode;
    }

    void strNode::flatten(){
      strNode *nodePos = this;

      while(nodePos){
        const int downCount = nodePos->down.size();

        if(downCount){
          for(int i = 1; i < downCount; ++i){
            strNode *d1 = lastNode(nodePos->down[i - 1]);
            strNode *d2 = nodePos->down[i];

            d1->right = d2;
            d2->left  = d1;
          }

          strNode *lastD = lastNode(nodePos->down[downCount - 1]);

          lastD->right = nodePos->right;

          if(nodePos->right)
            nodePos->right->left = lastD;

          nodePos->right         = nodePos->down[0];
          nodePos->down[0]->left = nodePos;

          nodePos->down.clear();
        }

        nodePos = nodePos->right;
      }
    }

    bool strNode::freeLeft(){
      if((left != NULL) && (left != this)){
        strNode *l = left;

        left = l->left;

        if(left != NULL)
          left->right = this;

        delete l;

        return true;
      }

      return false;
    }

    bool strNode::freeRight(){
      if((right != NULL) && (right != this)){
        strNode *r = right;

        right = r->right;

        if(right != NULL)
          right->left = this;

        delete r;

        return true;
      }

      return false;
    }

    void strNode::print(const std::string &tab){
      strNode *nodePos = this;

      while(nodePos){
        std::cout << tab << "[" << *nodePos << "] (" << nodePos->type << ")\n";

        const int downCount = (nodePos->down).size();

        if(downCount)
          printf("--------------------------------------------\n");

        for(int i = 0; i < downCount; ++i){
          (nodePos->down[i])->print(tab + "  ");
          printf("--------------------------------------------\n");
        }

        nodePos = nodePos->right;
      }
    }

    std::ostream& operator << (std::ostream &out, const strNode &n){
      out << n.value;
      return out;
    }

    void popAndGoRight(strNode *&node){
      strNode *left  = node->left;
      strNode *right = node->right;

      if(left != NULL)
        left->right = right;

      if(right != NULL)
        right->left = left;

      delete node;

      node = right;
    }

    void popAndGoLeft(strNode *&node){
      strNode *left  = node->left;
      strNode *right = node->right;

      if(left != NULL)
        left->right = right;

      if(right != NULL)
        right->left = left;

      delete node;

      node = left;
    }

    strNode* firstNode(strNode *node){
      if((node == NULL) ||
         (node->left == NULL))
        return node;

      strNode *end = node;

      while(end->left)
        end = end->left;

      return end;
    }

    strNode* lastNode(strNode *node){
      if((node == NULL) ||
         (node->right == NULL))
        return node;

      strNode *end = node;

      while(end->right)
        end = end->right;

      return end;
    }

    int length(strNode *node){
      int l = 0;

      while(node){
        ++l;
        node = node->right;
      }

      return l;
    }

    void free(strNode *node){
      const int downCount = (node->down).size();

      for(int i = 0; i < downCount; ++i)
        free( (node->down)[i] );

      while(node->freeRight())
        /* Do Nothing */;

      while(node->freeLeft())
        /* Do Nothing */;

      delete node;
    }
    //==============================================


    //---[ Exp Node ]-------------------------------
    expNode::expNode() :
      value(""),
      info(expType::root),

      up(NULL),

      leafCount(0),
      leaves(NULL),
      var(NULL),
      type(NULL) {}

    void expNode::loadFromNode(strNode *n){
      strNode *nClone = n->clone();
      initLoadFromNode(nClone);

      initOrganization();
      organizeLeaves();

      // [-] Need to free nClone;
    }

    void expNode::initLoadFromNode(strNode *n){
      strNode *nodeRoot = n;
      strNode *nodePos  = nodeRoot;

      while(nodePos){
        leafCount += (1 + nodePos->down.size());
        nodePos = nodePos->right;
      }

      if(leafCount == 0)
        return;

      nodePos = nodeRoot;

      leaves = new expNode*[leafCount];
      leafCount = 0;

      while(nodePos){
        expNode *&leaf = leaves[leafCount++];

        leaf        = new expNode;
        leaf->up    = this;
        leaf->value = nodePos->value;

        if(nodePos->type & unknownVariable){

          // [-] Temp until I merge with parser
          if(nodePos->down.size() &&
             (nodePos->down[0]->type & parentheses)){

            leaf->info = expType::function;
          }
          else{
            leaf->info = expType::variable;
          }
        }

        else if(nodePos->type & presetValue){
          leaf->info = expType::presetValue;
        }

        else if(nodePos->type & descriptorType){
          // Case const int [*] const [*] a;
          if(nodePos->type & operatorType){
            // [--] Check for custom variable-type when
            //        putting this in parser
            if(nodePos->left &&
               (nodePos->left->type & descriptorType)){

              leaf->info = expType::qualifier;
            }
          }
          else{
            if(nodePos->type & qualifierType)
              leaf->info = expType::qualifier;
            else
              leaf->info = expType::type;
          }
        }

        const int downCount = nodePos->down.size();

        for(int i = 0; i < downCount; ++i){
          strNode *downNode = nodePos->down[i];
          strNode *lastDown = lastNode(downNode);

          std::string sValue = downNode->value;

          // Get rid of ()'s and stuff
          popAndGoRight(downNode);
          popAndGoLeft(lastDown);

          expNode *&sLeaf = leaves[leafCount++];

          sLeaf        = new expNode;
          sLeaf->value = sValue;
          sLeaf->info  = expType::C;

          // Case: ()
          if(lastDown != NULL)
            sLeaf->initLoadFromNode(downNode);
        }

        nodePos = nodePos->right;
      }
    }

    void expNode::initOrganization(){
      // Init ()'s bottom -> up
      for(int i = 0; i < leafCount; ++i){
        if(leaves[i]->leafCount)
          leaves[i]->initOrganization();
      }

      //---[ Level 0 ]------
      // [a][::][b]
      mergeNamespaces();

      // [const] int [*] x
      mergeQualifiers();

      // [[const] [int] [*]] x
      mergeTypes();

      // [[[const] [int] [*]] [x]]
      mergeVariables();

      // [qualifiers] [type] (*[name]) ([args])
      mergeFunctionPointers();
      //====================
    }

    void expNode::organizeLeaves(){
      // Organize leaves bottom -> up
      for(int i = 0; i < leafCount; ++i){
        if((leaves[i]->leafCount) &&
           !(leaves[i]->info & (expType::type      |
                                expType::qualifier |
                                expType::function  |
                                expType::functionPointer))){

          leaves[i]->organizeLeaves();
        }
      }

      //---[ Level 1 ]------
      // class(...), class{1,2,3}
      mergeClassConstructs();

      // static_cast<>()
      mergeCasts();

      // func()
      mergeFunctionCalls();

      organizeLeaves(1);
      //====================

      //---[ Level 2 ]------
      // (class) x
      mergeClassCasts();

      // sizeof x
      mergeSizeOf();

      // new, new [], delete, delete []
      mergeNewsAndDeletes();

      organizeLeaves(2);
      //====================

      //---[ Level 3-14 ]---
      for(int i = 3; i <= 14; ++i)
        organizeLeaves(i);
      //====================

      //---[ Level 15 ]-----
      // throw x
      mergeThrows();
      //====================

      //---[ Level 16 ]-----
      organizeLeaves(16);
      //====================
    }

    void expNode::organizeLeaves(const int level){
      int leafPos = 0;

      while(leafPos < leafCount){
        if(leaves[leafPos]->leafCount){
          ++leafPos;
          continue;
        }

        opLevelMapIterator it = opLevelMap[level].find(leaves[leafPos]->value);

        if(it == opLevelMap[level].end()){
          ++leafPos;
          continue;
        }

        const int levelType = it->second;

        if(levelType & unitaryOperatorType){
          bool updateNow = true;

          // Cases:  1 + [-]1
          //         (+1)
          if((leaves[leafPos]->value.size() == 1) &&
             ((leaves[leafPos]->value[0] == '+') ||
              (leaves[leafPos]->value[0] == '-'))){

            if(leafPos &&
               !(leaves[leafPos - 1]->info & expType::LCR))
              updateNow = false;
          }

          if(updateNow){
            if(levelType & lUnitaryOperatorType)
              leafPos = mergeLeftUnary(leafPos);
            else
              leafPos = mergeRightUnary(leafPos);
          }
          else
            ++leafPos;
        }
        else if(levelType & binaryOperatorType)
          leafPos = mergeBinary(leafPos);
        else if(levelType & ternaryOperatorType)
          leafPos = mergeTernary(leafPos);
        else
          ++leafPos;
      }
    }

    int expNode::mergeRange(const int newLeafType,
                            const int leafPosStart,
                            const int leafPosEnd){
      expNode *newLeaf = new expNode;

      newLeaf->up        = this;
      newLeaf->info      = newLeafType;
      newLeaf->leafCount = (leafPosEnd - leafPosStart + 1);
      newLeaf->leaves    = new expNode*[newLeaf->leafCount];

      for(int i = 0; i < newLeaf->leafCount; ++i){
        newLeaf->leaves[i]     = leaves[leafPosStart + i];
        newLeaf->leaves[i]->up = newLeaf;
      }

      leaves[leafPosStart] = newLeaf;

      for(int i = (leafPosEnd + 1); i < leafCount; ++i)
        leaves[leafPosStart + i - leafPosEnd] = leaves[i];

      leafCount -= (newLeaf->leafCount - 1);

      return (leafPosStart + 1);
    }

    // [a][::][b]
    void expNode::mergeNamespaces(){
    }

    int expNode::mergeNamespace(const int leafPos){
      return 0;
    }

    // [const] int x
    void expNode::mergeQualifiers(){
      int leafPos = 0;

      while(leafPos < leafCount){
        if(leaves[leafPos]->info & expType::qualifier){
          int leafPosStart = leafPos;
          int leafPosEnd   = leafPos;

          while((leafPosEnd < leafCount) &&
                (leaves[leafPosEnd]->info & expType::qualifier))
            ++leafPosEnd;

          --leafPosEnd;

          leafPos = mergeRange(expType::qualifier,
                               leafPosStart, leafPosEnd);
        }
        else
          ++leafPos;
      }
    }

    // [[const] [int] [*]] x
    void expNode::mergeTypes(){
      int leafPos = 0;

      while(leafPos < leafCount){
        if(leaves[leafPos]->info & expType::type){
          int leafPosStart = leafPos;
          int leafPosEnd   = leafPos;

          if(leafPos &&
             (leaves[leafPos - 1]->info & expType::qualifier)){

            --leafPosStart;
          }

          if(((leafPos + 1) < leafCount) &&
             (leaves[leafPos + 1]->info & expType::qualifier)){

            ++leafPosEnd;
          }

          leafPos = mergeRange(expType::type,
                               leafPosStart, leafPosEnd);
        }
        else
          ++leafPos;
      }
    }

    // [[[const] [int] [*]] [x]]
    void expNode::mergeVariables(){
      int leafPos = 0;

      while(leafPos < leafCount){
        if((leaves[leafPos]->info & expType::type) && // [[const] [int] [*]]
           ((leafPos + 1) < leafCount)             && // [x]
           (leaves[leafPos + 1]->info & expType::variable)){

          leafPos = mergeRange(expType::variable,
                               leafPos, leafPos + 1);
        }
        else
          ++leafPos;
      }
    }

    // 1 [type]                           2 [(]       3 [(]
    // [[qualifiers] [type] [qualifiers]] [(*[name])] [([args])]
    void expNode::mergeFunctionPointers(){
      int leafPos = 0;

      while(leafPos < leafCount){
        if((leaves[leafPos]->info & expType::type)   &&     // 1
           ((leafPos + 2) < leafCount)               &&     // Has 2 & 3
           (leaves[leafPos + 1]->info == expType::C) &&     // 2
           (leaves[leafPos + 2]->info == expType::C) &&     // 3
           (leaves[leafPos + 1]->leaves[0]->value == "*")){ // 2.5

          expNode *newLeaf = new expNode;

          newLeaf->up        = this;
          newLeaf->info      = expType::functionPointer;
          newLeaf->leafCount = 3;
          newLeaf->leaves    = new expNode*[3];
          newLeaf->leaves[0] = leaves[leafPos];
          newLeaf->leaves[1] = leaves[leafPos + 1]->leaves[1];
          newLeaf->leaves[2] = leaves[leafPos + 2];

          // Merge arguments in [leafPos + 2]
          newLeaf->leaves[2]->mergeArguments();

          // Don't kill the name of the function pointer
          leaves[leafPos + 1]->leafCount = 1;
          freeLeaf(leafPos + 1);

          leaves[leafPos] = newLeaf;

          for(int i = (leafPos + 3); i < leafCount; ++i)
            leaves[i - 2] = leaves[i];

          ++leafPos;
          leafCount -= 2;
        }
        else
          ++leafPos;
      }
    }

    // class(...), class{1,2,3}
    void expNode::mergeClassConstructs(){
    }

    // static_cast<>()
    void expNode::mergeCasts(){
    }

    // func()
    void expNode::mergeFunctionCalls(){
    }

    void expNode::mergeArguments(){
      for(int i = 0; i < leafCount; i += 2){
        leaves[i/2] = leaves[i];

        if((i + 1) < leafCount)
          freeLeaf(i + 1);
      }

      leafCount = ((leafCount / 2) + 1);
    }

    // (class) x
    void expNode::mergeClassCasts(){
    }

    // sizeof x
    void expNode::mergeSizeOf(){
    }

    // new, new [], delete, delete []
    void expNode::mergeNewsAndDeletes(){
    }

    // throw x
    void expNode::mergeThrows(){
    }

    // [++]i
    int expNode::mergeLeftUnary(const int leafPos){
      expNode *leaf  = leaves[leafPos];
      expNode *sLeaf = leaves[leafPos + 1];

      for(int i = (leafPos + 1); i < leafCount; ++i)
        leaves[i - 1] = leaves[i];

      --leafCount;

      leaf->info      = expType::L;
      leaf->leafCount = 1;
      leaf->leaves    = new expNode*[1];
      leaf->leaves[0] = sLeaf;

      sLeaf->up = leaf;

      return (leafPos + 1);
    }

    // i[++]
    int expNode::mergeRightUnary(const int leafPos){
      expNode *leaf  = leaves[leafPos];
      expNode *sLeaf = leaves[leafPos - 1];

      leaves[leafPos - 1] = leaf;

      --leafCount;

      for(int i = leafPos; i < leafCount; ++i)
        leaves[i] = leaves[i + 1];

      leaf->info      = expType::R;
      leaf->leafCount = 1;
      leaf->leaves    = new expNode*[1];
      leaf->leaves[0] = sLeaf;

      sLeaf->up = leaf;

      return (leafPos + 1);
    }

    // a [+] b
    int expNode::mergeBinary(const int leafPos){
      expNode *leaf   = leaves[leafPos];
      expNode *sLeafL = leaves[leafPos - 1];
      expNode *sLeafR = leaves[leafPos + 1];

      leaves[leafPos - 1] = leaf;

      leafCount -= 2;

      for(int i = leafPos; i < leafCount; ++i)
        leaves[i] = leaves[i + 2];

      leaf->info      = (expType::L | expType::R);
      leaf->leafCount = 2;
      leaf->leaves    = new expNode*[2];
      leaf->leaves[0] = sLeafL;
      leaf->leaves[1] = sLeafR;

      sLeafL->up = leaf;
      sLeafR->up = leaf;

      return leafPos;
    }

    // a [?] b : c
    int expNode::mergeTernary(const int leafPos){
      expNode *leaf   = leaves[leafPos];
      expNode *sLeafL = leaves[leafPos - 1];
      expNode *sLeafC = leaves[leafPos + 1];
      expNode *sLeafR = leaves[leafPos + 3];

      leaves[leafPos - 1] = leaf;

      leafCount -= 4;

      for(int i = leafPos; i < leafCount; ++i)
        leaves[i] = leaves[i + 4];

      leaf->info      = (expType::L | expType::C | expType::R);
      leaf->leafCount = 3;
      leaf->leaves    = new expNode*[3];
      leaf->leaves[0] = sLeafL;
      leaf->leaves[1] = sLeafC;
      leaf->leaves[2] = sLeafR;

      sLeafL->up = leaf;
      sLeafC->up = leaf;
      sLeafR->up = leaf;

      return leafPos;
    }

    //---[ Custom Type Info ]---------
    bool expNode::qualifierEndsWithStar() const {
      if( !(info & expType::qualifier) )
        return false;

      if(leafCount)
        return leaves[leafCount - 1]->qualifierEndsWithStar();
      else
        return (value == "*");
    }

    bool expNode::typeEndsWithStar() const {
      if( !(info & expType::type) ||
          (leafCount == 0) )
        return false;

      if(leaves[leafCount - 1]->info & expType::qualifier)
        return leaves[leafCount - 1]->qualifierEndsWithStar();

      return false;
    }
    //================================

    void expNode::freeLeaf(const int leafPos){
      leaves[leafPos]->free();
      delete leaves[leafPos];
    }

    void expNode::free(){
      for(int i = 0; i < leafCount; ++i){
        leaves[i]->free();
        delete leaves[i];
      }

      delete [] leaves;
    }

    void expNode::print(const std::string &tab){
      std::cout << tab << "[";

      bool printedSomething = false;
      for(int i = 0; i < 10; ++i){
        if(info & (1 << i)){
          if(printedSomething)
            std::cout << ',';

          std::cout << i;

          printedSomething = true;
        }
      }

      std::cout << "] " << value << '\n';

      for(int i = 0; i < leafCount; ++i)
        leaves[i]->print(tab + "    ");
    }

    expNode::operator std::string () const {
      std::stringstream ss;
      ss << *this;
      return ss.str();
    }

    std::ostream& operator << (std::ostream &out, const expNode &n){
      switch(n.info){
      case (expType::root):{
        for(int i = 0; i < n.leafCount; ++i)
          out << *(n.leaves[i]);

        break;
      }

      case (expType::L):{
        out << n.value << *(n.leaves[0]);

        break;
      }
      case (expType::R):{
        out << *(n.leaves[0]) << n.value;

        break;
      }
      case (expType::L | expType::R):{
        out << *(n.leaves[0]) << n.value << *(n.leaves[1]);

        break;
      }
      case (expType::L | expType::C | expType::R):{
        out << *(n.leaves[0]) << '?' << *(n.leaves[1]) << ':' << *(n.leaves[2]);

        break;
      }
      case expType::C:{
        const char startChar = n.value[0];

        out << startChar;

        for(int i = 0; i < n.leafCount; ++i)
          out << *(n.leaves[i]);

        out << (char) ((')' * (startChar == '(')) +
                       (']' * (startChar == '[')) +
                       ('}' * (startChar == '{')));

        break;
      }

      case expType::qualifier:{
        if(n.leafCount){
          int leafPos = 0;

          std::string strLeaf = (std::string) *(n.leaves[0]);
          bool lastStar = (strLeaf == "*");

          out << strLeaf;

          for(int i = 1; i < n.leafCount; ++i){
            std::string strLeaf = (std::string) *(n.leaves[i]);
            const bool thisStar = (strLeaf == "*");

            if( !(thisStar && lastStar) )
              out << ' ';

            out << strLeaf;

            lastStar = thisStar;
          }
        }
        else{
          out << n.value;
        }

        break;
      }
      case expType::type:{
        // [const] [int] [*]
        if(n.leafCount){
          for(int i = 0; i < (n.leafCount - 1); ++i)
            out << *(n.leaves[i]) << ' ';

          out << *(n.leaves[n.leafCount - 1]);
        }
        // [int]
        else{
          out << n.value;
        }

        break;
      }
      case expType::presetValue:{
        out << n.value;

        break;
      }
      case expType::variable:{
        // [[[const] [int] [*]] [x]]
        if(n.leafCount){
          out << *(n.leaves[0]);

          if( !(n.leaves[0]->typeEndsWithStar()) )
            out << ' ';

          out << *(n.leaves[1]);
        }
        // [x]
        else{
          out << n.value;
        }

        break;
      }
      case expType::function:{
        out << n.value << '(';

        if(n.leafCount){
          for(int i = 0; i < (n.leafCount - 1); ++i)
            out << *(n.leaves[i]) << ", ";

          out << *(n.leaves[n.leafCount - 1]);
        }

        out << ')';

        break;
      }
      case expType::functionPointer:{
        out << *(n.leaves[0]) << " (*" << *(n.leaves[1]) << ")"
            << '(';

        expNode *argNode = n.leaves[2];

        if(argNode->leafCount){
          for(int i = 0; i < (argNode->leafCount - 1); ++i)
            out << *(argNode->leaves[i]) << ", ";

          out << *(argNode->leaves[argNode->leafCount - 1]);
        }

        out << ')';

        break;
      }
      };

      return out;
    }
  };
  //==============================================
};
