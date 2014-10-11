#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    namespace expType {
      const int root            = (1 << 0);

      const int LCR             = (7 << 1);
      const int L               = (1 << 1);
      const int C               = (1 << 2);
      const int R               = (1 << 3);

      const int qualifier       = (1 << 4);
      const int type            = (1 << 5);
      const int presetValue     = (1 << 6);
      const int variable        = (1 << 7);
      const int function        = (1 << 8);
      const int functionPointer = (1 << 9);
    };

    class expNode {
    public:
      std::string value;
      int info;

      expNode *up;

      int leafCount;
      expNode **leaves;
      varInfo *var;
      typeDef *type;

      expNode() :
        value(""),
        info(expType::root),

        up(NULL),

        leafCount(0),
        leaves(NULL),
        var(NULL),
        type(NULL) {}

      void loadFromNode(strNode *n){
        strNode *nClone = n->clone();
        initLoadFromNode(nClone);

        initOrganization();
        organizeLeaves();

        // [-] Need to free nClone;
      }

      void initLoadFromNode(strNode *n){
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

      void initOrganization(){
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

      void organizeLeaves(){
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

      void organizeLeaves(const int level){
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

      int mergeRange(const int newLeafType,
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
      void mergeNamespaces(){
      }

      int mergeNamespace(const int leafPos){
        return 0;
      }

      // [const] int x
      void mergeQualifiers(){
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
      void mergeTypes(){
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
      void mergeVariables(){
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
      void mergeFunctionPointers(){
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
      void mergeClassConstructs(){
      }

      // static_cast<>()
      void mergeCasts(){
      }

      // func()
      void mergeFunctionCalls(){
      }

      void mergeArguments(){
        for(int i = 0; i < leafCount; i += 2){
          leaves[i/2] = leaves[i];

          if((i + 1) < leafCount)
            freeLeaf(i + 1);
        }

        leafCount = ((leafCount / 2) + 1);
      }

      // (class) x
      void mergeClassCasts(){
      }

      // sizeof x
      void mergeSizeOf(){
      }

      // new, new [], delete, delete []
      void mergeNewsAndDeletes(){
      }

      // throw x
      void mergeThrows(){
      }

      // [++]i
      int mergeLeftUnary(const int leafPos){
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
      int mergeRightUnary(const int leafPos){
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
      int mergeBinary(const int leafPos){
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
      int mergeTernary(const int leafPos){
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

      void freeLeaf(const int leafPos){
        leaves[leafPos]->free();
        delete leaves[leafPos];
      }

      void free(){
        for(int i = 0; i < leafCount; ++i){
          leaves[i]->free();
          delete leaves[i];
        }

        delete [] leaves;
      }

      void print(const std::string &tab = ""){
        std::cout << tab << "| " << value << '\n';

        for(int i = 0; i < leafCount; ++i)
          leaves[i]->print(tab + "  ");
      }

      friend std::ostream& operator << (std::ostream &out, const expNode &n){

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
            for(int i = 0; i < (n.leafCount - 1); ++i)
              out << *(n.leaves[i]) << ' ';

            out << *(n.leaves[n.leafCount - 1]);
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
            out << *(n.leaves[0]) << ' ' << *(n.leaves[1]);
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

    void test(){
      strNode *n = labelCode( splitContent("const int * const * (*func)(int **x, int, int)") );
      // strNode *n = labelCode( splitContent("(1+2*3%2|1+10&3^1)") );

      expNode expRoot;
      expRoot.loadFromNode(n);

      std::cout
        << "expRoot = " << expRoot << '\n';

      expRoot.print();

      throw 1;
    }
  };
};

int main(int argc, char **argv){
  occa::parserNamespace::test();

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/test.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/openclTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/cudaTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/midg.okl");
  //   std::cout << parsedContent << '\n';
  // }

  {
    occa::parser parser;
    std::string parsedContent = parser.parseFile("tests/cleanTest.c");
    std::cout << parsedContent << '\n';
  }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/clangTest.c");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/addVectors.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("tests/PCGpart1.cl");
  //   std::cout << parsedContent << '\n';
  // }
}
