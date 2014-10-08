#include "occaParser.hpp"

namespace occa {
  namespace parserNamespace {
    namespace expType {
      const char L         = (1 << 0);
      const char C         = (1 << 1);
      const char R         = (1 << 2);
      const char type      = (1 << 3);
      const char var       = (1 << 4);
      const char function  = (1 << 5);
      const char pFunction = (1 << 6);
    };

    class expNode {
    public:
      std::string value;
      char info;

      int leafCount;
      expNode **leaves;
      varInfo *var;
      typeDef *type;

      expNode() :
        value(""),
        info(0),

        leafCount(0),
        leaves(NULL),
        var(NULL),
        type(NULL) {}

      void loadFromNode(strNode *n, const int depth = 0){
        strNode *nodeRoot = (depth ? n : n->clone());
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
          leaf->value = nodePos->value;

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
              sLeaf->loadFromNode(downNode, depth + 1);
          }

          nodePos = nodePos->right;
        }

        organizeLeaves();

        // [-] Need to free clone();
      }

      class int2 {
        int x, y;

        friend bool operator < (const int2 &a, const int2 &b){
          if(a.x != b.x)
            return (a.x < b.x);

          return (a.y < b.y);
        }
      };

      void organizeLeaves(){
        //---[ Level 0 ]------
        // [a][::][b]
        mergeNamespaces();
        //====================

        //---[ Level 1 ]------
        // class(...), class{1,2,3}
        markClassConstructs();

        // static_cast<>()
        markCasts();

        // func()
        markFunctionCalls();

        organizeLeaves(1);
        //====================

        //---[ Level 2 ]------
        // (class) x
        markClassCasts();

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
        for(int i = 0; i < leafCount; ++i){
          if(leaves[i]->leafCount)
            continue;

          opLevelMapIterator it = opLevelMap[level].find(leaves[i]->value);

          if(it == opLevelMap[level].end())
            continue;

          std::cout
            << "leaves[i]->value = " << leaves[i]->value << '\n';

          const int levelType = it->second;
          const int allTypes  = keywordType[leaves[i]->value];
        }
      }

      void mergeNamespaces(){
      }

      // class(...), class{1,2,3}
      void markClassConstructs(){
      }

      // static_cast<>()
      void markCasts(){
      }

      // func()
      void markFunctionCalls(){
      }

      // (class) x
      void markClassCasts(){
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
      void mergeLeftUnary(const int leafPos){
        expNode *leaf  = leaves[leafPos];
        expNode *sLeaf = leaves[leafPos + 1];

        for(int i = (leafPos + 1); i < leafCount; ++i)
          leaves[i - 1] = leaves[i];

        --leafCount;

        leaf->leafCount = 1;
        leaf->info      = expType::L;

        leaf->leaves    = new expNode*[1];
        leaf->leaves[0] = sLeaf;
      }

      // i[++]
      void mergeRightUnary(const int leafPos){
        expNode *leaf  = leaves[leafPos];
        expNode *sLeaf = leaves[leafPos - 1];

        leaves[leafPos - 1] = leaf;

        --leafCount;

        for(int i = leafPos; i < leafCount; ++i)
          leaves[i] = leaves[i + 1];

        leaf->leafCount = 1;
        leaf->info      = expType::R;

        leaf->leaves    = new expNode*[1];
        leaf->leaves[0] = sLeaf;
      }

      // a [+] b
      void mergeBinary(const int leafPos){
        expNode *leaf   = leaves[leafPos];
        expNode *sLeafL = leaves[leafPos - 1];
        expNode *sLeafR = leaves[leafPos + 1];

        leaves[leafPos - 1] = leaf;

        leafCount -= 2;

        for(int i = leafPos; i < leafCount; ++i)
          leaves[i] = leaves[i + 2];

        leaf->leafCount = 2;
        leaf->info      = (expType::L | expType::R);

        leaf->leaves    = new expNode*[2];
        leaf->leaves[0] = sLeafL;
        leaf->leaves[1] = sLeafR;
      }

      // a [?] b : c
      void mergeTernary(const int leafPos){
        expNode *leaf   = leaves[leafPos];
        expNode *sLeafL = leaves[leafPos - 1];
        expNode *sLeafC = leaves[leafPos + 1];
        expNode *sLeafR = leaves[leafPos + 3];

        leaves[leafPos - 1] = leaf;

        leafCount -= 4;

        for(int i = leafPos; i < leafCount; ++i)
          leaves[i] = leaves[i + 4];

        leaf->leafCount = 3;
        leaf->info      = (expType::L | expType::C | expType::R);

        leaf->leaves    = new expNode*[3];
        leaf->leaves[0] = sLeafL;
        leaf->leaves[1] = sLeafC;
        leaf->leaves[2] = sLeafR;
      }

      friend std::ostream& operator << (std::ostream &out, const expNode &n){
        out << n.value;

        for(int i = 0; i < n.leafCount; ++i)
          out << *(n.leaves[i]);

        if(n.info == expType::C){
          const char startChar = n.value[0];

          out << (char) ((')' * (startChar == '(')) +
                         (']' * (startChar == '[')) +
                         ('}' * (startChar == '{')));
        }

        return out;
      }
    };

    void test(){
      strNode *n = labelCode( splitContent("(1+2*3%2|1+10&3^1)") );

      expNode expRoot;
      expRoot.loadFromNode(n);

      std::cout
        << "expRoot = " << expRoot << '\n';

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
