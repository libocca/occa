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

        // [-] Need to free clone();
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
