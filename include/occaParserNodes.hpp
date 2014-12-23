#ifndef OCCA_PARSER_NODES_HEADER
#define OCCA_PARSER_NODES_HEADER

#include "occaParserDefines.hpp"
#include "occaParserTools.hpp"

namespace occa {
  namespace parserNS {
    //---[ Node ]-----------------------------------
    template <class TM>
    class node {
    public:
      node *left, *right, *up, *down;
      TM value;

      node();
      node(const TM &t);
      node(const node<TM> &n);

      node& operator = (const node<TM> &n);

      node<TM>* pop();

      node* push(node<TM> *n);
      node* push(const TM &t);

      node* pushDown(node<TM> *n);
      node* pushDown(const TM &t);

      void print(const std::string &tab = "");

      void printPtr(const std::string &tab = "");
    };

    template <class TM>
    void popAndGoRight(node<TM> *&n);

    template <class TM>
    void popAndGoLeft(node<TM> *&n);

    template <class TM>
    node<TM>* firstNode(node<TM> *n);

    template <class TM>
    node<TM>* lastNode(node<TM> *n);

    template <class TM>
    int length(node<TM> *n);
    //==============================================


    //---[ Str Node ]-------------------------------
    class strNode {
    public:
      strNode *left, *right, *up, *down;

      std::string value;
      int info, depth, sideDepth;

      int originalLine;

      strNode();
      strNode(const std::string &value_);
      strNode(const strNode &n);

      strNode& operator = (const strNode &n);

      void swapWith(strNode *n);
      void swapWithRight();
      void swapWithLeft();

      void moveLeftOf(strNode *n);
      void moveRightOf(strNode *n);

      strNode* clone() const;
      strNode* cloneTo(strNode *n) const;
      strNode* cloneNode() const;

      operator std::string () const;

      strNode* pop();

      strNode* push(strNode *node);
      strNode* push(const std::string &str);

      strNode* pushDown(strNode *node);
      strNode* pushDown(const std::string &str);

      node<strNode*> getStrNodesWith(const std::string &name_,
                                     const int type_ = everythingType);

      void flatten();

      bool freeLeft();
      bool freeRight();

      void print(const std::string &tab = "");

      std::string toString(const char spacing = ' ');
    };

    std::ostream& operator << (std::ostream &out, const strNode &n);

    void popAndGoRight(strNode *&node);
    void popAndGoLeft(strNode *&node);

    strNode* firstNode(strNode *node);
    strNode* lastNode(strNode *node);

    int length(strNode *node);

    void free(strNode *node);
    //==============================================
  };
};

#include "occaParserNodes.tpp"

#endif
