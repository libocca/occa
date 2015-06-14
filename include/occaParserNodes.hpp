#ifndef OCCA_PARSER_NODES_HEADER
#define OCCA_PARSER_NODES_HEADER

#include "occaParserDefines.hpp"

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
      node* push(node<TM> *nStart, node<TM> *nEnd);
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
  };
};

#include "occaParserNodes.tpp"

#endif
