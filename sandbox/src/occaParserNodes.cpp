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
  };
};
