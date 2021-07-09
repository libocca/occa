#include <occa/internal/lang/expr/lambdaNode.hpp>

namespace occa
{
  namespace lang
  {
    lambdaNode::lambdaNode(token_t *token_,
                           lambda_t &value_) : exprNode(token_),
                                               value(value_) {}

    lambdaNode::lambdaNode(const lambdaNode &node) : exprNode(node.token),
                                                     value(node.value) {}

    lambdaNode::~lambdaNode() {}

    udim_t lambdaNode::type() const
    {
      return exprNodeType::lambda;
    }

    exprNode *lambdaNode::clone() const
    {
      return new lambdaNode(token, value);
    }

    bool lambdaNode::hasAttribute(const std::string &attr) const
    {
      return value.hasAttribute(attr);
    }

    void lambdaNode::print(printer &pout) const
    {
      // pout << value;
      value.printDeclaration(pout);
    }

    void lambdaNode::debugPrint(const std::string &prefix) const
    {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                 << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (lambda)\n";
    }
  }
}
