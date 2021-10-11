#ifndef OCCA_INTERNAL_LANG_EXPR_EXPRNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_EXPRNODE_HEADER

#include <stack>
#include <vector>

#include <occa/internal/io/output.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/lang/printer.hpp>
#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/expr/exprNodeArray.hpp>

namespace occa {
  namespace lang {
    class exprNode;
    class type_t;
    class variable_t;
    class function_t;

    typedef std::vector<exprNode*> exprNodeVector;
    typedef std::stack<exprNode*>  exprNodeStack;
    typedef std::vector<token_t*>  tokenVector;

    // Variables to help make output prettier
    static const int PRETTIER_MAX_VAR_WIDTH  = 30;
    static const int PRETTIER_MAX_LINE_WIDTH = 80;

    namespace exprNodeType {
      extern const udim_t empty;
      extern const udim_t primitive;
      extern const udim_t char_;
      extern const udim_t string;
      extern const udim_t identifier;
      extern const udim_t type;
      extern const udim_t vartype;
      extern const udim_t variable;
      extern const udim_t function;
      extern const udim_t value;

      extern const udim_t rawOp;
      extern const udim_t leftUnary;
      extern const udim_t rightUnary;
      extern const udim_t binary;
      extern const udim_t ternary;
      extern const udim_t op;

      extern const udim_t pair;

      extern const udim_t subscript;
      extern const udim_t call;

      extern const udim_t sizeof_;
      extern const udim_t sizeof_pack_;
      extern const udim_t new_;
      extern const udim_t delete_;
      extern const udim_t throw_;

      extern const udim_t typeid_;
      extern const udim_t noexcept_;
      extern const udim_t alignof_;

      extern const udim_t const_cast_;
      extern const udim_t dynamic_cast_;
      extern const udim_t static_cast_;
      extern const udim_t reinterpret_cast_;

      extern const udim_t funcCast;
      extern const udim_t parenCast;
      extern const udim_t constCast;
      extern const udim_t staticCast;
      extern const udim_t reinterpretCast;
      extern const udim_t dynamicCast;

      extern const udim_t parentheses;
      extern const udim_t tuple;
      extern const udim_t cudaCall;
      extern const udim_t lambda;
      extern const udim_t dpcppLocalMemory;
      extern const udim_t dpcppAtomic;
    }

    class exprNode {
    public:
      token_t *token;

      exprNode(token_t *token_);

      virtual ~exprNode();

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast exprNode::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast exprNode::to",
                   ptr != NULL);
        return *ptr;
      }

      virtual udim_t type() const = 0;

      virtual exprNode* clone() const = 0;
      static exprNode* clone(exprNode *expr);

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      exprNodeVector getNestedChildren();

      void pushNestedChildNodes(exprNodeVector &children);

      virtual void pushChildNodes(exprNodeVector &children);

      bool replaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual bool safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual bool hasAttribute(const std::string &attr) const;

      virtual variable_t* getVariable();

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const = 0;

      std::string toString() const;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;

      void debugPrint() const;

      virtual void debugPrint(const std::string &prefix) const = 0;

      void childDebugPrint(const std::string &prefix) const;
    };

    io::output& operator << (io::output &out,
                             const exprNode &node);

    printer& operator << (printer &pout,
                          const exprNode &node);

    void cloneExprNodeVector(exprNodeVector &dest,
                             const exprNodeVector &src);

    void freeExprNodeVector(exprNodeVector &vec);
  }
}

#endif

