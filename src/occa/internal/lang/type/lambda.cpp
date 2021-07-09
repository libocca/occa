#include <occa/internal/lang/type/lambda.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa
{
  namespace lang
  {
    lambda_t::lambda_t()
        : function_t(),
          capture{capture_t::byReference},
          body{new blockStatement(nullptr,source)}
    {
    }

    lambda_t::lambda_t(capture_t capture_)
        : function_t(),
          capture{capture_},
          body{new blockStatement(nullptr,source)}
    {
    }

lambda_t::lambda_t(capture_t capture_,const blockStatement& body_)
        : function_t(),
          capture{capture_},
          body{new blockStatement(nullptr,body_)}
    {
    }

    lambda_t::lambda_t(const lambda_t &other)
        : function_t(other),
          capture{other.capture},
          body{new blockStatement(nullptr,*other.body)}
    {
    }

    lambda_t::~lambda_t()
    {
      delete body;
    }

    int lambda_t::type() const
    {
      return typeType::lambda;
    }

    type_t &lambda_t::clone() const
    {
      return *(new lambda_t(*this));
    }

    bool lambda_t::isNamed() const
    {
      return false;
    }

    dtype_t lambda_t::dtype() const
    {
      return dtype::byte;
    }

    bool lambda_t::equals(const type_t &other) const
    {
      const lambda_t &other_ = other.to<lambda_t>();

      if (capture != other_.capture)
      {
        return false;
      }
      if (body != other_.body)
      {
        return false;
      }
      return function_t::equals(other);
    }

    void lambda_t::debugPrint() const
    {
      printer pout(io::stderr);
      printDeclaration(pout);
    }

    void lambda_t::printDeclaration(printer &pout) const
    {
      pout << "[";
      switch (this->capture)
      {
      case capture_t::byValue:
        pout << "=";
        break;
      case capture_t::byReference:
        pout << "&";
        break;
      default:
        pout << "???";
        break;
      }
      pout << "](";

      const std::string argIndent = pout.indentFromNewline();
      const int argCount = (int)args.size();
      for (int i = 0; i < argCount; ++i)
      {
        if (i)
        {
          pout << ",\n"
               << argIndent;
        }
        args[i]->printDeclaration(pout);
      }
      pout << ") {";

      pout.printNewline();
      pout.pushInlined(false);
      pout.addIndentation();

      body->print(pout);

      pout.removeIndentation();
      pout.popInlined();
      pout.printNewline();
      pout.printIndentation();
      pout << "}\n";
    }
  } // namespace lang
} // namespace occa
