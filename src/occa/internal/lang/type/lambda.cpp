#include <occa/internal/lang/type/lambda.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa
{
  namespace lang
  {
    lambda_t::lambda_t()
        : type_t(),
          capture{capture_t::byReference},
          body{nullptr}
    {
    }

    lambda_t::lambda_t(capture_t capture_, const blockStatement &body_)
        : type_t(),
          capture{capture_},
          body(new blockStatement(body_.up, body_))
    {
    }

    lambda_t::lambda_t(const lambda_t &other)
        : type_t(other),
          capture{other.capture},
          body{nullptr}
    {
      const int count = (int)other.args.size();
      for (int i = 0; i < count; ++i)
      {
        args.push_back(
            &(other.args[i]->clone()));
      }

      if (other.body)
      {
        body = new blockStatement(other.body->up, *other.body);
      }
    }

    lambda_t::~lambda_t()
    {
      free();
    }

    void lambda_t::free()
    {
      const int count = (int)args.size();
      for (int i = 0; i < count; ++i)
      {
        delete args[i];
      }
      args.clear();

      // if (body)
      // {
      delete body;
      // body = nullptr;
      // }
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

    void lambda_t::addArgument(const variable_t &arg)
    {
      args.push_back(&(arg.clone()));
    }

    void lambda_t::addArguments(const variableVector &args_)
    {
      const int count = (int)args_.size();
      for (int i = 0; i < count; ++i)
      {
        args.push_back(&(args_[i].clone()));
      }
    }

    variable_t *lambda_t::removeArgument(const int index)
    {
      const int argCount = (int)args.size();
      if (index < 0 || argCount <= index)
      {
        return NULL;
      }
      variable_t *arg = args[index];
      args.erase(args.begin() + index);
      return arg;
    }

    bool lambda_t::equals(const type_t &other) const
    {
      const lambda_t &other_ = other.to<lambda_t>();

      const int argSize = (int)args.size();
      if (argSize != (int)other_.args.size())
      {
        return false;
      }
      if (capture != other_.capture)
      {
        return false;
      }
      if (body != other_.body)
      {
        return false;
      }

      for (int i = 0; i < argSize; ++i)
      {
        if (args[i]->vartype != other_.args[i]->vartype)
        {
          return false;
        }
      }
      return true;
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

      // pout.printStartIndentation();
      pout.pushInlined(false);
      // pout.addIndentation();

      const int argCount = (int)args.size();
      for (int i = 0; i < argCount; ++i)
      {
        if(i > 0) 
        {
        pout << ',';
        pout.printNewline();
        }
        args[i]->printDeclaration(pout);
      }

     
      pout.popInlined();
      // pout.printNewline();

      pout << ')';

      if (body)
      {
        body->print(pout);
      }
      else
      {
        pout << "{ }";
      }
    }
  } // namespace lang
} // namespace occa
