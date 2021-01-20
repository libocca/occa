#include <occa/functional/functionDefinition.hpp>
#include <occa/internal/functional/functionStore.hpp>
#include <occa/internal/utils/lex.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  functionDefinition::functionDefinition() {}

  int functionDefinition::functionArgumentCount() const {
    return (int) argTypes.size();
  }

  int functionDefinition::totalArgumentCount() const {
    return (int) (argTypes.size() + scope.args.size());
  }

  std::string functionDefinition::getFunctionSource(const std::string &functionName) {
    std::stringstream ss;

    const dtype_t &safeReturnType = returnType || dtype::void_;

    ss << safeReturnType.name() << ' ' << functionName << '('
       << argumentSource;

    // Add captured variables at the end
    if (scope.args.size()) {
      if (argTypes.size()) {
        ss << ", ";
      }
      ss << scope.getDeclarationSource();
    }

    ss << ") {\n"
       << bodySource << '\n'
       << "}\n";

    return ss.str();
  }

  hash_t functionDefinition::getHash(const occa::scope &scope_,
                                     const std::string &source_,
                                     const dtype_t &returnType_) {
    hash_t hash_ = (
      occa::hash(scope_)
      ^ occa::hash(source_)
      ^ occa::hash(returnType_.name())
    );
    return hash_;
  }

  functionDefinitionSharedPtr functionDefinition::cache(
    const occa::scope &scope_,
    const std::string &source_,
    const dtype_t &returnType_,
    const dtypeVector &argTypes_
  ) {
    hash_t hash_ = getHash(
      scope_,
      source_,
      returnType_
    );

    functionStore.lock(hash_);

    std::shared_ptr<functionDefinition> fnDefPtr;
    const bool createdPtr = functionStore.unsafeGetOrCreate(hash_, fnDefPtr);

    // Create the function definition
    if (createdPtr) {
      functionDefinition &fnDef = *(fnDefPtr.get());

      // Constructor args
      fnDef.scope      = scope_;
      fnDef.source     = source_;
      fnDef.returnType = returnType_;
      fnDef.argTypes   = argTypes_;

      // Generated args
      fnDef.hash           = hash_;
      fnDef.argumentSource = getArgumentSource(source_, scope_);
      fnDef.bodySource     = getBodySource(source_);
    }

    functionStore.unlock(hash_);

    return fnDefPtr;
  }

  void functionDefinition::skipLambdaCapture(const char *&c) {
    occa::lex::skipTo(c, '[');
    occa::lex::skipTo(c, ']');
  }

  std::string functionDefinition::getArgumentSource(const std::string &source_,
                                                    const occa::scope &scope_) {
    const char *root = source_.c_str();
    const char *start = root;
    const char *end = root;

    skipLambdaCapture(start);
    occa::lex::skipTo(start, '(');
    ++start;

    end = start;
    occa::lex::skipTo(end, ')');

    return strip(std::string(start, end - start));
  }

  std::string functionDefinition::getBodySource(const std::string &source_) {
    const char *root = source_.c_str();
    const char *start = root;
    const char *end = root + source_.size();

    skipLambdaCapture(start);
    occa::lex::skipTo(start, '{');
    ++start;

    for (; start <= end; --end) {
      if (*end == '}') {
        break;
      }
    }

    return strip(std::string(start, end - start));
  }
}
