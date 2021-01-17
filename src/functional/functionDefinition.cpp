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

  hash_t functionDefinition::getHash(const occa::scope &scope,
                                     const std::string &source,
                                     const dtype_t &returnType) {
    hash_t hash = (
      occa::hash(scope)
      ^ occa::hash(source)
      ^ occa::hash(returnType.name())
    );
    return hash;
  }

  functionDefinitionSharedPtr functionDefinition::cache(
    const occa::scope &scope,
    const std::string &source,
    const dtype_t &returnType,
    const dtypeVector &argTypes
  ) {
    hash_t hash = getHash(
      scope,
      source,
      returnType
    );

    functionStore.lock(hash);

    std::shared_ptr<functionDefinition> fnDefPtr;
    const bool createdPtr = functionStore.unsafeGetOrCreate(hash, fnDefPtr);

    // Create the function definition
    if (createdPtr) {
      functionDefinition &fnDef = *(fnDefPtr.get());

      // Constructor args
      fnDef.scope      = scope;
      fnDef.source     = source;
      fnDef.returnType = returnType;
      fnDef.argTypes   = argTypes;

      // Generated args
      fnDef.hash           = hash;
      fnDef.argumentSource = getArgumentSource(source, scope);
      fnDef.bodySource     = getBodySource(source);
    }

    functionStore.unlock(hash);

    return fnDefPtr;
  }

  void functionDefinition::skipLambdaCapture(const char *&c) {
    occa::lex::skipTo(c, '[');
    occa::lex::skipTo(c, ']');
  }

  std::string functionDefinition::getArgumentSource(const std::string &source,
                                                    const occa::scope &scope) {
    const char *root = source.c_str();
    const char *start = root;
    const char *end = root;

    skipLambdaCapture(start);
    occa::lex::skipTo(start, '(');
    ++start;

    end = start;
    occa::lex::skipTo(end, ')');

    return strip(std::string(start, end - start));
  }

  std::string functionDefinition::getBodySource(const std::string &source) {
    const char *root = source.c_str();
    const char *start = root;
    const char *end = root + source.size();

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
