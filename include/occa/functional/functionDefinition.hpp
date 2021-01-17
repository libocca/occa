#ifndef OCCA_FUNCTIONAL_FUNCTIONDEFINITION_HEADER
#define OCCA_FUNCTIONAL_FUNCTIONDEFINITION_HEADER

#include <memory>
#include <string>

#include <occa/dtype.hpp>
#include <occa/utils/hash.hpp>
#include <occa/functional/scope.hpp>

namespace occa {
  class functionDefinition;

  typedef std::shared_ptr<functionDefinition> functionDefinitionSharedPtr;

  class functionDefinition {
  public:
    occa::scope scope;
    std::string source;
    dtype_t returnType;
    dtypeVector argTypes;

    hash_t hash;
    std::string argumentSource;
    std::string bodySource;

    functionDefinition();

    int functionArgumentCount() const;
    int totalArgumentCount() const;

    std::string getFunctionSource(const std::string &functionName);

    static hash_t getHash(const occa::scope &scope,
                          const std::string &source,
                          const dtype_t &returnType);

    static functionDefinitionSharedPtr cache(
      const occa::scope &scope,
      const std::string &source,
      const dtype_t &returnType,
      const dtypeVector &argTypes
    );

    static void skipLambdaCapture(const char *&c);
    static std::string getArgumentSource(const std::string &source,
                                         const occa::scope &scope);
    static std::string getBodySource(const std::string &source);
  };
}

#endif
