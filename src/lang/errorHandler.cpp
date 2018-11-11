#include <occa/tools/sys.hpp>

#include <occa/lang/errorHandler.hpp>

namespace occa {
  namespace lang {
    errorHandler::errorHandler() :
      warnings(0),
      errors(0) {}

    void errorHandler::preprint(std::ostream &out) {}
    void errorHandler::postprint(std::ostream &out) {}

    void errorHandler::printNote(std::ostream &out,
                                 const std::string &message) {
      preprint(out);
      occa::printNote(out, message);
      postprint(out);
    }

    void errorHandler::printWarning(std::ostream &out,
                                    const std::string &message) {
      ++warnings;
      preprint(out);
      occa::printWarning(out, message);
      postprint(out);
    }

    void errorHandler::printError(std::ostream &out,
                                  const std::string &message) {
      ++errors;
      preprint(out);
      occa::printError(out, message);
      postprint(out);
    }
  }
}
