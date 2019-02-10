#include <occa/tools/sys.hpp>

#include <occa/lang/errorHandler.hpp>

namespace occa {
  namespace lang {
    errorHandler::errorHandler() :
      warnings(0),
      errors(0) {}

    void errorHandler::preprint(io::output &out) {}
    void errorHandler::postprint(io::output &out) {}

    void errorHandler::printNote(io::output &out,
                                 const std::string &message) {
      preprint(out);
      occa::printNote(out, message);
      postprint(out);
    }

    void errorHandler::printWarning(io::output &out,
                                    const std::string &message) {
      ++warnings;
      preprint(out);
      occa::printWarning(out, message);
      postprint(out);
    }

    void errorHandler::printError(io::output &out,
                                  const std::string &message) {
      ++errors;
      preprint(out);
      occa::printError(out, message);
      postprint(out);
    }
  }
}
