#ifndef OCCA_LANG_ERRORHANDLER_HEADER
#define OCCA_LANG_ERRORHANDLER_HEADER

#include <occa/io/output.hpp>

namespace occa {
  namespace lang {
    class errorHandler {
    public:
      mutable int warnings, errors;

      errorHandler();

      virtual void preprint(io::output &out);
      virtual void postprint(io::output &out);

      void printNote(io::output &out,
                     const std::string &message);

      inline void printNote(const std::string &message){
        printNote(io::stderr, message);
      }

      void printWarning(io::output &out,
                        const std::string &message);

      inline void printWarning(const std::string &message) {
        printWarning(io::stderr, message);
      }

      void printError(io::output &out,
                      const std::string &message);

      inline void printError(const std::string &message) {
        printError(io::stderr, message);
      }
    };
  }
}

#endif
