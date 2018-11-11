#ifndef OCCA_LANG_ERRORHANDLER_HEADER
#define OCCA_LANG_ERRORHANDLER_HEADER

namespace occa {
  namespace lang {
    class errorHandler {
    public:
      mutable int warnings, errors;

      errorHandler();

      virtual void preprint(std::ostream &out);
      virtual void postprint(std::ostream &out);

      void printNote(std::ostream &out,
                     const std::string &message);

      inline void printNote(const std::string &message){
        printNote(std::cerr, message);
      }

      void printWarning(std::ostream &out,
                        const std::string &message);

      inline void printWarning(const std::string &message) {
        printWarning(std::cerr, message);
      }

      void printError(std::ostream &out,
                      const std::string &message);

      inline void printError(const std::string &message) {
        printError(std::cerr, message);
      }
    };
  }
}

#endif
