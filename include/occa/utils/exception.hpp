#ifndef OCCA_UTILS_EXCEPTION_HEADER
#define OCCA_UTILS_EXCEPTION_HEADER

#include <stdexcept>
#include <ostream>

namespace occa {
  class exception : public std::exception {
   public:
    const std::string header;
    const std::string filename;
    const std::string function;
    const std::string message;
    const int line;

    std::string exceptionMessage;

    exception(const std::string &header_,
              const std::string &filename_,
              const std::string &function_,
              const int line_,
              const std::string &message_ = "");

    ~exception() throw();

    const char* what() const throw();

    std::string toString(const int stackTraceStart = 4) const;
    std::string location() const;
  };

  std::ostream& operator << (std::ostream& out,
                             const exception &exc);
}

#endif
