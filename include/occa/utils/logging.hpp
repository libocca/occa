#ifndef OCCA_UTILS_LOGGING_HEADER
#define OCCA_UTILS_LOGGING_HEADER

#include <iostream>
#include <sstream>

namespace occa {
  namespace io {
    class output;
  }

  void _message(const std::string &header,
                const bool exitInFailure,
                const std::string &filename,
                const std::string &function,
                const int line,
                const std::string &message);

  void warn(const std::string &filename,
            const std::string &function,
            const int line,
            const std::string &message);

  void error(const std::string &filename,
             const std::string &function,
             const int line,
             const std::string &message);

  void printWarning(io::output &out,
                    const std::string &message,
                    const std::string &code = "");

  void printWarning(const std::string &message,
                    const std::string &code = "");

  void printError(io::output &out,
                  const std::string &message,
                  const std::string &code = "");

  void printError(const std::string &message,
                  const std::string &code = "");
}

#endif
