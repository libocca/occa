#include <occa/utils/logging.hpp>

#include <occa/utils/exception.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  void _message(const std::string &header,
                const bool exitInFailure,
                const std::string &filename,
                const std::string &function,
                const int line,
                const std::string &message) {

    exception exp(header,
                  filename,
                  function,
                  line,
                  message);

    if (exitInFailure) {
      throw exp;
    }
    io::stderr << exp;
  }

  void warn(const std::string &filename,
            const std::string &function,
            const int line,
            const std::string &message) {
    _message("Warning", false,
             filename, function, line, message);
  }

  void error(const std::string &filename,
             const std::string &function,
             const int line,
             const std::string &message) {
    _message("Error", true,
             filename, function, line, message);
  }

  void printWarning(io::output &out,
                    const std::string &message,
                    const std::string &code) {
    if (env::OCCA_VERBOSE) {
      if (code.size()) {
        out << yellow("Warning " + code);
      } else {
        out << yellow("Warning");
      }
      out << ": " << message << '\n';
    }
  }

  void printWarning(const std::string &message,
                    const std::string &code) {
    printWarning(io::stderr, message, code);
  }

  void printError(io::output &out,
                  const std::string &message,
                  const std::string &code) {
    if (code.size()) {
      out << red("Error " + code);
    } else {
      out << red("Error");
    }
    out << ": " << message << '\n';
  }

  void printError(const std::string &message,
                  const std::string &code) {
    printError(io::stderr, message, code);
  }
}
