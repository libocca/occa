#include <sstream>

#include <occa/utils/exception.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  exception::exception(const std::string &header_,
                       const std::string &filename_,
                       const std::string &function_,
                       const int line_,
                       const std::string &message_) :
    header(header_),
    filename(filename_),
    function(function_),
    message(message_),
    line(line_),
    exceptionMessage(toString()) {}

  exception::~exception() throw() {}

  const char* exception::what() const throw() {
    return exceptionMessage.c_str();
  }

  std::string exception::toString(const int stackTraceStart) const {
    std::stringstream ss;

    // Pad message lines
    strVector lines = split(message, '\n');
    const int lineCount = (int) lines.size();

    for (int i = 0; i < lineCount; ++i) {
      if (i) {
        ss << "\n               ";
      }
      ss << lines[i];
    }
    const std::string prettyMessage = ss.str();
    ss.str("");

    std::string banner = "---[ " + header + " ]";
    ss << '\n'
       << banner << std::string(80 - banner.size(), '-') << '\n'
       << location()
       << "    Message  : " << prettyMessage << '\n'
       << "    Stack\n"
       << sys::stacktrace(stackTraceStart, "      ")
       << std::string(80, '=') << '\n';

    return ss.str();
  }

  std::string exception::location() const {
    std::stringstream ss;
    ss << "    File     : " << filename << '\n'
       << "    Line     : " << line     << '\n'
       << "    Function : " << function << '\n';
    return ss.str();
  }

  std::ostream& operator << (std::ostream& out,
                             const exception &exc) {
    out << exc.toString() << std::flush;
    return out;
  }
}
