#include <sstream>
#include <ctime>

#include "occa/tools/string.hpp"

#include "preprocessor.hpp"
#include "specialMacros.hpp"

namespace occa {
  // __FILE__
  fileMacro_t::fileMacro_t(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_) {
    name = "__FILE__";
  }

  std::string fileMacro_t::expand(char *&c) const {
    return preprocessor->currentFrame.filename();
  }

  // __LINE__
  lineMacro_t::lineMacro_t(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_) {
    name = "__LINE__";
  }

  std::string lineMacro_t::expand(char *&c) const {
    return toString(preprocessor->currentFrame.lineNumber);
  }

  // __DATE__
  dateMacro_t::dateMacro_t(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_) {
    name = "__DATE__";
  }

  std::string dateMacro_t::expand(char *&c) const {
    static char month[12][5] = {
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    };

    time_t t = ::time(NULL);
    struct tm *ct = ::localtime(&t);

    if (ct == NULL) {
      return "??? ?? ????";
    }

    std::stringstream ss;
    ss << month[ct->tm_mon] << ' ';
    if (ct->tm_mday < 10) {
      ss << ' ';
    }
    ss << ct->tm_mday << ' '
       << ct->tm_year + 1900;
    return ss.str();
  }

  // __TIME__
  timeMacro_t::timeMacro_t(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_) {
    name = "__TIME__";
  }

  std::string timeMacro_t::expand(char *&c) const {
    time_t t = ::time(NULL);
    struct tm *ct = ::localtime(&t);

    if (ct == NULL) {
      return "??:??:??";
    }

    std::stringstream ss;
    if (ct->tm_hour < 10) {
      ss << '0';
    }
    ss << ct->tm_hour << ':';
    if (ct->tm_min < 10) {
      ss << '0';
    }
    ss << ct->tm_min << ':';
    if (ct->tm_sec < 10) {
      ss << '0';
    }
    ss << ct->tm_sec;
    return ss.str();
  }

  // __COUNTER__
  counterMacro_t::counterMacro_t(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_),
    counter(0) {
    name = "__COUNTER__";
  }

  std::string counterMacro_t::expand(char *&c) const {
    return toString(counter++);
  }
}
