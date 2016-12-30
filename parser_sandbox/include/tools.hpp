#ifndef OCCA_PARSER_TOOLS_HEADER2
#define OCCA_PARSER_TOOLS_HEADER2

#include <iostream>

namespace occa {
  void skipTo(const char *&c, const char delimiter);
  void skipTo(const char *&c, const char delimiter, const char escapeChar);

  void skipTo(const char *&c, const std::string &match);
  void skipTo(const char *&c, const std::string &match, const char escapeChar);

  void skipToDelimiter(const char *&c, const std::string &delimiters);
  void skipToDelimiter(const char *&c, const std::string &delimiters, const char escapeChar);
}
#endif
