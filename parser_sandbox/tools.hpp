class stringChunk {
  const char *c;
  const size_t chars;
  std::string str;

  stringChunk(const char *c_, const size_t chars_) :
    c(c_),
    chars(chars_) {}

  stringChunk(const std::string &str_) :
    c(NULL),
    chars(0),
    str(str_) {}
}

void skipTo(const char *&c, const char delimiter) {
  while (*c != '\0') {
    if(*c == delimiter) {
      return;
    }
    ++c;
  }
}

void skipTo(const char *&c, const char delimiter, const char escapeChar) {
  while (*c != '\0') {
    if (*c == escapeChar) {
      c += 2;
      continue;
    }
    if(*c == delimiter) {
      return;
    }
    ++c;
  }
}

void skipTo(const char *&c, const std::string &match) {
  const size_t chars = match.size();
  const char *d      = match.c_str();

  while (*c != '\0') {
    for (size_t i = 0; i < chars; ++i) {
      if (c[i] != d[i]) {
        continue;
      }
      return;
    }
    ++c;
  }
}

void skipTo(const char *&c, const std::string &match, const char escapeChar) {
  const size_t chars = match.size();
  const char *d      = match.c_str();

  while (*c != '\0') {
    if (*c == escapeChar) {
      c += 2;
      continue;
    }
    for (size_t i = 0; i < chars; ++i) {
      if (c[i] != d[i]) {
        continue;
      }
      return;
    }
    ++c;
  }
}

void skipToDelimiter(const char *&c, const std::string &delimiters) {
  const size_t chars = delimiters.size();
  const char *d      = delimiters.c_str();

  while (*c != '\0') {
    for (size_t i = 0; i < chars; ++i) {
      if (*c == d[i]) {
        return;
      }
    }
    ++c;
  }
}

void skipToDelimiter(const char *&c, const std::string &delimiters, const char escapeChar) {
  const size_t chars = delimiters.size();
  const char *d      = delimiters.c_str();

  while (*c != '\0') {
    if (*c == escapeChar) {
      c += 2;
      continue;
    }
    for (size_t i = 0; i < chars; ++i) {
      if (*c == d[i]) {
        return;
      }
    }
    ++c;
  }
}
