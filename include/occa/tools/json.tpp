namespace occa {
  template <class TM>
  json& json::set(const char *key,
                  const TM &value) {
    type = object_;
    value_.object[key] = value;
    return *this;
  }

  template <class TM>
  json& json::set(const std::string &key,
                  const TM &value) {
    return set(key.c_str(), value);
  }

  template <class TM>
  TM json::get(const char *key,
               const TM &default_) const {
    const json *j = this;
    const char *c = key;
    while (*c != '\0') {
      if (j->type != object_) {
        return default_;
      }

      const char *cStart = c;
      lex::skipTo(c, '/');
      std::string nextKey(cStart, c - cStart);
      if (*c == '/') {
        ++c;
      }

      jsonObject::const_iterator it = j->value_.object.find(nextKey);
      if (it == j->value_.object.end()) {
        return default_;
      }
      j = &(it->second);
    }
    return *j;
  }

  template <class TM>
  TM json::get(const std::string &key,
               const TM &default_) const {
    return get<TM>(key.c_str(), default_);
  }

  template <class TM>
  std::vector<TM> json::getArray(const std::vector<TM> &default_) const {
    std::string empty;
    return getArray(empty.c_str(), default_);
  }

  template <class TM>
  std::vector<TM> json::getArray(const char *c,
                                 const std::vector<TM> &default_) const {
    const json *j = this;
    while (*c) {
      if (j->type != object_) {
        return default_;
      }

      const char *cStart = c;
      lex::skipTo(c, '/');
      std::string key(cStart, c - cStart);
      if (*c == '/') {
        ++c;
      }

      jsonObject::const_iterator it = j->value_.object.find(key);
      if (it == j->value_.object.end()) {
        return default_;
      }
      j = &(it->second);
    }
    if (j->type != array_) {
      return default_;
    }

    const int entries = (int) j->value_.array.size();
    std::vector<TM> ret;
    for (int i = 0; i < entries; ++i) {
      ret.push_back((TM) j->value_.array[i]);
    }
    return ret;
  }

  template <class TM>
  std::vector<TM> json::getArray(const std::string &s,
                                 const std::vector<TM> &default_) const {
    return get<TM>(s.c_str(), default_);
  }
}
