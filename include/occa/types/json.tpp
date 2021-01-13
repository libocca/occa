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
    const json value = getPathValue(key);
    if (value.isInitialized()) {
      return (TM) value;
    }
    return default_;
  }

  template <class TM>
  TM json::get(const std::string &key,
               const TM &default_) const {
    return get<TM>(key.c_str(), default_);
  }

  template <class TM>
  std::vector<TM> json::toVector(const std::vector<TM> &default_) const {
    if (!isArray()) {
      return default_;
    }

    std::vector<TM> ret;
    for (const json &entry : array()) {
      ret.push_back((TM) entry);
    }
    return ret;
  }

  template <class TM>
  std::vector<TM> json::toVector(const char *c,
                                 const std::vector<TM> &default_) const {
    return getPathValue(c).toVector<TM>(default_);
  }

  template <class TM>
  std::vector<TM> json::toVector(const std::string &s,
                                 const std::vector<TM> &default_) const {
    return getPathValue(s.c_str()).toVector<TM>(default_);
  }
}
