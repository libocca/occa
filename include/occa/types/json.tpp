namespace occa {
  template <class T>
  json& json::set(const char *key,
                  const T &value) {
    type = object_;
    value_.object[key] = value;
    return *this;
  }

  template <class T>
  json& json::set(const std::string &key,
                  const T &value) {
    return set(key.c_str(), value);
  }

  template <class T>
  T json::get(const char *key,
               const T &default_) const {
    const json value = getPathValue(key);
    if (value.isInitialized()) {
      return (T) value;
    }
    return default_;
  }

  template <class T>
  T json::get(const std::string &key,
               const T &default_) const {
    return get<T>(key.c_str(), default_);
  }

  template <class T>
  std::vector<T> json::toVector(const std::vector<T> &default_) const {
    if (!isArray()) {
      return default_;
    }

    std::vector<T> ret;
    for (const json &entry : array()) {
      ret.push_back((T) entry);
    }
    return ret;
  }

  template <class T>
  std::vector<T> json::toVector(const char *c,
                                 const std::vector<T> &default_) const {
    return getPathValue(c).toVector<T>(default_);
  }

  template <class T>
  std::vector<T> json::toVector(const std::string &s,
                                 const std::vector<T> &default_) const {
    return getPathValue(s.c_str()).toVector<T>(default_);
  }
}
