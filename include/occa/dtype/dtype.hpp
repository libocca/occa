#ifndef OCCA_DTYPE_DTYPE_HEADER
#define OCCA_DTYPE_DTYPE_HEADER

#include <iostream>
#include <map>
#include <vector>

#include <occa/tools/json.hpp>


namespace occa {
  class dtype;
  class dtypeField;

  typedef std::map<std::string, dtype*> dtypeNameMap_t;

  class dtype {
  private:
    std::string name;
    int bytes;
    std::vector<dtypeField> fields;

  public:
    dtype(const std::string &name_);
    dtype(const std::string &name_,
          const int bytes_);
    dtype(const dtype &other);

    const std::string& getName() const;
    int getBytes() const;

    dtype& addField(const std::string field,
                    const dtype &type);

    bool operator == (const dtype &other) const;
    bool operator != (const dtype &other) const;

    static const dtype& byName(const std::string name);

    static dtype fromJson(const std::string &str);
    static dtype fromJson(const json &j);

    json toJson() const;

    friend std::ostream& operator << (std::ostream &out,
                                      const dtype &type);
  };

  std::ostream& operator << (std::ostream &out,
                             const dtype &type);
  class dtypeField {
    friend class dtype;

  private:
    std::string name;
    dtype type;

  public:
    dtypeField(const std::string &name_,
               const dtype &type_);
    dtypeField(const dtypeField &other);
  };
}

#endif
