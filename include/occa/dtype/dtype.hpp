#ifndef OCCA_DTYPE_DTYPE_HEADER
#define OCCA_DTYPE_DTYPE_HEADER

#include <iostream>
#include <map>
#include <vector>

#include <occa/types/typedefs.hpp>


namespace occa {
  class dtype_t;
  class dtypeTuple_t;
  class dtypeStruct_t;
  class json;

  typedef std::map<std::string, const dtype_t*> dtypeGlobalMap_t;
  typedef std::map<std::string, dtype_t>        dtypeNameMap_t;
  typedef std::vector<const dtype_t*>           dtypeVector_t;

  class dtype_t {
  private:
    const dtype_t *ref;

    std::string name_;
    int bytes_;
    bool registered;

    dtypeTuple_t *tuple_;
    dtypeStruct_t *struct_;
    mutable dtypeVector_t flatDtype;

  public:
    dtype_t();

    dtype_t(const std::string &name__,
            const int bytes__ = 0,
            const bool registered_ = false);

    dtype_t(const std::string &name__,
            const dtype_t &other,
            const bool registered_ = false);

    dtype_t(const dtype_t &other);

    dtype_t& operator = (const dtype_t &other_);

    ~dtype_t();

    inline const dtype_t& self() const {
      return ref ? *ref : *this;
    }

    const std::string& name() const;
    int bytes() const;

    void registerType();
    bool isRegistered() const;

    // Tuple methods
    bool isTuple() const;
    int tupleSize() const;

    // Struct methods
    bool isStruct() const;
    int structFieldCount() const;
    const strVector& structFields() const;

    const dtype_t& operator [] (const int field) const;
    const dtype_t& operator [] (const std::string &field) const;

    dtype_t& addField(const std::string &field,
                      const dtype_t &dtype,
                      const int tupleSize_ = 1);

    // Dtype methods
    void setFlattenedDtype() const;
    void addFlatDtypes(dtypeVector_t &vec) const;

    bool operator == (const dtype_t &other) const;
    bool operator != (const dtype_t &other) const;
    const dtype_t& operator || (const dtype_t &other) const;

    bool matches(const dtype_t &other) const;

    bool canBeCastedTo(const dtype_t &other) const;

    static bool isCyclic(const dtypeVector_t &vec,
                         const int cycleLength);

    static dtype_t tuple(const dtype_t &dtype,
                         const int size,
                         const bool registered_ = false);

    static const dtype_t& getBuiltin(const std::string &name);

    json toJson(const std::string &name = "") const;
    void toJson(json &j, const std::string &name = "") const;

    static dtype_t fromJson(const std::string &str);
    static dtype_t fromJson(const json &j);

    std::string toString(const std::string &varName = "") const;

    friend std::ostream& operator << (std::ostream &out,
                                    const dtype_t &dtype);
  };

  std::ostream& operator << (std::ostream &out,
                           const dtype_t &dtype);


  //---[ Tuple ]------------------------
  class dtypeTuple_t {
    friend class dtype_t;

  private:
    const dtype_t dtype;
    int size;

    dtypeTuple_t(const dtype_t &dtype_,
                 const int size_);

    dtypeTuple_t* clone() const;

    bool matches(const dtypeTuple_t &other) const;

    void addFlatDtypes(dtypeVector_t &vec) const;

    void toJson(json &j, const std::string &name = "") const;
    static dtypeTuple_t fromJson(const json &j);

    std::string toString(const std::string &varName = "") const;
  };
  //====================================


  //---[ Struct ]-----------------------
  class dtypeStruct_t {
    friend class dtype_t;

  private:
    strVector fieldNames;
    dtypeNameMap_t fieldTypes;

    dtypeStruct_t();

    dtypeStruct_t* clone() const;

    bool matches(const dtypeStruct_t &other) const;

    int fieldCount() const;

    const dtype_t& operator [] (const int field) const;
    const dtype_t& operator [] (const std::string &field) const;

    void addField(const std::string &field,
                  const dtype_t &dtype);

    void addFlatDtypes(dtypeVector_t &vec) const;

    void toJson(json &j, const std::string &name = "") const;
    static dtypeStruct_t fromJson(const json &j);

    std::string toString(const std::string &varName = "") const;
  };
  //====================================
}

#endif
