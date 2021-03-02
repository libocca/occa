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

  /**
   * @startDoc{dtype_t}
   *
   * Description:
   *   Represents a data type, such as:
   *   - `occa::dtype::void_` &rarr; `void`
   *   - `occa::dtype::float_` &rarr; `float`
   *   - `occa::dtype::byte` &rarr; A wildcard type, matching anything
   *
   *   [[dtype_t]] data types are used to hold type information on many things, such as
   *   [[memory]] object types or [[kernel]] argument types.
   *
   * @endDoc
   */
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

  /**
   * @startDoc{name}
   *
   * Description:
   *   Returns the name of the data type
   *
   * @endDoc
   */
    const std::string& name() const;

  /**
   * @startDoc{bytes}
   *
   * Description:
   *   Return the `sizeof` types the underyling type
   *
   * @endDoc
   */
    int bytes() const;

  /**
   * @startDoc{registerType}
   *
   * Description:
   *   Register the data type so it can be used in [[kernels|kernel]].
   *
   *   It states this new type is a base type and treated like a singleton.
   *
   * @endDoc
   */
    void registerType();

    bool isRegistered() const;

    // Tuple methods
    /**
     * @startDoc{isTuple}
     *
     * Description:
     *   Returns `true` if the data type holds a tuple type.
     *   For example: `occa::dtype::int2` is a tuple of two `int`s
     *
     * @endDoc
     */
    bool isTuple() const;

    /**
     * @startDoc{tupleSize}
     *
     * Description:
     *   Return how big the tuple is, for example `int2` would return `2`
     *
     * @endDoc
     */
    int tupleSize() const;

    // Struct methods
    /**
     * @startDoc{isStruct}
     *
     * Description:
     *   Returns `true` if the data type represents a struct.
     *   It's different that a tuple since it can keep distinct data types in its fields.
     *
     * @endDoc
     */
    bool isStruct() const;

    /**
     * @startDoc{structFieldCount}
     *
     * Description:
     *   Returns how many fields are defined in the struct
     *
     * @endDoc
     */
    int structFieldCount() const;

    /**
     * @startDoc{structFieldNames}
     *
     * Description:
     *   Return the list of field names for the struct
     *
     * @endDoc
     */
    const strVector& structFieldNames() const;

    /**
     * @startDoc{operator_bracket[0]}
     *
     * Description:
     *   Return the [[dtype_t]] for the field by the given index
     *
     * @endDoc
     */
    const dtype_t& operator [] (const int field) const;

    /**
     * @startDoc{operator_bracket[1]}
     *
     * Description:
     *   Same as above but for the field name rather than index
     *
     * @endDoc
     */
    const dtype_t& operator [] (const std::string &field) const;

    /**
     * @startDoc{addField}
     *
     * Description:
     *   Add a field to the struct type
     *
     * @endDoc
     */
    dtype_t& addField(const std::string &field,
                      const dtype_t &dtype,
                      const int tupleSize_ = 1);

    // Dtype methods
    void setFlattenedDtype() const;
    void addFlatDtypes(dtypeVector_t &vec) const;

    /**
     * @startDoc{operator_equals}
     *
     * Description:
     *   Compare if two data types have the same reference
     *
     * @endDoc
     */
    bool operator == (const dtype_t &other) const;

    /**
     * @startDoc{operator_equals}
     *
     * Description:
     *   Compare if two data types have different references
     *
     * @endDoc
     */
    bool operator != (const dtype_t &other) const;

    const dtype_t& operator || (const dtype_t &other) const;

    /**
     * @startDoc{matches}
     *
     * Description:
     *   Check whether two types are equivalent, even if their references don't match
     *
     * @endDoc
     */
    bool matches(const dtype_t &other) const;

    /**
     * @startDoc{canBeCastedTo}
     *
     * Description:
     *   Check whether flattened, two types can be matched.
     *   For example:
     *
     *   - `int` can be casted to `int2` and vice-versa.
     *   - A struct of `[int, float, int, float]` fields can be casted to a struct of `[int, float]` fields.
     *
     * @endDoc
     */
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
