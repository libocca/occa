#ifndef OCCA_INTERNAL_LANG_KERNELMETADATA_HEADER
#define OCCA_INTERNAL_LANG_KERNELMETADATA_HEADER

#include <occa/utils/hash.hpp>
#include <occa/types/json.hpp>
#include <occa/dtype.hpp>

namespace occa {
  namespace lang {
    class kernelMetadata_t;

    typedef std::map<std::string, kernelMetadata_t> kernelMetadataMap;
    typedef std::map<std::string, hash_t> strHashMap;

    class argMetadata_t {
    public:
      bool isConst;
      bool isPtr;
      dtype_t dtype;
      std::string name;

      argMetadata_t();

      argMetadata_t(const bool isConst_,
                   const bool isPtr_,
                   const dtype_t &dtype_,
                   const std::string &name_);

      static argMetadata_t fromJson(const json &j);
      json toJson() const;
    };

    class kernelMetadata_t {
    public:
      bool initialized;
      std::string name;
      std::vector<argMetadata_t> arguments;

      kernelMetadata_t();

      bool isInitialized() const;

      kernelMetadata_t& operator += (const argMetadata_t &argInfo);

      static kernelMetadata_t fromJson(const json &j);
      json toJson() const;
    };

    class sourceMetadata_t {
     public:
      kernelMetadataMap kernelsMetadata;
      strHashMap dependencyHashes;

      sourceMetadata_t();

      json getKernelMetadataJson() const;
      json getDependencyJson() const;

      static sourceMetadata_t fromBuildFile(const std::string &filename);
    };
  }
}

#endif
