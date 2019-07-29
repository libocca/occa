#ifndef OCCA_LANG_KERNELMETADATA_HEADER
#define OCCA_LANG_KERNELMETADATA_HEADER

#include <occa/tools/json.hpp>
#include <occa/dtype.hpp>

namespace occa {
  namespace lang {
    class kernelMetadata_t;

    typedef std::map<std::string, kernelMetadata_t> kernelMetadataMap;

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

    kernelMetadataMap getBuildFileMetadata(const std::string &filename);
  }
}

#endif
