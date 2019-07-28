#ifndef OCCA_LANG_KERNELMETADATA_HEADER
#define OCCA_LANG_KERNELMETADATA_HEADER

#include <occa/tools/json.hpp>
#include <occa/dtype.hpp>

namespace occa {
  namespace lang {
    class kernelMetadata;

    typedef std::map<std::string, kernelMetadata> kernelMetadataMap;

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

    class kernelMetadata {
    public:
      bool initialized;
      std::string name;
      std::vector<argMetadata_t> arguments;

      kernelMetadata();

      bool isInitialized() const;

      kernelMetadata& operator += (const argMetadata_t &argInfo);

      static kernelMetadata fromJson(const json &j);
      json toJson() const;
    };

    kernelMetadataMap getBuildFileMetadata(const std::string &filename);
  }
}

#endif
