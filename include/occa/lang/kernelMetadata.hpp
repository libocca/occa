#ifndef OCCA_LANG_KERNELMETADATA_HEADER
#define OCCA_LANG_KERNELMETADATA_HEADER

#include <occa/tools/json.hpp>
#include <occa/dtype.hpp>

namespace occa {
  namespace lang {
    class kernelMetadata;

    typedef std::map<std::string, kernelMetadata> kernelMetadataMap;

    class argumentInfo {
    public:
      bool isConst;
      bool isPtr;
      dtype_t dtype;

      argumentInfo();

      argumentInfo(const bool isConst_,
                   const bool isPtr_,
                   const dtype_t &dtype_);

      static argumentInfo fromJson(const json &j);
      json toJson() const;
    };

    class kernelMetadata {
    public:
      bool initialized;
      std::string name;
      std::vector<argumentInfo> arguments;

      kernelMetadata();

      bool isInitialized() const;

      kernelMetadata& operator += (const argumentInfo &argInfo);

      static kernelMetadata fromJson(const json &j);
      json toJson() const;
    };

    kernelMetadataMap getBuildFileMetadata(const std::string &filename);
  }
}

#endif
