#ifndef OCCA_LANG_KERNELMETADATA_HEADER
#define OCCA_LANG_KERNELMETADATA_HEADER

#include <occa/tools/json.hpp>

namespace occa {
  namespace lang {
    class kernelMetadata;

    typedef std::map<std::string, kernelMetadata> kernelMetadataMap;

    class argumentInfo {
    public:
      bool isConst;

      argumentInfo(const bool isConst_ = false);

      static argumentInfo fromJson(const json &j);
      json toJson() const;
    };

    class kernelMetadata {
    public:
      std::string name;
      std::vector<argumentInfo> arguments;

      kernelMetadata();

      kernelMetadata& operator += (const argumentInfo &argInfo);

      bool argIsConst(const int pos) const;

      static kernelMetadata fromJson(const json &j);
      json toJson() const;
    };

    kernelMetadataMap getBuildFileMetadata(const std::string &filename);
  }
}

#endif
