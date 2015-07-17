#ifndef OCCA_LIBRARY_HEADER
#define OCCA_LIBRARY_HEADER

#include "occa/tools.hpp"
#include "occa/base.hpp"

namespace occa {
  namespace fileDatabase {
    typedef std::map<std::string,int>  stiMap_t;
    typedef stiMap_t::iterator         stiMapIterator;
    typedef stiMap_t::const_iterator   cStiMapIterator;

    typedef std::map<int,std::string>  itsMap_t;
    typedef itsMap_t::iterator         itsMapIterator;
    typedef itsMap_t::const_iterator   cItsMapIterator;

    extern mutex_t mutex;

    extern itsMap_t itsMap;
    extern stiMap_t stiMap;

    extern int filesInDatabase;

    int getFileID(const std::string &filename);
    std::string getFilename(const int id);
  }

  namespace library {
    class infoID_t {
    public:
      int modelID;
      std::string kernelName;

      inline infoID_t() :
        modelID(-1),
        kernelName("") {}

      inline infoID_t(const infoID_t &id) :
        modelID(id.modelID),
        kernelName(id.kernelName) {}

      inline friend bool operator < (const infoID_t &a, const infoID_t &b){
        if(a.modelID != b.modelID)
          return (a.modelID < b.modelID);

        return (a.kernelName < b.kernelName);
      }
    };

    class infoHeader_t {
    public:
      int fileID;
      uint32_t mode;
      uint64_t flagsOffset, flagsBytes;
      uint64_t contentOffset, contentBytes;
      uint64_t kernelNameOffset, kernelNameBytes;
    };

    typedef std::map<infoID_t,infoHeader_t> headerMap_t;
    typedef headerMap_t::iterator           headerMapIterator;
    typedef headerMap_t::const_iterator     cHeaderMapIterator;

    typedef std::map<std::string,std::vector<int> > kernelMap_t;
    typedef kernelMap_t::iterator                   kernelMapIterator;
    typedef kernelMap_t::const_iterator             cKernelMapIterator;

    typedef std::map<deviceIdentifier,int>   deviceModelMap_t;
    typedef deviceModelMap_t::iterator       deviceModelMapIterator;
    typedef deviceModelMap_t::const_iterator cDeviceMapIterator;

    extern mutex_t headerMutex, kernelMutex;
    extern mutex_t deviceIDMutex, deviceModelMutex;
    extern mutex_t scratchMutex;

    extern headerMap_t headerMap;
    extern kernelMap_t kernelMap;

    extern deviceModelMap_t deviceModelMap;

    extern std::string scratchPad;

    extern int currentDeviceID;

    size_t addToScratchPad(const std::string &s);

    void load(const std::string &filename);
    void save(const std::string &filename);

    int genDeviceID();

    int deviceModelID(occa::device_v *dHandle);
    int deviceModelID(const occa::deviceIdentifier &id);

    occa::kernelDatabase loadKernelDatabase(const std::string &kernelName);

    occa::kernel loadKernel(occa::device_v *dHandle,
                            const std::string &kernelName);
  }
}

#endif
