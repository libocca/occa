#ifndef OCCA_LIBRARY_HEADER
#define OCCA_LIBRARY_HEADER

#include "occaTools.hpp"
#include "occaBase.hpp"

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
  };

  namespace library {
    class infoID_t {
    public:
      deviceIdentifier devID;
      std::string kernelName;

      inline infoID_t() :
        devID(),
        kernelName("") {}

      inline infoID_t(const infoID_t &id) :
        devID(id.devID),
        kernelName(id.kernelName) {}

      inline friend bool operator < (const infoID_t &a, const infoID_t &b){
        const int sc = a.devID.compare(b.devID);

        if(sc)
          return (sc < 0);

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

    extern mutex_t mutex;
    extern headerMap_t headerMap;

    extern std::string scratchPad;

    size_t addToScratchPad(const std::string &s);

    void load(const std::string &filename);
    void save(const std::string &filename);

    occa::kernel loadKernel(occa::device &dev,
                            const std::string &kernelName);
  };
};

#endif
