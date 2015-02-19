#include "occaLibrary.hpp"

namespace occa {
  namespace fileDatabase {
    mutex_t mutex;

    itsMap_t itsMap;
    stiMap_t stiMap;

    int filesInDatabase = 0;

    int getFileID(const std::string &filename){
      int id;

      mutex.lock();

      stiMapIterator it = stiMap.find(filename);

      if(it == stiMap.end()){
        id = (filesInDatabase++);

        stiMap[filename] = id;
        itsMap[id]       = filename;
      }
      else
        id = (it->second);

      mutex.unlock();

      return id;
    }

    std::string getFilename(const int id){
      OCCA_CHECK((0 <= id) && (id < filesInDatabase),
                 "File with ID [" << id << "] was not found");

      mutex.lock();

      std::string filename = itsMap[id];

      mutex.unlock();

      return filename;
    }
  };

  namespace library {
    mutex_t headerMutex, kernelMutex;
    mutex_t deviceIDMutex, deviceModelMutex;
    mutex_t scratchMutex;

    headerMap_t headerMap;
    kernelMap_t kernelMap;

    deviceModelMap_t deviceModelMap;

    std::string scratchPad;

    int currentDeviceID = 0;

    size_t addToScratchPad(const std::string &s){
      scratchMutex.lock();

      size_t offset = scratchPad.size();
      scratchPad += s;

      scratchMutex.unlock();

      return offset;
    }

    void load(const std::string &filename){
      //---[ Load file ]------
      std::string sBuffer = readFile(filename);
      const char *buffer  = sBuffer.c_str();

      //---[ Read file ]------
      const uint32_t *buffer32 = (const uint32_t*) buffer;
      const uint64_t *buffer64;

      const uint32_t headerCount = *(buffer32++);

      for(uint32_t i = 0; i < headerCount; ++i){
        infoID_t infoID;

        const int mode_ = *(buffer32++);

        buffer64 = (const uint64_t*) buffer32;
        const uint64_t flagsOffset = *(buffer64++);
        const uint64_t flagsBytes  = *(buffer64++);

        const uint64_t contentOffset = *(buffer64++);
        const uint64_t contentBytes  = *(buffer64++);

        const uint64_t kernelNameOffset = *(buffer64++);
        const uint64_t kernelNameBytes  = *(buffer64++);

        buffer32 = (const uint32_t*) buffer64;

        infoID.kernelName = std::string(buffer + kernelNameOffset,
                                        kernelNameBytes);

        deviceIdentifier identifier(mode_,
                                    buffer + flagsOffset, flagsBytes);

        infoID.modelID = deviceModelID(identifier);

        kernelMutex.lock();
        kernelMap[infoID.kernelName].push_back(infoID.modelID);
        kernelMutex.unlock();

        //---[ Input to header map ]----
        headerMutex.lock();
        infoHeader_t &h = headerMap[infoID];

        h.fileID = fileDatabase::getFileID(filename);
        h.mode   = mode_;

        h.flagsOffset = flagsOffset;
        h.flagsBytes  = flagsBytes;

        h.contentOffset = contentOffset;
        h.contentBytes  = contentBytes;

        h.kernelNameOffset = kernelNameOffset;
        h.kernelNameBytes  = kernelNameBytes;
        headerMutex.unlock();
        //==============================
      }
    }

    void save(const std::string &filename){
      headerMutex.lock();

      FILE *outFD = fopen(filename.c_str(), "wb");

      uint32_t headerCount = headerMap.size();

      if(headerCount == 0){
        headerMutex.unlock();
        return;
      }

      fwrite(&headerCount, sizeof(uint32_t), 1, outFD);

      cHeaderMapIterator it = headerMap.begin();

      const uint64_t headerOffset = headerCount * ((  sizeof(uint32_t)) +
                                                   (6*sizeof(uint64_t)));

      uint64_t contentOffsets = sizeof(headerCount) + headerOffset;

      for(uint32_t i = 0; i < headerCount; ++i){
        const infoHeader_t &h = it->second;

        fwrite(&(h.mode), sizeof(uint32_t), 1, outFD);

        fwrite(&(contentOffsets), sizeof(uint64_t), 1, outFD);
        fwrite(&(h.flagsBytes)  , sizeof(uint64_t), 1, outFD);
        contentOffsets += h.flagsBytes;

        fwrite(&(contentOffsets), sizeof(uint64_t), 1, outFD);
        fwrite(&(h.contentBytes), sizeof(uint64_t), 1, outFD);
        contentOffsets += h.contentBytes;

        fwrite(&(contentOffsets)   , sizeof(uint64_t), 1, outFD);
        fwrite(&(h.kernelNameBytes), sizeof(uint64_t), 1, outFD);
        contentOffsets += h.kernelNameBytes;

        ++it;
      }

      it = headerMap.begin();

      for(uint32_t i = 0; i < headerCount; ++i){
        const infoHeader_t &h = it->second;
        ++it;

        char *buffer = new char[std::max(h.flagsBytes,
                                         std::max(h.contentBytes,
                                                  h.kernelNameBytes))];

        if(0 <= h.fileID){
          const std::string hFilename = fileDatabase::getFilename(h.fileID);
          FILE *inFD = fopen(hFilename.c_str(), "rb");

          fseek(inFD, h.flagsOffset, SEEK_SET);
          ignoreResult( fread(buffer , sizeof(char), h.flagsBytes, inFD) );
          fwrite(buffer, sizeof(char), h.flagsBytes, outFD);

          ignoreResult( fread(buffer , sizeof(char), h.contentBytes, inFD) );
          fwrite(buffer, sizeof(char), h.contentBytes, outFD);

          ignoreResult( fread(buffer , sizeof(char), h.kernelNameBytes, inFD) );
          fwrite(buffer, sizeof(char), h.kernelNameBytes, outFD);

          fclose(inFD);

          delete [] buffer;
        }
        else{
          const char *c  = scratchPad.c_str();
          const char *c1 = c + h.flagsOffset;
          const char *c2 = c + h.contentOffset;
          const char *c3 = c + h.kernelNameOffset;

          fwrite(c1, sizeof(char), h.flagsBytes     , outFD);
          fwrite(c2, sizeof(char), h.contentBytes   , outFD);
          fwrite(c3, sizeof(char), h.kernelNameBytes, outFD);
        }
      }

      fclose(outFD);

      headerMutex.unlock();
    }

    int genDeviceID(){
      deviceIDMutex.lock();
      const int id = (currentDeviceID++);
      deviceIDMutex.unlock();

      return id;
    }

    int deviceModelID(occa::device &dev){
      return deviceModelID(dev.getIdentifier());
    }

    int deviceModelID(const occa::deviceIdentifier &id){
      deviceModelMutex.lock();

      deviceModelMapIterator it = deviceModelMap.find(id);

      int dID;

      if(it != deviceModelMap.end())
        dID = it->second;
      else{
        dID = deviceModelMap.size();
        deviceModelMap[id] = dID;
      }

      deviceModelMutex.unlock();

      return dID;
    }

    occa::kernelDatabase loadKernelDatabase(const std::string &kernelName){
      kernelDatabase kdb(kernelName);

      kernelMutex.lock();

      kernelMapIterator it = kernelMap.find(kernelName);

      if(it != kernelMap.end()){
        std::vector<int> &ids = it->second;

        const int idCount = ids.size();

        for(int i = 0; i < idCount; ++i)
          kdb.modelKernelIsAvailable(ids[i]);
      }

      kernelMutex.unlock();

      return kdb;
    }

    kernel loadKernel(occa::device_v *dHandle,
                      const std::string &kernelName){
      infoID_t infoID;

      infoID.modelID    = dHandle->modelID();
      infoID.kernelName = kernelName;

      headerMutex.lock();
      const infoHeader_t &h = headerMap[infoID];
      headerMutex.unlock();

      const std::string hFilename = fileDatabase::getFilename(h.fileID);
      FILE *inFD = fopen(hFilename.c_str(), "rb");

      char *buffer = new char[h.contentBytes + 1];
      buffer[h.contentBytes] = '\0';

      fseek(inFD, h.contentOffset, SEEK_SET);

      ignoreResult( fread(buffer, sizeof(char), h.contentBytes, inFD) );

      fclose(inFD);

      kernel k = kernel(dHandle->loadKernelFromLibrary(buffer, kernelName));

      delete [] buffer;
      fclose(inFD);

      return k;
    }
  };
};
