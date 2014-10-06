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
      OCCA_CHECK((0 <= id) && (id < filesInDatabase));

      mutex.lock();

      std::string filename = itsMap[id];

      mutex.unlock();

      return filename;
    }
  };

  namespace library {
    mutex_t mutex;
    headerMap_t headerMap;

    std::string scratchPad;

    size_t addToScratchPad(const std::string &s){
      mutex.lock();

      size_t offset = scratchPad.size();
      scratchPad += s;

      mutex.unlock();

      return offset;
    }

    void load(const std::string &filename){
      //---[ Load file ]------
      struct stat fileInfo;

      int fileHandle = ::open(filename.c_str(), O_RDWR);
      const int status = fstat(fileHandle, &fileInfo);

      const uintptr_t bytes = fileInfo.st_size;

      if(status != 0)
        printf("File [%s] gave a bad stat", filename.c_str());

      char *buffer = (char*) malloc(bytes + 1);
      buffer[bytes] = '\0';

      std::ifstream fs(filename.c_str());
      if(!fs) {
        std::cerr << "Unable to read file " << filename;
        throw 1;
      }

      fs.read(buffer, bytes);

      //---[ Read file ]------
      uint32_t *buffer32 = (uint32_t*) buffer;
      uint64_t *buffer64;

      const uint32_t headerCount = *(buffer32++);

      for(uint32_t i = 0; i < headerCount; ++i){
        infoID_t infoID;

        infoID.devID.mode_ = *(buffer32++);

        buffer64 = (uint64_t*) buffer32;
        const uint64_t flagsOffset = *(buffer64++);
        const uint64_t flagsBytes  = *(buffer64++);

        const uint64_t contentOffset = *(buffer64++);
        const uint64_t contentBytes  = *(buffer64++);

        const uint64_t kernelNameOffset = *(buffer64++);
        const uint64_t kernelNameBytes  = *(buffer64++);

        infoID.kernelName = std::string(buffer + kernelNameOffset,
                                        kernelNameBytes);

        infoID.devID.load(buffer + flagsOffset, flagsBytes);

        //---[ Input to header map ]----
        mutex.lock();
        infoHeader_t &h = headerMap[infoID];

        h.fileID = fileDatabase::getFileID(filename);
        h.mode   = infoID.devID.mode_;

        h.flagsOffset = flagsOffset;
        h.flagsBytes  = flagsBytes;

        h.contentOffset = contentOffset;
        h.contentBytes  = contentBytes;

        h.kernelNameOffset = kernelNameOffset;
        h.kernelNameBytes  = kernelNameBytes;
        mutex.unlock();
        //==============================
      }

      free(buffer);
    }

    void save(const std::string &filename){
      mutex.lock();

      FILE *outFD = fopen(filename.c_str(), "wb");

      uint32_t headerCount = headerMap.size();

      if(headerCount == 0){
        mutex.unlock();
        return;
      }

      fwrite(&headerCount, sizeof(uint32_t), 1, outFD);

      cHeaderMapIterator it = headerMap.begin();

      const uint64_t headerOffset = headerCount * ((  sizeof(uint32_t)) +
                                                   (6*sizeof(uint64_t)));

      uint64_t contentOffsets = headerOffset;

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

        char *buffer = new char[std::max(h.flagsBytes,
                                         std::max(h.contentBytes,
                                                  h.kernelNameBytes))];

        if(0 <= h.fileID){
          const std::string hFilename = fileDatabase::getFilename(h.fileID);
          FILE *inFD = fopen(hFilename.c_str(), "rb");

          fseek(inFD, h.flagsOffset, SEEK_SET);
          fread(buffer , sizeof(char), h.flagsBytes, inFD);
          fwrite(buffer, sizeof(char), h.flagsBytes, outFD);

          fread(buffer , sizeof(char), h.contentBytes, inFD);
          fwrite(buffer, sizeof(char), h.contentBytes, outFD);

          fread(buffer , sizeof(char), h.kernelNameBytes, inFD);
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

      mutex.unlock();
    }

    kernel loadKernel(occa::device &dev,
                      const std::string &kernelName){
      infoID_t infoID;

      infoID.devID      = dev.getIdentifier();
      infoID.kernelName = kernelName;

      mutex.lock();
      const infoHeader_t &h = headerMap[infoID];
      mutex.unlock();

      const std::string hFilename = fileDatabase::getFilename(h.fileID);
      FILE *inFD = fopen(hFilename.c_str(), "rb");

      char *buffer = new char[h.contentBytes];

      fseek(inFD, h.contentOffset, SEEK_SET);
      fread(buffer, sizeof(char), h.contentBytes, inFD);

      fclose(inFD);

      kernel k = dev.loadKernelFromLibrary(buffer, kernelName);

      delete [] buffer;
      fclose(inFD);

      return k;
    }
  };
};
