#include <iostream>

#include "occa.hpp"
#include "occa/tools.hpp"

std::string ispcCreateSourceFileFrom(const std::string &filename,
                                     const std::string &hash,
                                     const occa::kernelInfo &info){

  std::string cached_filename =
    occa::sys::getFilename("[ispc]/" + hash + "/source.ispc");

  if(occa::sys::fileExists(cached_filename))
    return cached_filename;

  occa::sys::mkpath(occa::getFileDirectory(cached_filename));

  occa::setupOccaHeaders(info);

  std::ofstream fs;
  fs.open(cached_filename.c_str());

  if(filename.substr(filename.find_last_of(".") + 1) != "ispc")
    fs << occa::readFile("ISPC.hpp");

  fs << info.header << occa::readFile(filename);

  fs.close();

  return cached_filename;
}

occa::kernel buildKernelFromISPC(occa::device &device,
                                 const std::string &filename,
                                 const std::string &kernelname,
                                 const occa::kernelInfo &info){
  occa::kernel knl;
  occa::mode mode = occa::strToMode(device.mode());
  OCCA_CHECK(mode == occa::Serial, "ispc unsupported device mode");

  // std::string ispc_flags = " -g -O2 --pic --opt=force-aligned-memory ";
  std::string ispc_flags = " -g -O2 --pic ";

  std::string salt = info.salt() + ispc_flags;
  std::string hash = occa::getFileContentHash(filename, salt);

  if(!occa::haveHash(hash)){
    occa::waitForHash(hash);
  }

  if(std::system("test -f tasksys.cpp")){
    occa::releaseHash(hash, 0);
    OCCA_CHECK(false, "The required ispc tasksys.cpp file not found;"
               " you can download it with:\n     curl -O "
               "https://raw.githubusercontent.com/ispc/ispc/master/examples/tasksys.cpp");
  }

  std::string basename = occa::sys::getFilename("[ispc]/" + hash + "/");
  occa::sys::mkpath(basename);

  std::string cached_filename = ispcCreateSourceFileFrom(filename, hash, info);

  // Create backend kernel file from ispc source
  std::string obj = basename + "/binary.o";
  std::string sobj = basename + "/binary.so";

  if(!occa::sys::fileExists(obj)){
    std::string command = "ispc " + ispc_flags + " " + cached_filename + " -o " + obj;
    std::cout << "Compiling [" << filename << "]\n" << command << "\n";
    if(std::system(command.c_str())){
      occa::releaseHash(hash, 0);
      OCCA_CHECK(false, "ispc error");
    }
    std::string sharedcommand = "g++ -fopenmp -Wl,-E -g -pipe -O2 -pipe -fPIC " +
      obj + " tasksys.cpp -shared -o " + sobj;
     std::cout << "Creating shared object [" << filename << "]\n" <<
       sharedcommand << "\n";

     if(std::system(sharedcommand.c_str())){
       occa::releaseHash(hash, 0);
       OCCA_CHECK(false, "ispc compilation error");
     }
  }
  else{
    std::cout << "Found cached binary of [" << occa::compressFilename(filename)
              << "] in [" << occa::compressFilename(sobj) << "]\n";
  }

  occa::releaseHash(hash);

  knl = device.buildKernelFromBinary(sobj, kernelname);

  return knl;
}


int main(int argc, char **argv){
  occa::printAvailableDevices();

  int entries = 37;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for(int i = 0; i < entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::device device;
  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;
  occa::kernelInfo info;

  device.setup("mode = Serial");

  // More work is needed for the OpenCL and CUDA backends to work with
  // the .occa kernel.
  //
  // device.setup("mode = OpenCL, platformID = 0, deviceID = 0");
  // device.setup("mode = CUDA, deviceID = 0");

  info.mode = occa::strToMode(device.mode());

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  // std::string filename = "addVectors.ispc";
  std::string filename = "addVectors.occa";

  std::string kernelname = "addVectors";

  if(occa::strToMode(device.mode()) == occa::Serial)
    addVectors = buildKernelFromISPC(device, filename, kernelname, info);
  else
    addVectors = device.buildKernelFromSource(filename, kernelname, info);

  int dims = 1;
  int itemsPerGroup = 16;
  int groups((entries + itemsPerGroup - 1)/itemsPerGroup);

  addVectors.setWorkingDims(dims, itemsPerGroup, groups);

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for(int i = 0; i < entries; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  for(int i = 0; i < entries; ++i){
    if(ab[i] != (a[i] + b[i]))
      throw 1;
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  addVectors.free();
  o_a.free();
  o_b.free();
  o_ab.free();
  device.free();

  return 0;
}
