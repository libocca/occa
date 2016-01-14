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

occa::kernel buildKernelFromLoopy(occa::device &device,
                                  const std::string &filename,
                                  const std::string &kernelname,
                                  const occa::kernelInfo &info){
  occa::kernel knl;
  occa::mode mode = occa::strToMode(device.mode());

  std::string loopy_target;
  std::string loopy_flags;
  switch(mode){
  case occa::Serial:
    loopy_target = "ispc-occa";
    loopy_flags = "";
    break;
  case occa::CUDA:
    loopy_target = "cuda";
    loopy_flags = " --occa-add-dummy-arg ";
    break;
  case occa::OpenCL:
    loopy_target = "opencl";
    loopy_flags = " --occa-add-dummy-arg ";
    break;
  default:
    OCCA_CHECK(false, "loopy unsupported device mode");
  }

  std::string salt = info.salt() + loopy_flags;
  std::string hash = occa::getFileContentHash(filename, salt);

  if(!occa::haveHash(hash)){
    occa::waitForHash(hash);
  }

  std::string basename = occa::sys::getFilename("[loopy]/" + hash);
  occa::sys::mkpath(basename);

  // Create a file of the defines to pass to loopy
  std::string defines_filename = basename + "/defines.h";
  if(!occa::sys::fileExists(defines_filename)){
    std::ofstream fs;
    fs.open(defines_filename.c_str());
    fs << info.header;
    fs.close();
  }

  // Create backend kernel file from loopy source
  std::string out_filename = basename + "/loopy_source" + "." + loopy_target;
  if(!occa::sys::fileExists(out_filename)){
    std::string command = "loopy " + loopy_flags + " --target " + loopy_target +
      " --occa-defines " + defines_filename + " " + filename + " " + out_filename;
    std::cout << "Loopying [" << filename << "]\n" << command << "\n";

    if(std::system(command.c_str())){
      occa::releaseHash(hash, 0);
      OCCA_CHECK(false, "loopy error");
    }
  }
  else{
    std::cout << "Found cached source of [" << occa::compressFilename(filename)
              << "] in [" << occa::compressFilename(out_filename) << "]\n";
  }

  occa::releaseHash(hash);

  // Compile the backend file
  switch(mode){
  case occa::Serial:
    knl = buildKernelFromISPC(device, out_filename, kernelname, info);
    break;
  case occa::CUDA:
  case occa::OpenCL:
    knl = device.buildKernelFromSource(out_filename, kernelname, info);
    break;
  default:
    OCCA_CHECK(false, "loopy unsupported device mode");
  }


  return knl;
}

int main(int argc, char **argv){
  occa::printAvailableDevices();

  // For ispc this needs to be a multiple of the vector length
  int entries = 40;

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
  // device.setup("mode = OpenCL, platformID = 0, deviceID = 0");
  // device.setup("mode = CUDA, deviceID = 0");
  info.mode = occa::strToMode(device.mode());

  info.addDefine("N", "234");

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  addVectors =
    buildKernelFromLoopy(device, "addVectors.floopy", "addVectors", info);

  int dims = 1;
  // For ispc this needs to be the vector length
  int itemsPerGroup = 8;
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
