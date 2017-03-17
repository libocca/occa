#include <iostream>

#include "occa.hpp"
#include "occa/tools.hpp"

std::string ispcCreateSourceFileFrom(const std::string &filename,
                                     const std::string &hash,
                                     const occa::kernelInfo &info){

  std::string cached_filename =
    occa::sys::getFilename("[ispc]/" + filename + "_" + hash + ".cpp");

  if(occa::sys::fileExists(cached_filename))
    return cached_filename;

  occa::sys::mkpath(occa::sys::getFilename("[ispc]/"));

  occa::setupOccaHeaders(info);

  std::ofstream fs;
  fs.open(cached_filename.c_str());

  if(filename.substr(filename.find_last_of(".") + 1) != "ispc"){
    //    fs << "#include \"" << info.getModeHeaderFilename() << "\"\n"
       //<< "#include \"" << occa::sys::getFilename("[occa]/primitives.hpp") << "\"\n"
      fs << occa::readFile("patchISPC.hpp");
  }

  fs << info.header
     << occa::readFile(filename);

  fs.close();

  return cached_filename;
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
  info.mode = occa::Serial;
  info.addDefine("OCCA_SERIAL_DEFINES_HEADER", 1);

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  //  std::string ispc_file = "addVectors.ispc";
  std::string ispc_file = "addVectors.occa";

  // Note the following is not working yet;  The right headers need developed
  // to get this occa file to compile with ispc.
  //
  // std::string ispc_file = "addVectors.occa";

  std::string ispc_flags = " -g -O2 --pic -I " + occa::env::OCCA_DIR + "/include";
  std::string hash = occa::getFileContentHash(ispc_file, ispc_flags);

  std::string ispc_src_file = ispcCreateSourceFileFrom(ispc_file, hash, info);
  std::string ispc_object =
    occa::sys::getFilename("[ispc]/" + ispc_file + "_" + hash + ".o");
  std::string ispc_sharedobject =
    occa::sys::getFilename("[ispc]/" + ispc_file + "_" + hash + ".so");

  if(!occa::haveHash(hash)){
    occa::waitForHash(hash);
  }

  if(std::system("test -f tasksys.cpp")){
    occa::releaseHash(hash, 0);
    OCCA_CHECK(false, "The required ispc tasksys.cpp file not found;"
               " you can download it with:\n     curl -O "
               "https://raw.githubusercontent.com/ispc/ispc/master/examples/tasksys.cpp");
  }

  if(!occa::sys::fileExists(ispc_object)){
    // Hack to make sure the ispc directory exists.  TODO figure out the occa
    // way to do this?
    if(std::system("mkdir -p ~/._occa/libraries/ispc")){
      occa::releaseHash(hash, 0);
      OCCA_CHECK(false, "problems making ispc dir");
    }

    std::string command =
      "ispc " + ispc_flags + " " + ispc_src_file + " -o " + ispc_object;

    std::cout << "Compiling [" << ispc_file << "]\n" << command << "\n";

    if(std::system(command.c_str())){
      occa::releaseHash(hash, 0);
      OCCA_CHECK(false, "ispc compilation error");
    }
    std::string sharedcommand = "g++ -Wl,-E -g -pipe -O2 -pipe -fPIC " +
      ispc_object + " tasksys.cpp -shared -o " + ispc_sharedobject;
    std::cout << "Creating shared object [" << ispc_file << "]\n" <<
      sharedcommand << "\n";

    if(std::system(sharedcommand.c_str())){
      occa::releaseHash(hash, 0);
      OCCA_CHECK(false, "ispc compilation error");
    }
  }

  addVectors = device.buildKernelFromBinary(ispc_sharedobject, "addVectors");
  occa::releaseHash(hash);

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
