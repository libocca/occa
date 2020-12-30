#include <iostream>

#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace openmp {
    std::string notSupported = "N/A";

    std::string baseCompilerFlag(const int vendor_) {
      if (vendor_ & (sys::vendor::GNU |
                     sys::vendor::PPC |
                     sys::vendor::LLVM)) {
        return "-fopenmp";
      } else if (vendor_ & sys::vendor::Intel) {
        return "-qopenmp";
      } else if (vendor_ & sys::vendor::Pathscale) {
        return "-openmp";
      } else if (vendor_ & sys::vendor::IBM) {
        return "-qsmp";
      } else if (vendor_ & sys::vendor::PGI) {
        return "-mp";
      } else if (vendor_ & sys::vendor::HP) {
        return "+Oopenmp";
      } else if (vendor_ & sys::vendor::VisualStudio) {
        return "/openmp";
      } else if (vendor_ & sys::vendor::Cray) {
        return "";
      }

      return openmp::notSupported;
    }

    std::string compilerFlag(const int vendor_,
                             const std::string &compiler) {

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      const std::string safeCompiler = io::slashToSnake(compiler);
      std::stringstream ss;

      const std::string openmpTest = env::OCCA_DIR + "/include/occa/scripts/compilerSupportsOpenMP.cpp";
      hash_t hash = occa::hashFile(openmpTest);
      hash ^= occa::hash(vendor_);
      hash ^= occa::hash(compiler);

      const std::string srcFilename = io::cacheFile(openmpTest, "compilerSupportsOpenMP.cpp", hash);
      const std::string binaryFilename = io::dirname(srcFilename) + "binary";
      const std::string outFilename = io::dirname(srcFilename) + "output";

      io::lock_t lock(hash, "openmp-compiler");
      if (lock.isMine()
          && !io::isFile(outFilename)) {
        // Try to compile a minimal OpenMP file to see whether
        // the compiler supports OpenMP or not
        std::string flag = baseCompilerFlag(vendor_);
        ss << compiler
           << ' '    << flag
           << ' '    << srcFilename
           << " -o " << binaryFilename
           << " > /dev/null 2>&1";

        const std::string compileLine = ss.str();
        const int compileError = system(compileLine.c_str());

        if (compileError) {
          flag = openmp::notSupported;
        }

        io::write(outFilename, flag);

        return flag;
      }

      std::string flag = openmp::notSupported;
      ss << io::read(outFilename);
      ss >> flag;

      return flag;
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      return "/openmp"; // VS Compilers support OpenMP
#endif
    }
  }
}
