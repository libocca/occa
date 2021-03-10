#define OCCA_GNU_VENDOR          0
#define OCCA_LLVM_VENDOR         1
#define OCCA_INTEL_VENDOR        2
#define OCCA_PATHSCALE_VENDOR    3
#define OCCA_IBM_VENDOR          4
#define OCCA_PGI_VENDOR          5
#define OCCA_HP_VENDOR           6
#define OCCA_VISUALSTUDIO_VENDOR 7
#define OCCA_CRAY_VENDOR         8
#define OCCA_PPC_VENDOR          9
#define OCCA_NOT_FOUND           10

int main(int argc, char **argv) {
#if  defined(__xlc__)  || defined(__xlC__)    \
  || defined(__IBMC__) || defined(__IBMCPP__) \
  || defined(__ibmxl__)
  return OCCA_IBM_VENDOR;

#elif defined(__ICC) || defined(__INTEL_COMPILER)
  return OCCA_INTEL_VENDOR;

#elif defined(__HP_cc) || defined(__HP_aCC)
  return OCCA_HP_VENDOR;

#elif defined(__PGI)
  return OCCA_PGI_VENDOR;

#elif defined(_CRAYC)
  return OCCA_CRAY_VENDOR;

#elif defined(__PATHSCALE__) || defined(__PATHCC__)
  return OCCA_PATHSCALE_VENDOR;

#elif defined(_MSC_VER)
  return OCCA_VISUALSTUDIO_VENDOR;

#elif defined(__clang__)
  return OCCA_LLVM_VENDOR;

#elif defined(__powerpc__)
  return OCCA_PPC_VENDOR;

// Clang also defines __GNUC__, so check for it after __clang__
#elif defined(__GNUC__) || defined(__GNUG__)
  return OCCA_GNU_VENDOR;

#else
  return OCCA_NOT_FOUND
#endif
}
