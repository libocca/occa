#include <occa/defines.hpp>
#include <occa/types/typedefs.hpp>
#include <occa/internal/io/output.hpp>

// Only used by Visual Studio
#ifdef _MSC_VER


// NBN: extra utility routines for Visual Studio / Windows
//
// 1. getVScompilerScript() - find VC compiler
// 2. initModesForVS()      - force registration of occa modes
// 3. getline(...)          - add missing routine
// 4. getProcFreq()         - helper routine
// 5. mkdtemp (...)         - add missing routine


//---------------------------------------------------------
// 1. getVScompilerScript() - find VC compiler
//---------------------------------------------------------
namespace occa {
  namespace io {

    std::string getVScompilerScript()
    {
      std::string byteness;
      if (sizeof(void*) == 4) {
        byteness = "x86 ";               // NBN: 32bit not supported by recent CUDA
      } else if (sizeof(void*) == 8) {
        byteness = "x64";
      } else {
        OCCA_FORCE_ERROR("sizeof(void*) is not equal to 4 or 8");
      }

      // get Visual C++ installation directory
      char* sz = NULL; size_t len = 0;

# if   (OCCA_VS_VERSION >= 1933)
      _dupenv_s(&sz, &len, "VSINSTALLDIR");         // NBN: MSVC++ 17.x - Visual Studio 2022
# elif (OCCA_VS_VERSION >= 1920)
      _dupenv_s(&sz, &len, "VS2019INSTALLDIR" );    // NBN: MSVC++ 16.x - Visual Studio 2019
# else
      // Handle other versions here:
      // NBN: add as required

      OCCA_FORCE_ERROR("Please adjust getVScompilerScript for Visual Studio version ["
        << OCCA_VS_VERSION
        << ".  Exiting simulation");

# endif

      std::string VSdir(sz);
      ::free(sz);
      std::string compilerEnvScript;
      if (!VSdir.empty()) {
      //compilerEnvScript = "\"" + VSdir + "\\VC\\Auxiliary\\Build\\vcvarsall.bat\" " + byteness;
        compilerEnvScript = "\"" + VSdir +   "VC\\Auxiliary\\Build\\vcvarsall.bat\" " + byteness;
      } else {
        io::stdout << "WARNING: Visual Studio 'vcvarsall.bat' not found.\n";
      }
      
      return compilerEnvScript;
    }
  }
}


//---------------------------------------------------------
// 2. initModesForVS() - force registration of occa modes
//---------------------------------------------------------
#include <occa/internal/modes/serial/registration.hpp>
#include <occa/internal/modes/openmp/registration.hpp>
#include <occa/internal/modes/cuda/registration.hpp>
//#include <occa/internal/modes/opencl/registration.hpp>

namespace occa {

  // NBN: referring to these modes in a source file appears to make 
  // VC compiler enable them. No need to actually call this routine.
  bool initModesForVS()
  {
    if (occa::getModeMap().size() < 1) {
      printf("no occa modes are enabled?");
      return false;
    }

    // Add modes as required
    occa::serial::mode.getDescription();
    occa::openmp::mode.getDescription();
    occa::cuda::mode.getDescription();
  //occa::opencl::mode.getDescription();

    // check which modes are enabled in occa library
    occa::printModeInfo();
    return true;
  }
}


//---------------------------------------------------------
// 3. getline(...) - add missing routine
//---------------------------------------------------------
int64_t getline(char** line, size_t* len, FILE* fp) 
{
  // NBN: Visual Studio replacement for POSIX getline
  // See: https://solarianprogrammer.com/2019/04/03/c-programming-read-file-lines-fgets-getline-implement-portable-getline/

  // Check if either line, len or fp are NULL pointers
  if (line == NULL || len == NULL || fp == NULL) {
    errno = EINVAL;
    return -1;
  }

  // Use a chunk array of 128 bytes as parameter for fgets
  char chunk[128] = {'\0'};

  // NBN: in case file has no '\n', use chunkFlag 
  // to flag when all chars have been read.
  int chunkFlag = (128-1);
//int chunkFlag = sizeof(chunk)-1;  // 127

  // Allocate a block of memory for *line if it is NULL or smaller than the chunk array
  if (*line == NULL || *len < sizeof(chunk)) {
    *len = sizeof(chunk);
    if ((*line = (char*) malloc(*len)) == NULL) {
      errno = ENOMEM;
      return -1;
    }
  }

  // Make the string empty
  (*line)[0] = '\0';

  // read from current stream position
  while (fgets(chunk, sizeof(chunk), fp) != NULL) {
    // Resize the line buffer if necessary
    size_t len_used = strlen(*line);
    size_t chunk_used = strlen(chunk);

    if (*len - len_used < chunk_used) {
      // Check for overflow
      if (*len > SIZE_MAX / 2) {
        errno = EOVERFLOW;
        return -1;
      }
      else {
        *len *= 2;
      }

      if ((*line = (char*)realloc(*line, *len)) == NULL) {
        errno = ENOMEM;
        return -1;
      }
    }

    // Copy the chunk to the end of the line buffer
    memcpy(*line + len_used, chunk, chunk_used);
    len_used += chunk_used;
    (*line)[len_used] = '\0';

    // if *line contains '\n',
    if ((*line)[len_used - 1] == '\n') {
      // ... return current length of line buffer
      return len_used;
    }

    // if all chars in *line have been read,
    if (chunk_used < chunkFlag) {
      // ... return current length of line buffer
      return len_used;
    }
  }

  return -1;
}


//---------------------------------------------------------
// 4. getProcFreq() - helper routine
//---------------------------------------------------------
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <powrprof.h>
#include <vector>

// link the necessary libray
#pragma comment(lib, "Powrprof.lib")

typedef struct _PROCESSOR_POWER_INFORMATION {
    ULONG  Number;
    ULONG  MaxMhz;
    ULONG  CurrentMhz;
    ULONG  MhzLimit;
    ULONG  MaxIdleState;
    ULONG  CurrentIdleState;
} PROCESSOR_POWER_INFORMATION, *PPROCESSOR_POWER_INFORMATION;


occa::udim_t getProcFreq()
{
  SYSTEM_INFO si = {0}; GetSystemInfo(&si);
  int nprocs = si.dwNumberOfProcessors;
  std::vector<PROCESSOR_POWER_INFORMATION> buf(nprocs);
  DWORD dwSize = sizeof(PROCESSOR_POWER_INFORMATION) * nprocs;
  CallNtPowerInformation(ProcessorInformation, NULL, 0, &buf[0], dwSize);
  occa::udim_t max_Mhz = buf[0].MaxMhz;
  buf.clear();
  return (max_Mhz*1000*1000);  // return Hz
}


//---------------------------------------------------------
// 5. mkdtemp (...) - add missing routine
// NBN: 2023/01/20 - no longer needed?
//---------------------------------------------------------
#if (0)
#include <process.h>
#include <direct.h>
#include <errno.h>
#include <random>

// NBN: reproduce expected behavior of mkdtemp()
//      adapted from adaption from OpenBSD
// 
// Generate a unique temporary directory from TEMPLATE.
// The last six characters of TEMPLATE must be "XXXXXX";
// these are replaced with string that makes filename unique.
// The directory is created, and its name is returned.

// characters used to build temporary file name [62 chars]
static const char s_letters[] =
"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

// if success, return name of newly created directory
// else, return NULL
char* mkdtemp (char *TMPL)
{
  uint64_t value = 0;
  int fd = -1;
  int save_errno = errno;
  int len = strlen (TMPL);

  // check that template name ends with "XXXXXX"
  if (len < 6 || memcmp (&TMPL[len-6],  "XXXXXX", 6)) {
    _set_errno (EINVAL);
    return NULL;
  }

  // NBN: make static? Avoid recreating when building lots of kernel files
  static std::random_device rd;
  static std::mt19937_64 gen(rd());

  value = gen();                          // generate a random 64 bit value
  value ^= ((uint64_t) _getpid()) << 32;  // adjust value for this proc

  // number of times to try generating a unique temp name.
  int attempts = 1;

  // pointer into template name string
  char* XXXXXX = &TMPL[len - 6]; // point to start of [XXXXXX]

  for (int count = 0; count < attempts; value += 7777, ++count) 
  {
    uint64_t v = value;

    // fill in the random bits.
    XXXXXX[0] = s_letters[v % 62];  v /= 62;
    XXXXXX[1] = s_letters[v % 62];  v /= 62;
    XXXXXX[2] = s_letters[v % 62];  v /= 62;
    XXXXXX[3] = s_letters[v % 62];  v /= 62;
    XXXXXX[4] = s_letters[v % 62];  v /= 62;
    XXXXXX[5] = s_letters[v % 62];


#if (OCCA_OS != OCCA_WINDOWS_OS)
    fd = __mkdir (TMPL, S_IRUSR | S_IWUSR | S_IXUSR);
#else
    fd =  _mkdir (TMPL);
#endif

    if (0 == fd) {
      _set_errno (save_errno);  // restore old errno
      return TMPL;              // return name of created directory
    }
    else 
    {
      if (EEXIST != errno) { 
        // abort: some other error state
        return NULL; 
      } 
      else { 
        // directory exists. try again with new temp_name
        printf("*** directory %s exists (attempt %d of %d)\n", TMPL, count, attempts);
      }
    }
  }

  // Exceeded max combinations to try.
  _set_errno (EEXIST);
  return NULL;
}
#endif  // no longer needed
#endif  // _MSC_VER
