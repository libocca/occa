#pragma once

#include <occa/types/json.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>
#include <filesystem>
#include <functional>

#if BUILD_WITH_CLANG_BASED_TRANSPILER
#include <oklt/core/transpiler_session/user_output.h>
#include <oklt/core/error.h>
#endif

namespace occa {
namespace transpiler {

void makeMetadata(lang::sourceMetadata_t &sourceMetadata,
                  const std::string &jsonStr);


#if BUILD_WITH_CLANG_BASED_TRANSPILER
struct Transpiler {
    using SuccessFunc = std::function<bool(const oklt::UserOutput &output, bool hasLauncher)>;
    using FailFunc = std::function<void(const std::vector<oklt::Error> &errors)>;
    using WrongInputFile = std::function<void(const std::string &filename)>;
    using WrongBackend = std::function<void(const std::string &mode)>;

    Transpiler(SuccessFunc success,
               FailFunc fail,
               WrongInputFile wrongInputFile,
               WrongBackend wrongBackend);
    ~Transpiler() = default;
    Transpiler(const Transpiler&) = delete;
    Transpiler & operator = (const Transpiler &) = delete;
    Transpiler(Transpiler &&) = delete;
    Transpiler & operator = (Transpiler &&) = delete;

    bool run(const std::string &filename,
             const std::string &mode,
             const occa::json &kernelProps);
private:
    SuccessFunc _success;
    FailFunc _fail;
    WrongInputFile _wrongInput;
    WrongBackend _wrongBackend;
};
#endif


}
}
