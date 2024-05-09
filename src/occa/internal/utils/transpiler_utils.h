#pragma once

#include <occa/types/json.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>
#include <filesystem>

namespace occa {
namespace transpiler {

std::string getKernelHash(const json &kernelProp);
std::vector<std::string> buildDefines(const json &kernelProp);
std::vector<std::filesystem::path> buildIncludes(const json &kernelProp);
void makeMetadata(lang::sourceMetadata_t &sourceMetadata,
                  const std::string &jsonStr);

}
}
