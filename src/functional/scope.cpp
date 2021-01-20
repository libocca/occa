#include <sstream>

#include <occa/functional/scope.hpp>

namespace occa {
  scope::scope() {}

  scope::scope(occa::device device_) :
    device(device_) {}

  scope::scope(scopeKernelArgInitializerList args_,
               const occa::json &props_) :
    props(props_) {
    for (const scopeKernelArg &arg : args_) {
      add(arg);
    }
  }

  scope::scope(const occa::json &props_) :
    props(props_) {}

  template <>
  void scope::add(const std::string &name,
                  occa::memory &mem) {
    add({name, mem, mem.dtype(), false});
  }

  template <>
  void scope::add(const std::string &name,
                  const occa::memory &mem) {
    add({name, mem, mem.dtype(), true});
  }

  scope scope::operator + (const scope &other) const {
    scope ret = *this;
    ret += other;
    return ret;
  }

  scope& scope::operator += (const scope &other) {
    for (const scopeKernelArg &arg : other.args) {
      add(arg);
    }
    props += other.props;

    return *this;
  }

  occa::device scope::getDevice() const {
    return device;
  }

  int scope::size() const {
    return (int) args.size();
  }

  std::string scope::getDeclarationSource() const {
    std::string str;

    bool isFirst = true;
    for (const scopeKernelArg &arg : args) {
      if (!isFirst) {
        str += ", ";
      }
      str += arg.getDeclaration();
      isFirst = false;
    }

    return str;
  }

  std::string scope::getCallSource() const {
    std::string str;

    bool isFirst = true;
    for (const scopeKernelArg &arg : args) {
      if (!isFirst) {
        str += ", ";
      }
      str += arg.name;
      isFirst = false;
    }

    return str;
  }

  kernelArg scope::getArg(const std::string &name) const {
    const int argCount = (int) args.size();

    for (int i = 0; i < argCount; ++i) {
      const scopeKernelArg &arg = args[i];
      if (arg.name == name) {
        return arg;
      }
    }

    OCCA_FORCE_ERROR("Missing argument [" << name << "]");
    return kernelArg();
  }

  hash_t scope::hash() const {
    return (
      occa::hash(props)
      ^ occa::hash(args)
    );
  }

  template <>
  hash_t hash(const occa::scope &scope) {
    return scope.hash();
  }
}
