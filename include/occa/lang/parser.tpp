namespace occa {
  namespace lang {
    template <class funcType>
    void parser_t::setArgumentsFor(funcType &func) {
      tokenRangeVector argRanges;
      getArgumentRanges(argRanges);

      const int argCount = (int) argRanges.size();
      if (!argCount) {
        return;
      }

      for (int i = 0; i < argCount; ++i) {
        context.push(argRanges[i].start,
                     argRanges[i].end);

        func += loadVariable();

        context.pop();
        if (!success) {
          break;
        }
        context.set(argRanges[i].end + 1);
      }
    }

    template <class attributeType>
    void parser_t::addAttribute() {
      attributeType *attr = new attributeType();
      const std::string name = attr->name();

      OCCA_ERROR("Attribute [" << name << "] already exists",
                 attributeMap.find(name) == attributeMap.end());

      attributeMap[name] = attr;
    }
  }
}
