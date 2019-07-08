namespace occa {
  namespace lang {
    template <class funcType>
    void variableLoader_t::setArgumentsFor(funcType &func) {
      tokenRangeVector argRanges;
      getArgumentRanges(tokenContext, argRanges);

      const int argCount = (int) argRanges.size();
      if (!argCount) {
        return;
      }

      for (int i = 0; i < argCount; ++i) {
        tokenContext.push(argRanges[i].start,
                          argRanges[i].end);

        variable_t arg;
        success = loadVariable(arg);
        tokenContext.pop();

        if (!success) {
          return;
        }

        func.addArgument(arg);
        tokenContext.set(argRanges[i].end + 1);
      }
    }
  }
}
