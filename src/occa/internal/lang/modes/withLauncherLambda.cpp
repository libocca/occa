#include <occa/internal/lang/modes/withLauncherLambda.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/modes/okl.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>
#include <occa/internal/lang/builtins/types.hpp>

namespace occa
{
  namespace lang
  {
    namespace okl
    {
      withLauncherLambda::withLauncherLambda(const occa::properties &settings_) : parser_t(settings_),
                                                                                  launcherParser(settings["launcher"]),
                                                                                  memoryType(NULL)
      {
        launcherParser.settings["okl/validate"] = false;
      }

      //---[ Public ]-------------------
      bool withLauncherLambda::succeeded() const
      {
        return (success && launcherParser.success);
      }

      void withLauncherLambda::writeLauncherSourceToFile(const std::string &filename) const
      {
        launcherParser.writeToFile(filename);
      }
      //================================

      void withLauncherLambda::launcherClear()
      {
        launcherParser.onClear();

        // Will get deleted by the parser
        memoryType = NULL;
      }

      void withLauncherLambda::afterParsing()
      {
        if (!success)
          return;
        if (settings.get("okl/validate", true))
        {
          success = kernelsAreValid(root);
        }

        if (!memoryType)
        {
          identifierToken memoryTypeSource(originSource::builtin,
                                           "occa::modeMemory_t");
          memoryType = new typedef_t(vartype_t(),
                                     memoryTypeSource);
        }

        root.addToScope(*memoryType);

        if (!success)
          return;
        setOKLLoopIndices();

        if (!success)
          return;
        setupLauncherParser();

        if (!success)
          return;
        beforeKernelSplit();

        if (!success)
          return;
        splitKernels();

        if (!success)
          return;
        setupKernels();

        if (!success)
          return;
        afterKernelSplit();
      }

      void withLauncherLambda::beforeKernelSplit() {}

      void withLauncherLambda::afterKernelSplit() {}

      void withLauncherLambda::setOKLLoopIndices()
      {
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             root,
                             kernelSmnts);

        const int kernelCount = (int)kernelSmnts.size();
        for (int i = 0; i < kernelCount; ++i)
        {
          functionDeclStatement &kernelSmnt = (*((functionDeclStatement *)kernelSmnts[i]));
          okl::setLoopIndices(kernelSmnt);
          if (!success)
            return;
        }
      }

      void withLauncherLambda::setupLauncherParser()
      {
        // // Clone source
        // blockStatement &rootClone = (blockStatement &)root.clone();

        // launcherParser.root.swap(rootClone);
        // delete &rootClone;
        // launcherParser.setupKernels();

        // // Remove outer loops
        // statementPtrVector kernelSmnts;
        // findStatementsByAttr(statementType::functionDecl,
        //                      "kernel",
        //                      launcherParser.root,
        //                      kernelSmnts);

        // const int kernelCount = (int)kernelSmnts.size();
        // for (int i = 0; i < kernelCount; ++i)
        // {
        //   functionDeclStatement &kernelSmnt = (*((functionDeclStatement *)kernelSmnts[i]));
        //   removeLauncherOuterLoops(kernelSmnt);
        //   if (!success)
        //     return;
        //   setupLauncherKernelArgs(kernelSmnt);
        //   if (!success)
        //     return;
        // }

        // setupLauncherHeaders();
      }

      void withLauncherLambda::removeLauncherOuterLoops(functionDeclStatement &kernelSmnt)
      {
        // statementPtrVector outerSmnts;
        // findStatementsByAttr(statementType::for_,
        //                      "outer",
        //                      kernelSmnt,
        //                      outerSmnts);

        // const int outerCount = (int)outerSmnts.size();
        // int kernelIndex = 0;
        // for (int i = 0; i < outerCount; ++i)
        // {
        //   forStatement &forSmnt = *((forStatement *)outerSmnts[i]);
        //   if (!isOuterMostOuterLoop(forSmnt))
        //   {
        //     continue;
        //   }
        //   setKernelLaunch(kernelSmnt,
        //                   forSmnt,
        //                   kernelIndex++);
        // }
      }

      bool withLauncherLambda::isOuterMostOuterLoop(forStatement &forSmnt)
      {
        return isOuterMostOklLoop(forSmnt, "outer");
      }

      bool withLauncherLambda::isOuterMostInnerLoop(forStatement &forSmnt)
      {
        return isOuterMostOklLoop(forSmnt, "inner");
      }

      bool withLauncherLambda::isOuterMostOklLoop(forStatement &forSmnt,
                                                  const std::string &attr)
      {
        statement_t *smnt = forSmnt.up;
        while (smnt)
        {
          if ((smnt->type() & statementType::for_) && smnt->hasAttribute(attr))
          {
            return false;
          }
          smnt = smnt->up;
        }
        return true;
      }

      void withLauncherLambda::setKernelLaunch(functionDeclStatement &kernelSmnt,
                                               forStatement &forSmnt,
                                               const int kernelIndex)
      {
        forStatement *innerSmnt = getInnerMostInnerLoop(forSmnt);
        if (!innerSmnt)
        {
          success = false;
          forSmnt.printError("No [@inner] for-loop found");
          return;
        }

        statementPtrVector path;
        oklForStatement::getOKLLoopPath(*innerSmnt, path);

        // Create block in case there are duplicate variable names
        blockStatement &launchBlock = (*new blockStatement(forSmnt.up, forSmnt.source));
        forSmnt.up->addBefore(forSmnt, launchBlock);

        // Get max count
        int outerCount = 0;
        int innerCount = 0;
        const int pathCount = (int)path.size();
        for (int i = 0; i < pathCount; ++i)
        {
          forStatement &pathSmnt = *((forStatement *)path[i]);
          oklForStatement oklForSmnt(pathSmnt);
          if (!oklForSmnt.isValid())
          {
            success = false;
            return;
          }
          const bool isOuter = pathSmnt.hasAttribute("outer");
          outerCount += isOuter;
          innerCount += !isOuter;
        }
        const int outerDims = outerCount;
        const int innerDims = innerCount;

        // TODO 1.1: Properly fix this
        for (int i = 0; i < pathCount; ++i)
        {
          forStatement &pathSmnt = *((forStatement *)path[i]);
          oklForStatement oklForSmnt(pathSmnt);

          launchBlock.add(pathSmnt.init->clone(&launchBlock));

          const bool isOuter = pathSmnt.hasAttribute("outer");
          outerCount -= isOuter;
          innerCount -= !isOuter;

          const int index = (isOuter
                                 ? outerCount
                                 : innerCount);
          token_t *source = pathSmnt.source;
          const std::string &name = (isOuter
                                         ? "outer"
                                         : "inner");
          launchBlock.add(
              *(new expressionStatement(
                  &launchBlock,
                  setDim(source, name, index,
                         oklForSmnt.getIterationCount()))));
        }

        launchBlock.addFirst(
            *(new expressionStatement(
                &launchBlock,
                *(new identifierNode(forSmnt.source,
                                     "inner.dims = " + occa::toString(innerDims))))));
        launchBlock.addFirst(
            *(new expressionStatement(
                &launchBlock,
                *(new identifierNode(forSmnt.source,
                                     "outer.dims = " + occa::toString(outerDims))))));
        launchBlock.addFirst(
            *(new expressionStatement(
                &launchBlock,
                *(new identifierNode(forSmnt.source,
                                     "occa::dim outer, inner")))));
        // Wrap kernel
        std::stringstream ss;
        ss << "occa::kernel kernel(deviceKernel["
           << kernelIndex
           << "])";
        launchBlock.add(
            *(new expressionStatement(
                &launchBlock,
                *(new identifierNode(forSmnt.source,
                                     ss.str())))));
        // Set run dims
        launchBlock.add(
            *(new expressionStatement(
                &launchBlock,
                *(new identifierNode(forSmnt.source,
                                     "kernel.setRunDims(outer, inner)")))));
        // launch kernel
        std::string kernelCall = "kernel(";
        function_t &func = kernelSmnt.function;
        const int argCount = (int)func.args.size();
        for (int i = 0; i < argCount; ++i)
        {
          variable_t &arg = *(func.args[i]);
          if (i)
          {
            kernelCall += ", ";
          }
          kernelCall += arg.name();
        }
        kernelCall += ')';
        launchBlock.add(
            *(new expressionStatement(
                &launchBlock,
                *(new identifierNode(forSmnt.source,
                                     kernelCall)))));

        forSmnt.removeFromParent();

        // TODO 1.1: Delete after properly cloning the declaration statement
        // delete &forSmnt;
      }

      void withLauncherLambda::setupLauncherKernelArgs(functionDeclStatement &kernelSmnt)
      {
        // function_t &func = kernelSmnt.function;

        // // Create new types
        // identifierToken kernelTypeSource(kernelSmnt.source->origin,
        //                                  "occa::modeKernel_t");
        // type_t &kernelType = *(new typedef_t(vartype_t(),
        //                                      kernelTypeSource));

        // // Convert pointer arguments to modeMemory_t
        // int argCount = (int)func.args.size();
        // for (int i = 0; i < argCount; ++i)
        // {
        //   variable_t &arg = *(func.args[i]);
        //   arg.vartype = arg.vartype.flatten();
        //   if (arg.vartype.isPointerType())
        //   {
        //     arg.vartype = *memoryType;
        //     arg += pointer_t();
        //   }
        // }

        // // Add kernel array as the first argument
        // identifierToken kernelVarSource(kernelSmnt.source->origin,
        //                                 "*deviceKernel");
        // variable_t &kernelVar = *(new variable_t(kernelType,
        //                                          &kernelVarSource));
        // kernelVar += pointer_t();

        // func.args.insert(func.args.begin(),
        //                  &kernelVar);

        // kernelSmnt.addToScope(kernelType);
        // kernelSmnt.addToScope(kernelVar);

        // //handle shared variables
      }

      void withLauncherLambda::setupLauncherHeaders()
      {
        // // TODO 1.1: Remove hack after methods are properly added
        // const int headerCount = 2;
        // std::string headers[headerCount] = {
        //     "include <occa/core/base.hpp>",
        //     "include <occa/modes/serial/kernel.hpp>"};
        // for (int i = 0; i < headerCount; ++i)
        // {
        //   std::string header = headers[i];
        //   directiveToken token(root.source->origin,
        //                        header);
        //   launcherParser.root.addFirst(
        //       *(new directiveStatement(&root, token)));
        // }
      }

      int withLauncherLambda::getInnerLoopLevel(forStatement &forSmnt)
      {
        statement_t *smnt = forSmnt.up;
        int level = 0;
        while (smnt)
        {
          if ((smnt->type() & statementType::for_) && smnt->hasAttribute("inner"))
          {
            ++level;
          }
          smnt = smnt->up;
        }
        return level;
      }

      forStatement *withLauncherLambda::getInnerMostInnerLoop(forStatement &forSmnt)
      {
        statementPtrVector innerSmnts;
        findStatementsByAttr(statementType::for_,
                             "inner",
                             forSmnt,
                             innerSmnts);

        int maxLevel = -1;
        forStatement *innerMostInnerLoop = NULL;

        const int innerCount = (int)innerSmnts.size();
        for (int i = 0; i < innerCount; ++i)
        {
          forStatement &innerSmnt = *((forStatement *)innerSmnts[i]);
          const int level = getInnerLoopLevel(innerSmnt);
          if (level > maxLevel)
          {
            maxLevel = level;
            innerMostInnerLoop = &innerSmnt;
          }
        }

        return innerMostInnerLoop;
      }

      exprNode &withLauncherLambda::setDim(token_t *source,
                                           const std::string &name,
                                           const int index,
                                           exprNode *value)
      {
        identifierNode var(source, name);
        primitiveNode idx(source, index);
        subscriptNode access(source, var, idx);
        exprNode &assign = (*(new binaryOpNode(source,
                                               op::assign,
                                               access,
                                               *value)));
        delete value;
        return assign;
      }

      void withLauncherLambda::splitKernels()
      {
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             root,
                             kernelSmnts);

        const int kernelCount = (int)kernelSmnts.size();
        for (int i = 0; i < kernelCount; ++i)
        {
          functionDeclStatement &kernelSmnt = (*((functionDeclStatement *)kernelSmnts[i]));
          splitKernel(kernelSmnt);
          if (!success)
            return;
        }
      }

      void withLauncherLambda::splitKernel(functionDeclStatement &kernelSmnt)
      {
        statementPtrVector outerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             kernelSmnt,
                             outerSmnts);

        statementPtrVector newKernelSmnts;

        const int outerCount = (int)outerSmnts.size();
        int kernelIndex = 0;
        for (int i = 0; i < outerCount; ++i)
        {
          forStatement &forSmnt = *((forStatement *)outerSmnts[i]);
          if (!isOuterMostOuterLoop(forSmnt))
          {
            continue;
          }
          newKernelSmnts.push_back(
              extractLoopAsKernel(kernelSmnt,
                                  forSmnt,
                                  kernelIndex++));
        }

        int smntIndex = kernelSmnt.childIndex();
        for (int i = (kernelIndex - 1); i >= 0; --i)
        {
          root.add(*(newKernelSmnts[i]),
                   smntIndex);
        }

        root.remove(kernelSmnt);
        root.removeFromScope(kernelSmnt.function.name(), true);

        // TODO 1.1: Find out what causes segfault here
        // delete &kernelSmnt;
      }

      statement_t *withLauncherLambda::extractLoopAsKernel(functionDeclStatement &kernelSmnt,
                                                           forStatement &forSmnt,
                                                           const int kernelIndex)
      {

        /*       function_t &oldFunction = kernelSmnt.function;
        function_t &newFunction = (function_t&) oldFunction.clone();
        std::stringstream ss;
        ss << +"_occa_" << newFunction.name() << "_" << kernelIndex;
        newFunction.source->value = ss.str();

        functionDeclStatement &newKernelSmnt = *(
          new functionDeclStatement(&root,
                                    newFunction)
        );
        newKernelSmnt.attributes = kernelSmnt.attributes;
        newKernelSmnt.addFunctionToParentScope();
*/
        function_t &oldFunction = kernelSmnt.function;
        function_t &newFunction = (function_t &)oldFunction.clone();
        std::stringstream ss;
        ss << newFunction.name();
        newFunction.source->value = ss.str();

        functionDeclStatement &newKernelSmnt = *(
            new functionDeclStatement(&root,
                                      newFunction));
        newKernelSmnt.attributes = kernelSmnt.attributes;
        newKernelSmnt.addFunctionToParentScope();

        // Clone for-loop and replace argument variables
        forStatement &newForSmnt = (forStatement &)forSmnt.clone();
        newKernelSmnt.set(newForSmnt);

        const int argc = (int)newFunction.args.size();
        for (int i = 0; i < argc; ++i)
        {
          newForSmnt.replaceVariable(
              *oldFunction.args[i],
              *newFunction.args[i]);
        }

        return &newKernelSmnt;
      }

      void withLauncherLambda::setupKernels()
      {
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             root,
                             kernelSmnts);

        const int kernelCount = (int)kernelSmnts.size();
        for (int i = 0; i < kernelCount; ++i)
        {

          functionDeclStatement &kernelSmnt = (*((functionDeclStatement *)kernelSmnts[i]));
          statementPtrVector outerSmnts;
          findStatementsByAttr(statementType::for_,
                               "outer",
                               kernelSmnt,
                               outerSmnts);

          forStatement &forSmnt = *((forStatement *)outerSmnts[0]);
          forStatement *innerSmnt = getInnerMostInnerLoop(forSmnt);

          if (!innerSmnt)
          {
            success = false;
            forSmnt.printError("No [@inner] for-loop found");
            return;
          }

          statementPtrVector path;
          oklForStatement::getOKLLoopPath(*innerSmnt, path);

          // Create block in case there are duplicate variable names
          blockStatement &block = (*new blockStatement(forSmnt.up, forSmnt.source));

          // Get max count
          int outerCount = 0;
          int innerCount = 0;
          const int pathCount = (int)path.size();
          for (int ii = 0; ii < pathCount; ++ii)
          {
            forStatement &pathSmnt = *((forStatement *)path[ii]);
            oklForStatement oklForSmnt(pathSmnt);
            if (!oklForSmnt.isValid())
            {
              success = false;
              return;
            }
            const bool isOuter = pathSmnt.hasAttribute("outer");
            outerCount += isOuter;
            innerCount += !isOuter;
          }
          // const int outerDims = outerCount;
          // const int innerDims = innerCount;

          for (int ii = 0; ii < pathCount; ++ii)
          {
            forStatement &pathSmnt = *((forStatement *)path[ii]);
            oklForStatement oklForSmnt(pathSmnt);

            block.add(pathSmnt.init->clone(&block));
            const bool isOuter = pathSmnt.hasAttribute("outer");
            outerCount -= isOuter;
            innerCount -= !isOuter;

            const int index = (isOuter
                                   ? (2 - outerCount)
                                   : (2 - innerCount));

            token_t *source = pathSmnt.source;
            const std::string &name = (isOuter
                                           ? "group_range"
                                           : "local_range");
            block.add(
                *(new expressionStatement(
                    &block,
                    setDim(source, name, index,
                           oklForSmnt.getIterationCount()))));
          }

          std::stringstream kernel_range;
          kernel_range << "\n"
                       << "  ::sycl::range<3> local_range{1,1,1};\n"
                       << "  ::sycl::range<3> group_range{1,1,1};\n";

          kernel_range << block.toString();
          kernel_range << "  ::sycl::nd_range<3> kernel_range(local_range*group_range,local_range);\n"
                       << "\n";

          //-----
          // Find shared local variables
          std::stringstream slm_ranges;
          std::stringstream slm_variables;

          statementPtrVector sharedSmnts;
          findStatementsByAttr(statementType::declaration,
                               "shared",
                               kernelSmnt,
                               sharedSmnts);
          for (auto s : sharedSmnts)
          {
            //At this stage we know that we handle a shared local variable declaration
            //So we need to extract the type, the number of elements and the name.
            declarationStatement *decVariable = (declarationStatement *)s;
            variableDeclarationVector decVector = decVariable->declarations;
            for (auto f : decVector)
            {
              arrayVector variable_dimensions = f.variable->vartype.arrays;
              size_t variable_rank = variable_dimensions.size();

              if (variable_rank > 0)
              {
                std::string variable_name = f.variable->name();
                std::string variable_range_name = variable_name + "_range";
                std::string variable_type = f.variable->vartype.name();

                slm_ranges << "  "
                           << "sycl::range<" << variable_rank << "> "
                           << variable_range_name
                           << "(" << variable_dimensions[0].size->toString();
                for (size_t ii{1}; ii < variable_rank; ++ii)
                {
                  slm_ranges << "," << variable_dimensions[ii].size->toString();
                }
                slm_ranges << ");\n";

                // @todo: Check if we need to do something different for scalars!
                // @note: OKL *requires* all shared variables to be arrays.
                // auto XXX = sycl::accessor<type,rank,sycl::access::mode::read_write,sycl::target::local>(range,cgh)
                slm_variables << "      "
                              << "auto " << variable_name << " = "
                              << "sycl::accessor<"
                              << variable_type << ","
                              << variable_rank << ","
                              << "sycl::access::mode::read_write,"
                              << "sycl::access::target::local>"
                              << "(" << variable_range_name << ",h);"
                              << "\n";
                s->removeFromParent();
              }
            }
          }

          slm_ranges << "\n";
          slm_variables << "\n";

          setupOccaFors(kernelSmnt);
          identifierToken *stt = new identifierToken(originSource::builtin, "variable");
          std::string sa = kernel_range.str() + slm_ranges.str();
          sa += "q_->submit([&](sycl::handler &h){\n";
          sa += slm_variables.str();
          sa += "  h.parallel_for(kernel_range, [=] (sycl::nd_item<3> i_dpcpp_iterator)\n";

          identifierNode *strnodea = new identifierNode(stt, sa);
          expressionStatement *exprSmnta = new expressionStatement(&kernelSmnt, *strnodea);
          exprSmnta->hasSemicolon = false;
          kernelSmnt.children.insert(kernelSmnt.children.begin(), exprSmnta);
          identifierToken *stte = new identifierToken(originSource::builtin, "variable");
          std::string sae = ");\n});\n";
          // sae += "q->wait();";
          identifierNode *strnodeae = new identifierNode(stte, sae);
          expressionStatement *exprSmntae = new expressionStatement(&kernelSmnt, *strnodeae);
          exprSmntae->hasSemicolon = false;
          kernelSmnt.children.push_back(exprSmntae);

          if (!success)
            return;
        }
      }

      void withLauncherLambda::setupOccaFors(functionDeclStatement &kernelSmnt)
      {
        statementPtrVector outerSmnts, innerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             kernelSmnt,
                             outerSmnts);
        findStatementsByAttr(statementType::for_,
                             "inner",
                             kernelSmnt,
                             innerSmnts);

        const int outerCount = (int)outerSmnts.size();
        for (int i = 0; i < outerCount; ++i)
        {
          replaceOccaFor(*((forStatement *)outerSmnts[i]));
        }

        const bool applyBarriers = usesBarriers();

        const int innerCount = (int)innerSmnts.size();
        for (int i = 0; i < innerCount; ++i)
        {
          forStatement &innerSmnt = *((forStatement *)innerSmnts[i]);
          // TODO 1.1: Only apply barriers when needed in the last inner-loop
          if (applyBarriers &&
              isOuterMostInnerLoop(innerSmnt))
          {
            addBarriersAfterInnerLoop(innerSmnt);
            if (!success)
              return;
          }

          replaceOccaFor(innerSmnt);
          if (!success)
            return;
        }
      }

      void withLauncherLambda::addBarriersAfterInnerLoop(forStatement &forSmnt)
      {
        statementExprMap exprMap;
        findStatements(exprNodeType::op,
                       forSmnt,
                       writesToShared,
                       exprMap);

        // No need to add barriers
        if (!exprMap.size())
        {
          return;
        }

        statement_t &barrierSmnt = (*(new emptyStatement(forSmnt.up,
                                                         forSmnt.source)));

        identifierToken barrierToken(forSmnt.source->origin,
                                     "barrier");

        barrierSmnt.attributes["barrier"] = (attributeToken_t(*(getAttribute("barrier")),
                                                              barrierToken));

        forSmnt.up->addAfter(forSmnt,
                             barrierSmnt);
      }

      bool withLauncherLambda::writesToShared(exprNode &expr)
      {
        // TODO 1.1: Propertly check read<-->write or write<-->write ordering
        // exprOpNode &opNode = (exprOpNode&) expr;
        // if (!(opNode.opType() & (operatorType::increment |
        //                          operatorType::decrement |
        //                          operatorType::assignment))) {
        //   return false;
        // }

        // Get updated variable
        variable_t *var = expr.getVariable();
        return (var &&
                var->hasAttribute("shared"));
      }

      void withLauncherLambda::replaceOccaFor(forStatement &forSmnt)
      {
        oklForStatement oklForSmnt(forSmnt);

        std::string iteratorName;
        const int loopIndex = oklForSmnt.oklLoopIndex();
        if (oklForSmnt.isOuterLoop())
        {
          iteratorName = getOuterIterator(loopIndex);
        }
        else
        {
          iteratorName = getInnerIterator(loopIndex);
        }

        identifierToken iteratorSource(oklForSmnt.iterator->source->origin,
                                       iteratorName);
        identifierNode iterator(&iteratorSource,
                                iteratorName);

        // Create iterator declaration
        variableDeclaration decl;
        decl.variable = oklForSmnt.iterator;
        decl.value = oklForSmnt.makeDeclarationValue(iterator);

        // Replace for-loops with blocks
        const int childIndex = forSmnt.childIndex();
        blockStatement &blockSmnt = *(new blockStatement(forSmnt.up,
                                                         forSmnt.source));
        blockSmnt.swap(forSmnt);
        blockSmnt.up->children[childIndex] = &blockSmnt;

        // Add declaration before block
        declarationStatement &declSmnt = (*(new declarationStatement(blockSmnt.up,
                                                                     forSmnt.source)));
        declSmnt.declarations.push_back(decl);

        blockSmnt.addFirst(declSmnt);
        delete &forSmnt;
      }

      bool withLauncherLambda::usesBarriers()
      {
        return true;
      }
    } // namespace okl
  }   // namespace lang
} // namespace occa
