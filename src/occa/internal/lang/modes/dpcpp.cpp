#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/modes/dpcpp.hpp>
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
      dpcppParser::dpcppParser(const occa::json &settings_)
          : withLauncher(settings_),
            kernel(externC),
            device("SYCL_EXTERNAL", qualifierType::custom),
            shared("auto", qualifierType::custom)
      {
        okl::addOklAttributes(*this);
      }

      void dpcppParser::onClear()
      {
        launcherClear();
      }

      void dpcppParser::beforePreprocessing()
      {
        preprocessor.addCompilerDefine("OCCA_USING_GPU", "1");
      }

      void dpcppParser::beforeKernelSplit()
      {
        if (!success)
          return;
        addExtensions();

        // updateConstToConstant();

        if (!success)
          return;
        setFunctionQualifiers();

        if (!success)
          return;
        setSharedQualifiers();
      }

      void dpcppParser::afterKernelSplit()
      {
        addBarriers();

        if (!success)
          return;
        setupKernels();

        //  if (!success) return;
        // setupAtomics();
      }

      std::string dpcppParser::getOuterIterator(const int loopIndex)
      {
        std::string name = "i_dpcpp_iterator.get_group(";
        name += occa::toString(dpcppDimensionOrder(loopIndex));
        name += ")";
        return name;
      }

      std::string dpcppParser::getInnerIterator(const int loopIndex)
      {
        std::string name = "i_dpcpp_iterator.get_local_id(";
        name += occa::toString(dpcppDimensionOrder(loopIndex));
        name += ")";
        return name;
      }

      void dpcppParser::addExtensions()
      {
        if (!settings.has("extensions"))
        {
          return;
        }

        occa::json &extensions = settings["extensions"];
        if (!extensions.isObject())
        {
          return;
        }

        // jsonObject &extensionObj = extensions.object();
        // jsonObject::iterator it = extensionObj.begin();
        // while (it != extensionObj.end()) {
        //   const std::string &extension = it->first;
        //   const bool enabled = it->second;
        //   if (enabled) {
        //     root.addFirst(
        //       *(new pragmaStatement(
        //           &root,
        //           pragmaToken(root.source->origin,
        //                       "OPENCL EXTENSION "+ extension + " : enable\n")
        //         ))
        //     );
        //   }
        //   ++it;
        // }
      }

      void dpcppParser::addBarriers()
      {
        statementArray::from(root)
            .flatFilterByStatementType(statementType::empty, "barrier")
            .forEach([&](statement_t *smnt) {
              // TODO 1.1: Implement proper barriers
              emptyStatement &emptySmnt = (emptyStatement &)*smnt;

              statement_t &barrierSmnt = (*(new sourceCodeStatement(
                  emptySmnt.up,
                  emptySmnt.source,
                  "i_dpcpp_iterator.barrier(sycl::access::fence_space::local_space);")));

              emptySmnt.replaceWith(barrierSmnt);

              delete &emptySmnt;
            });
      }

      void dpcppParser::setupKernels()
      {
        statementArray::from(root)
            .flatFilterByStatementType(
                statementType::functionDecl | statementType::function,
                "kernel")
            .forEach([&](statement_t *smnt) {
              function_t *function;

              if (smnt->type() & statementType::functionDecl)
              {
                function = &(((functionDeclStatement *)smnt)->function());

                migrateLocalDecls((functionDeclStatement &)*smnt);
                if (!success)
                  return;
              }
              else
              {
                function = &(((functionStatement *)smnt)->function());
              }

              setKernelQualifiers(*function);
            });
      }

      void dpcppParser::setFunctionQualifiers()
      {
        root.children
            .filterByStatementType(statementType::functionDecl)
            .forEach([&](statement_t *smnt) {
              functionDeclStatement &funcDeclSmnt = (functionDeclStatement &)*smnt;

              // Only add __device__ to non-kernel functions
              if (funcDeclSmnt.hasAttribute("kernel"))
              {
                return;
              }

              vartype_t &vartype = funcDeclSmnt.function().returnType;
              vartype.qualifiers.addFirst(vartype.origin(),
                                          device);
            });
      }

      void dpcppParser::setSharedQualifiers()
      {
        statementArray::from(root)
            .nestedForEachDeclaration([&](variableDeclaration &decl) {
              variable_t &var = decl.variable();
              if (var.hasAttribute("shared"))
              {
                var.add(0, shared);
              }
            });
      }

      void dpcppParser::setKernelQualifiers(function_t &function)
      {
        function.returnType.add(0, kernel);

        variableVector args;
        variable_t queueArg(syclQueuePtr, "q_");
        args.push_back(queueArg);

        // function.addArgument(queueArg); //const identifierToken &typeToken_, const type_t &type_

        variable_t ndRangeArg(syclNdRangePtr, "ndrange");
        args.push_back(ndRangeArg);
        // function.addArgument(ndRangeArg); //const identifierToken &typeToken_, const type_t &type_

        const int argCount = (int)function.args.size();
        for (int ai = 0; ai < argCount; ++ai)
        {
          variable_t arg(*function.removeArgument(0));

          vartype_t &type = arg.vartype;
          if (!(type.isPointerType() || type.referenceToken))
          {
            operatorToken opToken(arg.source->origin, op::bitAnd);
            type.setReferenceToken(&opToken);
          }
          // function.addArgument(arg);
          args.push_back(arg);
        }

        function.addArguments(args);
      }

      void dpcppParser::migrateLocalDecls(functionDeclStatement &kernelSmnt)
      {
        statementArray::from(kernelSmnt)
            .nestedForEachDeclaration([&](variableDeclaration &decl, declarationStatement &declSmnt) {
              variable_t &var = decl.variable();
              if (var.hasAttribute("shared"))
              {
                declSmnt.removeFromParent();
                kernelSmnt.addFirst(declSmnt);
              }
            });
      }

      // static bool transformBlockStatement(blockStatement &blockSmnt)
      // {
      //   return false;
      // }
      // static bool transformBasicExpressionStatement(expressionStatement &exprSmnt)
      // {
      //   return false;
      // }

    } // namespace okl
  }   // namespace lang
} // namespace occa
