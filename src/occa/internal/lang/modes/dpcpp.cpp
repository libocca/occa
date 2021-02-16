#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/modes/dpcpp.hpp>
#include <occa/internal/lang/modes/okl.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/expr.hpp>
// #include <occa/internal/lang/expr/lambdaNode.hpp>

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
        // setupHeaders();

        // if (!success)
          // return;
        // addExtensions();

        // if(!success)
        //   return;
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
        setupHeaders();

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

      // @note: As of SYCL 2020 this will need to change from `CL/sycl.hpp` to `sycl.hpp`
      void dpcppParser::setupHeaders()
      {
        root.addFirst(
            *(new directiveStatement(
                &root,
                directiveToken(root.source->origin, "include <CL/sycl.hpp>"))));
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

        // @todo: Enable dpcpp extensions

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
        root.children
            .filterByStatementType(
              statementType::functionDecl | statementType::function,
              "kernel"
            )
            .forEach([&](statement_t *smnt) {
              function_t * function;

              if (smnt->type() & statementType::functionDecl)
              {
                functionDeclStatement &k = ((functionDeclStatement &)*smnt);
                function = &(k.function());

                variable_t sycl_queue(syclQueue, "q_");
                sycl_queue += pointer_t();

                variable_t sycl_ndrange(syclNdRange, "ndrange_");
                sycl_ndrange += pointer_t();
                
                function->addArgumentFirst(sycl_ndrange);
                function->addArgumentFirst(sycl_queue);

                blockStatement &outer_statement = *(new blockStatement(&root, k.source->clone()));
                blockStatement &inner_statement = *(new blockStatement(&outer_statement, k.source->clone()));
                
                //-----
                migrateLocalDecls(k, inner_statement);
                if (!success)
                  return;

                //-------
                // @todo Refactor into separate functions
                variable_t sycl_nditem(syclNdItem, "it_");

                lambda_t & sycl_kernel = * (new lambda_t(capture_t::byValue, k));
                sycl_kernel.addArgument(sycl_nditem);
                lambdaNode sycl_kernel_node(sycl_kernel.source,sycl_kernel);

                leftUnaryOpNode sycl_ndrange_node(
                    sycl_ndrange.source,
                    op::dereference,
                    variableNode(sycl_ndrange.source, sycl_ndrange.clone()));

                exprNodeVector parallelfor_args;
                parallelfor_args.push_back(&sycl_ndrange_node);
                parallelfor_args.push_back(&sycl_kernel_node);

                identifierNode parallelfor_node(
                    new identifierToken(originSource::builtin, "parfor"),
                    "parallel_for");

                callNode parallelfor_call_node(
                    parallelfor_node.token,
                    parallelfor_node,
                    parallelfor_args);

                variable_t sycl_handler(syclHandler, "cgh_");
                sycl_handler.vartype.setReferenceToken(
                    new operatorToken(sycl_handler.source->origin, op::address));

                binaryOpNode cgh_parallelfor(
                    sycl_handler.source,
                    op::dot,
                    variableNode(sycl_handler.source, sycl_handler.clone()),
                    parallelfor_call_node);

                inner_statement.add(*(new expressionStatement(nullptr, cgh_parallelfor)));
                //
                //-------
                // @todo Refactor into separate functions

                lambda_t & cg_function = *(new lambda_t(capture_t::byReference, inner_statement));
                cg_function.addArgument(sycl_handler);
                lambdaNode cg_function_node(cg_function.source, cg_function);

                exprNodeVector submit_args;
                submit_args.push_back(&cg_function_node);

                identifierNode submit_node(
                    new identifierToken(originSource::builtin, "qsub"),
                    "submit");

                callNode submit_call_node(
                    submit_node.token,
                    submit_node,
                    submit_args);

                binaryOpNode q_submit(
                    sycl_queue.source,
                    op::arrow,
                    variableNode(sycl_queue.source, sycl_queue.clone()),
                    submit_call_node);

                outer_statement.addFirst(*(new expressionStatement(nullptr, q_submit)));

                k.set(outer_statement);
                //------
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
              vartype.qualifiers.addFirst(vartype.origin(),device);
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

        for (auto arg : function.args)
        {
          vartype_t &type = arg->vartype;
          type = type.flatten();
          if (!(type.isPointerType() || type.referenceToken))
          {
            type.setReferenceToken(arg->source);
          }
        }
      }

      void dpcppParser::migrateLocalDecls(functionDeclStatement &fromSmnt,
                                          blockStatement &toSmnt)
      {
        statementArray::from(fromSmnt)
            .nestedForEachDeclaration([&](variableDeclaration &decl, declarationStatement &declSmnt) {
              variable_t &var = decl.variable();
              if (var.hasAttribute("shared"))
              {
                declSmnt.removeFromParent();
                toSmnt.addFirst(declSmnt);
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
