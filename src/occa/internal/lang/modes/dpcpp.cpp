#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/modes/dpcpp.hpp>
#include <occa/internal/lang/modes/okl.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/expr.hpp>
// #include <stringstream>

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
        setupAtomics();

        if (!success)
          return;
        // Do this last!
        setupKernels();
      }

      std::string dpcppParser::getOuterIterator(const int loopIndex)
      {
        return "item_.get_group(" + occa::toString(dpcppDimensionOrder(loopIndex)) + ")";
      }

      std::string dpcppParser::getInnerIterator(const int loopIndex)
      {
        return "item_.get_local_id(" + occa::toString(dpcppDimensionOrder(loopIndex)) + ")";
      }

      std::string dpcppParser::launchBoundsAttribute(const int innerDims[3])
      {
        std::stringstream ss; 
        ss << "[[sycl::reqd_work_group_size("
           << innerDims[2]
           << ","
           << innerDims[1]
           << ","
           << innerDims[0]
           << ")]]\n";
        return ss.str();
      }

      // @note: As of SYCL 2020 this will need to change from `CL/sycl.hpp` to `sycl.hpp`
      void dpcppParser::setupHeaders()
      {
        root.addFirst(
            *(new directiveStatement(
                &root,
                directiveToken(root.source->origin, "include <CL/sycl.hpp>\n using namespace sycl;\n"))));
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
            .forEach([&](statement_t *smnt)
                     {
                       emptyStatement &emptySmnt = (emptyStatement &)*smnt;

                       statement_t &barrierSmnt = (*(new sourceCodeStatement(
                           emptySmnt.up,
                           emptySmnt.source,
                           "item_.barrier(sycl::access::fence_space::local_space);")));

                       emptySmnt.replaceWith(barrierSmnt);

                       delete &emptySmnt;
                     });
      }

      void dpcppParser::setupKernels()
      {
        root.children
            .filterByStatementType(
                statementType::functionDecl | statementType::function,
                "kernel")
            .forEach([&](statement_t *smnt)
                     {
                       function_t *function;

                       if (smnt->type() & statementType::functionDecl)
                       {
                         functionDeclStatement &k = ((functionDeclStatement &)*smnt);
                         function = &(k.function());

                         migrateLocalDecls(k);
                         if (!success)
                           return;

                         variable_t sycl_nditem(syclNdItem, "item_");

                         variable_t sycl_handler(syclHandler, "handler_");
                         sycl_handler.vartype.setReferenceToken(
                             new operatorToken(sycl_handler.source->origin, op::address));

                         variable_t sycl_ndrange(syclNdRange, "range_");
                         sycl_ndrange += pointer_t();

                         variable_t sycl_queue(syclQueue, "queue_");
                         sycl_queue += pointer_t();

                         function->addArgumentFirst(sycl_ndrange);
                         function->addArgumentFirst(sycl_queue);

                         lambda_t &cg_function = *(new lambda_t(capture_t::byReference));
                         cg_function.addArgument(sycl_handler);

                         lambda_t &sycl_kernel = *(new lambda_t(capture_t::byValue));
                         sycl_kernel.addArgument(sycl_nditem);

                         sycl_kernel.body->swap(k);

                         lambdaNode sycl_kernel_node(sycl_kernel.source, sycl_kernel);

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

                         binaryOpNode cgh_parallelfor(
                             sycl_handler.source,
                             op::dot,
                             variableNode(sycl_handler.source, sycl_handler.clone()),
                             parallelfor_call_node);

                         cg_function.body->add(*(new expressionStatement(nullptr, cgh_parallelfor)));

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

                         k.addFirst(*(new expressionStatement(nullptr, q_submit)));
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
            .forEach([&](statement_t *smnt)
                     {
                       functionDeclStatement &funcDeclSmnt = (functionDeclStatement &)*smnt;

                       if (funcDeclSmnt.hasAttribute("kernel"))
                       {
                         return;
                       }

                       vartype_t &vartype = funcDeclSmnt.function().returnType;
                       vartype.qualifiers.addFirst(vartype.origin(), device);
                     });
      }

      void dpcppParser::setSharedQualifiers()
      {
        statementArray::from(root).nestedForEachDeclaration(
            [&](variableDeclaration &decl, declarationStatement &declSmnt)
            {
              variable_t &var = decl.variable();

              if (var.hasAttribute("shared"))
              {
                auto *shared_value = new dpcppLocalMemoryNode(var.source->clone(),
                                                              var.vartype,
                                                              "item_");

                decl.setValue(shared_value);
                var.vartype.setType(auto_);
                var.vartype.setReferenceToken(var.source);
                var.vartype.arrays.clear();
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

      void dpcppParser::migrateLocalDecls(functionDeclStatement &kernelSmnt)
      {
        statementArray::from(kernelSmnt)
            .nestedForEachDeclaration([&](variableDeclaration &decl, declarationStatement &declSmnt)
                                      {
                                        variable_t &var = decl.variable();
                                        if (var.hasAttribute("shared"))
                                        {
                                          declSmnt.removeFromParent();
                                          kernelSmnt.addFirst(declSmnt);
                                        }
                                      });
      }

      void dpcppParser::setupAtomics()
      {
        success &= attributes::atomic::applyCodeTransformation(
            root,
            transformAtomicBlockStatement,
            transformAtomicBasicExpressionStatement);
      }

      bool dpcppParser::transformAtomicBlockStatement(blockStatement &blockSmnt)
      {
        bool transform_successful{true};
        statementArray::from(blockSmnt)
            .flatFilterByStatementType(statementType::expression)
            .forEach([&](statement_t *smnt)
                     {
                       expressionStatement &exprSmnt = static_cast<expressionStatement &>(*smnt);

                       // if(!transformAtomicBasicExpressionStatement(exprSmnt))
                       // {
                       //   transform_successful = false;
                       //   return;
                       // }
                       transformAtomicBasicExpressionStatement(exprSmnt);
                     });

        return transform_successful;
      }

      bool dpcppParser::transformAtomicBasicExpressionStatement(expressionStatement &exprSmnt)
      {
        expressionStatement &atomicSmnt = dynamic_cast<expressionStatement &>(exprSmnt.clone());

        const opType_t &opType = expr(atomicSmnt.expr).opType();

        exprNode *variable_node{nullptr};
        if (opType & operatorType::unary)
        {
          if (opType & operatorType::leftUnary)
          {
            variable_node = ((leftUnaryOpNode *)atomicSmnt.expr)->value;
          }
          else if (opType & operatorType::rightUnary)
          {
            variable_node = ((rightUnaryOpNode *)atomicSmnt.expr)->value;
          }
        }
        else if (opType & operatorType::binary)
        {
          binaryOpNode &binaryNode = *static_cast<binaryOpNode *>(atomicSmnt.expr);
          variable_node = binaryNode.leftValue;
        }
        else
        {
          atomicSmnt.printError("Unable to transform @atomic code");
          return false;
        }

        variable_t &atomic_var = *(variable_node->getVariable());
        vartype_t atomic_type = atomic_var.vartype;

        auto *atomic_ref = new dpcppAtomicNode(atomic_var.source, atomic_type, *variable_node);
        atomicSmnt.replaceExprNode(variable_node, atomic_ref);

        exprSmnt.replaceWith(atomicSmnt);
        delete &exprSmnt;

        return true;
      }

    } // namespace okl
  }   // namespace lang
} // namespace occa
