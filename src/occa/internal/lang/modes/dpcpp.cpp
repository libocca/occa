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
      dpcppParser::dpcppParser(const occa::properties &settings_)
          : withLauncherLambda(settings_),
            device("SYCL_EXTERNAL", qualifierType::custom),
            shared("__shared__", qualifierType::custom)
      {
        okl::addAttributes(*this);
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
        updateConstToConstant();

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
      }

      std::string dpcppParser::getOuterIterator(const int loopIndex)
      {
        std::string name = "i_dpcpp_iterator.get_group(";
        name += (char)('2' - loopIndex);
        // name +=  (char)('1'-loopIndex);
        name = name.append(")");
        return name;
      }

      std::string dpcppParser::getInnerIterator(const int loopIndex)
      {
        std::string name = "i_dpcpp_iterator.get_local_id(";
        //name += (char)('0'+loopIndex);
        name += (char)('2' - loopIndex);
        name = name.append(")");
        return name;
      }

      void dpcppParser::updateConstToConstant()
      {
        /*        const int childCount = (int) root.children.size();
        for (int i = 0; i < childCount; ++i) {
          statement_t &child = *(root.children[i]);
          if (child.type() != statementType::declaration) {
            continue;
          }
          declarationStatement &declSmnt = ((declarationStatement&) child);
          const int declCount = declSmnt.declarations.size();
          for (int di = 0; di < declCount; ++di) {
            variable_t &var = *(declSmnt.declarations[di].variable);
            if (var.has(const_) && !var.has(typedef_)) {
              var -= const_;
              var += constant;
            }
          }
        }*/
      }

      void dpcppParser::setFunctionQualifiers()
      {
        statementPtrVector funcSmnts;
        findStatementsByType(statementType::functionDecl,
                             root,
                             funcSmnts);

        const int funcCount = (int)funcSmnts.size();
        for (int i = 0; i < funcCount; ++i)
        {
          functionDeclStatement &funcSmnt = (*((functionDeclStatement *)funcSmnts[i]));
          if (funcSmnt.hasAttribute("kernel"))
          {
            continue;
          }
          vartype_t &vartype = funcSmnt.function.returnType;
          vartype.qualifiers.addFirst(vartype.origin(),
                                      device);
        }
      }

      void dpcppParser::setSharedQualifiers()
      {
        /*        statementExprMap exprMap;
        findStatements(statementType::declaration,
                       exprNodeType::variable,
                       root,
                       sharedVariableMatcher,
                       exprMap);

        statementExprMap::iterator it = exprMap.begin();
        while (it != exprMap.end()) {
          declarationStatement &declSmnt = *((declarationStatement*) it->first);
          const int declCount = declSmnt.declarations.size();
          for (int i = 0; i < declCount; ++i) {
            variable_t &var = *(declSmnt.declarations[i].variable);
            if (!var.hasAttribute("shared")) {
              continue;
            }
            var += shared;
          }
          ++it;
        }
*/
        /*	      root.addToScope(*memoryType);
        std::cout<<"DOING SOME TESTS"<<std::endl;
        std::cout<<root.toString()<<std::endl;
        std::cout<<"ENDING SOME TESTS"<<std::endl;
        statementExprMap exprMap;
        findStatements(statementType::declaration,
                       root,
                       sharedVariableMatcher,
                       exprMap);

        //handle shared variables
        for(auto e : exprMap){
        for(auto *v : e.second){
                std::cout<<"ExpressionNode : "<<v->toString()<<std::endl;
                variable_t *var = v->getVariable();
                if(var->hasAttribute("shared")){
                        std::cout<<"Variable has statement Shared : "<<var->name()<<std::endl;
                }
        }
        }
*/
        /*	                     root.addToScope(*memoryType);
        std::cout<<"DOING SOME TESTS"<<std::endl;
//        std::cout<<root.toString()<<std::endl;
        std::cout<<"ENDING SOME TESTS"<<std::endl;

        statementExprMap exprMap;
        findStatements(statementType::all,
                       root,
                       sharedVariableMatcher,
                       exprMap);

        //handle shared variables
        //td::map<statement_t*, exprNodeVector>
        for(auto e : exprMap){
		if(e.first->type()==statementType::declaration){
			//At this stage we know that we handle a shared local variable declaration
			//So we need to extract the type, the number of elements and the name.
			declarationStatement* decVariable = (declarationStatement*)e.first;
			variableDeclarationVector decVector = decVariable->declarations;
			for(auto f : decVector){
				std::cout<<f.variable->name()<<std::endl;//Variable name
				std::cout<<f.variable->vartype.name()<<std::endl;//Variable type
				std::cout<<f.variable->vartype.arrays[0].size->toString()<<std::endl;//Variable type
				std::cout<<f.variable->vartype.arrays[1].size->toString()<<std::endl;//Variable type

				
				//std::cout<<f.value->type()<<std::endl;
			}	
                	//std::cout<<"Statement : "<<e.first->toString()<<std::endl;
                	//std::cout<<"StatementName : "<<e.first->statementName()<<std::endl;
		}
		
        }
*/
      }

      void dpcppParser::addBarriers()
      {

        statementPtrVector statements;
        findStatementsByAttr(statementType::empty,
                             "barrier",
                             root,
                             statements);

        const int count = (int)statements.size();
        for (int i = 0; i < count; ++i)
        {
          // TODO 1.1: Implement proper barriers
          emptyStatement &smnt = *((emptyStatement *)statements[i]);

          statement_t &barrierSmnt = (*(new expressionStatement(
              smnt.up,
              *(new identifierNode(smnt.source,
                                   "i_dpcpp_iterator.barrier(sycl::access::fence_space::local_space)")))));

          smnt.up->addBefore(smnt,
                             barrierSmnt);

          smnt.up->remove(smnt);
          delete &smnt;
        }
      }

      void dpcppParser::setupKernels()
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
          setKernelQualifiers(kernelSmnt);
          if (!success)
            return;
        }
        // const bool includingStd = settings.get("serial/include_std", true);
        strVector headers;
        headers.push_back("include <CL/sycl.hpp>\n");

        const int headerCount = (int)headers.size();
        for (int i = 0; i < headerCount; ++i)
        {
          std::string header = headers[i];
          // TODO 1.1: Remove hack after methods are properly added
          if (0 == i)
          {
            header += "\nusing namespace cl::sycl;";
          }
          directiveToken token(root.source->origin, header);

          root.addFirst(*(new directiveStatement(&root, token)));
        }
      }

      void dpcppParser::setKernelQualifiers(functionDeclStatement &kernelSmnt)
      {
        vartype_t &vartype = kernelSmnt.function.returnType;
        vartype.qualifiers.addFirst(vartype.origin(),
                                    externC);

        function_t &func = kernelSmnt.function;

        const int argCount = (int)func.args.size();

        variable_t queueArg(syclQueuePtr, "q_");

        func.addArgument(queueArg); //const identifierToken &typeToken_, const type_t &type_
        variable_t ndRangeArg(syclNdRangePtr, "ndrange");

        func.addArgument(ndRangeArg); //const identifierToken &typeToken_, const type_t &type_

        for (int i = 0; i < argCount; ++i)
        {
          variable_t arg(*func.removeArgument(0));
          vartype_t &type = arg.vartype;
          if (!(type.isPointerType() || type.referenceToken))
          {
            operatorToken opToken(arg.source->origin, op::bitAnd);
            type.setReferenceToken(&opToken);
          }
          func.addArgument(arg);
        }
      }

      bool dpcppParser::sharedVariableMatcher(exprNode &expr)
      {
        return expr.hasAttribute("shared");
      }
    } // namespace okl
  }   // namespace lang
} // namespace occa
