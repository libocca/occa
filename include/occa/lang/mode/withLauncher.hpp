#ifndef OCCA_LANG_MODES_WITHLAUNCHER_HEADER
#define OCCA_LANG_MODES_WITHLAUNCHER_HEADER

#include <occa/lang/parser.hpp>
#include <occa/lang/mode/serial.hpp>
#include <occa/lang/builtins/transforms/finders.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class withLauncher : public parser_t {
      public:
        serialParser hostParser;

        withLauncher(const occa::properties &settings_ = occa::properties());

        //---[ Public ]-----------------
        virtual bool succeeded() const;

        void writeHostSourceToFile(const std::string &filename) const;
        //==============================

        void hostClear();

        void afterParsing();

        virtual void beforeKernelSplit();
        virtual void afterKernelSplit();

        void setOKLLoopIndices();

        void setupHostParser();

        void removeHostOuterLoops(functionDeclStatement &kernelSmnt);

        bool isOuterMostOuterLoop(forStatement &forSmnt);

        bool isOuterMostInnerLoop(forStatement &forSmnt);

        bool isOuterMostOklLoop(forStatement &forSmnt,
                                const std::string &attr);

        void setKernelLaunch(functionDeclStatement &kernelSmnt,
                             forStatement &forSmnt,
                             const int kernelIndex);

        void setupHostKernelArgs(functionDeclStatement &kernelSmnt);
        void setupHostHeaders();

        int getInnerLoopLevel(forStatement &forSmnt);

        forStatement* getInnerMostInnerLoop(forStatement &forSmnt);

        exprNode& setDim(token_t *source,
                         const std::string &name,
                         const int index,
                         exprNode *value);

        void splitKernels();

        void splitKernel(functionDeclStatement &kernelSmnt);

        statement_t* extractLoopAsKernel(functionDeclStatement &kernelSmnt,
                                         forStatement &forSmnt,
                                         const int kernelIndex);

        void setupKernels();

        void setupOccaFors(functionDeclStatement &kernelSmnt);

        void addBarriersAfterInnerLoop(forStatement &forSmnt);

        static bool writesToShared(exprNode &expr);

        void replaceOccaFor(forStatement &forSmnt);

        virtual bool usesBarriers();

        virtual std::string getOuterIterator(const int loopIndex) = 0;
        virtual std::string getInnerIterator(const int loopIndex) = 0;
      };
    }
  }
}

#endif
