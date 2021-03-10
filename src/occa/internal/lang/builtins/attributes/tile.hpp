#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_TILE_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_TILE_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    class variable_t;
    class blockStatement;
    class forStatement;

    namespace okl {
      class oklForStatement;
    }

    namespace attributes {
      class tile : public attribute_t {
       public:
        tile();

        virtual const std::string& name() const;

        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
        bool validArgs(const attributeToken_t &attr) const;
        bool validKwargs(const attributeToken_t &attr) const;

        static bool applyCodeTransformations(blockStatement &root);

        static void setupNewForStatements(attributeToken_t &attr,
                                          okl::oklForStatement &oklForSmnt,
                                          variable_t &blockIter,
                                          forStatement &blockForSmnt,
                                          forStatement &innerForSmnt);

        static void setupBlockForStatement(okl::oklForStatement &oklForSmnt,
                                           exprNode &tileSize,
                                           variable_t &blockIter,
                                           forStatement &blockForSmnt,
                                           forStatement &innerForSmnt);

        static void setupInnerForStatement(okl::oklForStatement &oklForSmnt,
                                           exprNode &tileSize,
                                           variable_t &blockIter,
                                           forStatement &blockForSmnt,
                                           forStatement &innerForSmnt);

        static void setupCheckStatement(attributeToken_t &attr,
                                        okl::oklForStatement &oklForSmnt,
                                        variable_t &blockIter,
                                        forStatement &blockForSmnt,
                                        forStatement &innerForSmnt);

        static void floatOuterLoopUp(forStatement &outerForSmnt);
      };
    }
  }
}

#endif
