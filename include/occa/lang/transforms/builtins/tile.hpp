#ifndef OCCA_LANG_TRANSFORMS_BUILTINS_TILE_HEADER
#define OCCA_LANG_TRANSFORMS_BUILTINS_TILE_HEADER

#include <occa/lang/transforms/statementTransform.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      class oklForStatement;
    }

    namespace transforms {
      class tile : public statementTransform {
      public:
        tile();

        virtual statement_t* transformStatement(statement_t &smnt);

        void setupNewForStatements(attributeToken_t &attr,
                                   okl::oklForStatement &oklForSmnt,
                                   variable_t &blockIter,
                                   forStatement &blockForSmnt,
                                   forStatement &innerForSmnt);

        void setupBlockForStatement(okl::oklForStatement &oklForSmnt,
                                    exprNode &tileSize,
                                    variable_t &blockIter,
                                    forStatement &blockForSmnt,
                                    forStatement &innerForSmnt);

        void setupInnerForStatement(okl::oklForStatement &oklForSmnt,
                                    exprNode &tileSize,
                                    variable_t &blockIter,
                                    forStatement &blockForSmnt,
                                    forStatement &innerForSmnt);

        void setupCheckStatement(attributeToken_t &attr,
                                 okl::oklForStatement &oklForSmnt,
                                 variable_t &blockIter,
                                 forStatement &blockForSmnt,
                                 forStatement &innerForSmnt);
      };

      bool applyTileTransforms(statement_t &smnt);
    }
  }
}

#endif
