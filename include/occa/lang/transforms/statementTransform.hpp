#ifndef OCCA_LANG_TRANSFORMS_STATEMENTTRANSFORM_HEADER
#define OCCA_LANG_TRANSFORMS_STATEMENTTRANSFORM_HEADER

namespace occa {
  namespace lang {
    class parser_t;
    class statement_t;
    class blockStatement;
    class forStatement;
    class ifStatement;
    class elifStatement;
    class whileStatement;
    class switchStatement;

    class statementTransform {
    public:
      bool downToUp;
      int validStatementTypes;

      statementTransform();

      bool apply(statement_t &smnt);

      virtual statement_t* transformStatement(statement_t &smnt) = 0;

      statement_t* transform(statement_t &smnt);

      statement_t* transformBlockStatement(blockStatement &smnt);

      bool transformChildrenStatements(blockStatement &smnt);

      bool transformStatementInPlace(statement_t *&smnt);

      bool transformInnerStatements(blockStatement &smnt);

      bool transformForInnerStatements(forStatement &smnt);

      bool transformIfInnerStatements(ifStatement &smnt);

      bool transformElifInnerStatements(elifStatement &smnt);

      bool transformWhileInnerStatements(whileStatement &smnt);

      bool transformSwitchInnerStatements(switchStatement &smnt);
    };
  }
}

#endif
