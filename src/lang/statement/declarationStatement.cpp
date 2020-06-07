#include <occa/lang/statement/declarationStatement.hpp>
#include <occa/lang/statement/blockStatement.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/variable.hpp>

namespace occa {
  namespace lang {
    declarationStatement::declarationStatement(blockStatement *up_,
                                               token_t *source_) :
        statement_t(up_, source_),
        declaredType(false) {}

    declarationStatement::declarationStatement(blockStatement *up_,
                                               const declarationStatement &other) :
      statement_t(up_, other),
      declaredType(other.declaredType) {
      const int count = (int) other.declarations.size();
      if (!count) {
        return;
      }
      declarations.reserve(count);
      for (int i = 0; i < count; ++i) {
        addDeclaration(other.declarations[i].clone());
      }
    }

    declarationStatement::~declarationStatement() {
      if (up) {
        clearDeclarations();
      } else {
        freeDeclarations();
      }
    }

    void declarationStatement::clearDeclarations() {
      const int count = (int) declarations.size();
      for (int i = 0; i < count; ++i) {
        declarations[i].clear();
      }
      declarations.clear();
    }

    void declarationStatement::freeDeclarations() {
      const int count = (int) declarations.size();
      for (int i = 0; i < count; ++i) {
        variable_t *var = declarations[i].variable;
        // The scope has its own typedef copy
        // We have to delete the variable-typedef
        if (up && up->hasDirectlyInScope(var->name())) {
          up->removeFromScope(var->name());
          var = NULL;
        }
        delete var;
        declarations[i].clear();
      }
      declarations.clear();
    }

    statement_t& declarationStatement::clone_(blockStatement *up_) const {
      return *(new declarationStatement(up_, *this));
    }

    int declarationStatement::type() const {
      return statementType::declaration;
    }

    std::string declarationStatement::statementName() const {
      return "declaration";
    }

    bool declarationStatement::addDeclaration(const variableDeclaration &decl,
                                              const bool force) {
      variable_t &var = *(decl.variable);
      bool success = true;
      if (!up) {
        delete &var;
        return false;
      }

      if (var.vartype.has(typedef_)) {
        // Typedef
        declaredType = true;

        const typedef_t *originalTypedef = dynamic_cast<const typedef_t*>(var.vartype.type);

        const bool typedefingStruct = (
          originalTypedef != NULL
          && originalTypedef->baseType.has(struct_)
          && originalTypedef->declaredBaseType
        );

        typedef_t *type = NULL;

        if (typedefingStruct) {
          // Struct typedefs already allocate a new type
          type = (typedef_t*) originalTypedef;
        } else {
          type = new typedef_t(var.vartype);
          if (var.source) {
            type->setSource(*var.source);
          }
        }

        if (var.vartype.type) {
          type->attributes = var.vartype.type->attributes;
        }

        type->attributes.insert(var.attributes.begin(),
                                var.attributes.end());

        success = up->addToScope(*type, force);

        // This type typedef's a struct so we need to add that
        // type to the current scope
        if (success && typedefingStruct) {
          struct_t &structType = *((struct_t*) type->baseType.type);
          success = up->addToScope(structType,
                                   force);
        }

        if (!success) {
          delete type;
        }
      } else if (var.vartype.definesStruct()) {
        // Struct
        declaredType = true;

        success = up->addToScope(var.vartype.type->clone(),
                                 force);
      } else {
        // Variable
        success = up->addToScope(var, force);
      }

      if (success) {
        declarations.push_back(decl);
      } else {
        delete &var;
      }
      return success;
    }

    void declarationStatement::print(printer &pout) const {
      const int count = (int) declarations.size();
      if (!count) {
        return;
      }

      const variableDeclaration &firstDecl = declarations[0];

      // Pretty print newlines around the struct definition
      const bool printNewlines = (
        declaredType
        && firstDecl.variable->vartype.definesStruct()
      );

      if (printNewlines) {
        pout.printNewlines(2);
      }

      pout.printStartIndentation();

      firstDecl.print(pout, declaredType);
      for (int i = 1; i < count; ++i) {
        pout << ", ";
        declarations[i].printAsExtra(pout);
      }
      pout << ';';

      if (printNewlines) {
        pout.printNewlines(2);
      } else {
        pout.printEndNewline();
      }
    }
  }
}
