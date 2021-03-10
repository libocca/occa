#include <occa/internal/lang/expr/binaryOpNode.hpp>
#include <occa/internal/lang/expr/emptyNode.hpp>
#include <occa/internal/lang/expr/identifierNode.hpp>
#include <occa/internal/lang/loaders/attributeLoader.hpp>
#include <occa/internal/lang/loaders/variableLoader.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statementContext.hpp>
#include <occa/internal/lang/token.hpp>

namespace occa {
  namespace lang {
    attributeLoader_t::attributeLoader_t(tokenContext_t &tokenContext_,
                                         statementContext_t &smntContext_,
                                         parser_t &parser_,
                                         nameToAttributeMap &attributeMap_) :
      tokenContext(tokenContext_),
      smntContext(smntContext_),
      parser(parser_),
      attributeMap(attributeMap_),
      success(true) {}

    bool attributeLoader_t::loadAttributes(attributeTokenMap &attrs) {
      while (success &&
             (token_t::safeOperatorType(tokenContext[0]) & operatorType::attribute)) {
        loadAttribute(attrs);
        if (!success) {
          break;
        }
      }
      return success;
    }

    void attributeLoader_t::loadAttribute(attributeTokenMap &attrs) {
      // Skip [@] token
      ++tokenContext;

      if (!(tokenContext[0]->type() & tokenType::identifier)) {
        tokenContext.printError("Expected an attribute name");
        success = false;
        return;
      }

      identifierToken &nameToken = *((identifierToken*) tokenContext[0]);
      ++tokenContext;

      attribute_t *attribute = getAttribute(attributeMap,
                                            nameToken.value);
      if (!attribute) {
        nameToken.printError("Unknown attribute");
        success = false;
        return;
      }

      tokenRangeVector argRanges;
      const bool hasArgs = (
        token_t::safeOperatorType(tokenContext[0]) & operatorType::parenthesesStart
      );
      if (hasArgs) {
        tokenContext.pushPairRange();
        getArgumentRanges(tokenContext, argRanges);
      }
      if (!success) {
        if (hasArgs) {
          tokenContext.popAndSkip();
        }
        return;
      }

      attributeToken_t attr(*attribute, nameToken);
      setAttributeArgs(attr, argRanges);
      if (hasArgs) {
        tokenContext.popAndSkip();
      }
      if (success) {
        attrs[nameToken.value] = attr;
        success = attribute->isValid(attr);
      }
    }

    void attributeLoader_t::setAttributeArgs(attributeToken_t &attr,
                                             tokenRangeVector &argRanges) {
      const int args = (int) argRanges.size();
      bool foundNamedArg = false;
      for (int i = 0; i < args; ++i) {
        attributeArg_t arg;
        tokenContext.push(argRanges[i].start,
                          argRanges[i].end);

        if (!tokenContext.size()) {
          arg.expr = new emptyNode();
          attr.args.push_back(arg);
          tokenContext.popAndSkip();
          continue;
        }

        // Load args
        loadAttributes(arg.attributes);
        if (!tokenContext.size()) {
          arg.expr = new emptyNode();
          attr.args.push_back(arg);
          tokenContext.popAndSkip();
          continue;
        }

        // Get argument
        arg.expr = tokenContext.parseExpression(smntContext,
                                                parser);
        if (!success) {
          tokenContext.pop();
          arg.clear();
          return;
        }

        std::string argName;
        // Check for
        // |---[=] (binary)
        // |   |
        // |   |---[argName] (identifier)
        // |   |
        // |   |---[arg] (?)
        if (arg.expr->type() & exprNodeType::binary) {
          binaryOpNode &equalsNode = arg.expr->to<binaryOpNode>();
          if ((equalsNode.opType() & operatorType::assign) &&
              (equalsNode.leftValue->type() & exprNodeType::identifier)) {
            argName = equalsNode.leftValue->to<identifierNode>().value;
            arg.expr = equalsNode.rightValue->clone();
            delete &equalsNode;
          }
        }

        if (!argName.size() &&
            foundNamedArg) {
          tokenContext.printError("All arguments after a named argument"
                                  " must also be named");
          success = false;
          tokenContext.pop();
          arg.clear();
          return;
        }

        if (!argName.size()) {
          attr.args.push_back(arg);
        } else {
          attr.kwargs[argName] = arg;
        }
        tokenContext.popAndSkip();
      }
    }

    bool loadAttributes(tokenContext_t &tokenContext,
                        statementContext_t &smntContext,
                        parser_t &parser,
                        nameToAttributeMap &attributeMap,
                        attributeTokenMap &attrs) {
      attributeLoader_t loader(tokenContext, smntContext, parser, attributeMap);
      return loader.loadAttributes(attrs);
    }

    attribute_t* getAttribute(nameToAttributeMap &attributeMap,
                              const std::string &name) {
      nameToAttributeMap::iterator it = attributeMap.find(name);
      if (it == attributeMap.end()) {
        return NULL;
      }
      return it->second;
    }
  }
}
