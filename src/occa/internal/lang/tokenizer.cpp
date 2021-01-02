#include <occa/internal/utils/lex.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/tokenizer.hpp>
#include <occa/internal/lang/token.hpp>

namespace occa {
  namespace lang {
    int getEncodingType(const std::string &str) {
      int encoding      = 0;
      int encodingCount = 0;
      const char *c = str.c_str();
      while (*c) {
        int newEncoding = 0;
        switch (*c) {
        case 'u': {
          if (c[1] == '8') {
            newEncoding = encodingType::u8;
            ++c;
          } else {
            newEncoding = encodingType::u;
          }
          break;
        }
        case 'U':
          newEncoding = encodingType::U; break;
        case 'L':
          newEncoding = encodingType::L; break;
        case 'R':
          newEncoding = encodingType::R; break;
        }
        if (!newEncoding ||
            (newEncoding & encoding)) {
          return encodingType::none;
        }
        encoding |= newEncoding;
        ++encodingCount;
        ++c;
      }
      if ((encodingCount == 1) ||
          ((encodingCount == 2) && (encoding & encodingType::R))) {
        return encoding;
      }
      return encodingType::none;
    }

    int getCharacterEncoding(const std::string &str) {
      const int encoding = getEncodingType(str);
      if (!encoding ||
          (encoding & (encodingType::u8 |
                       encodingType::R))) {
        return encodingType::none;
      }
      return encoding;
    }

    int getStringEncoding(const std::string &str) {
      return getEncodingType(str);
    }

    tokenizer_t::tokenizer_t() :
      origin(originSource::string),
      fp(origin.position) {
      setup();
    }

    tokenizer_t::tokenizer_t(const char *root) :
      origin(originSource::string,
             filePosition(root)),
      fp(origin.position) {
      setup();
    }

    tokenizer_t::tokenizer_t(file_t *file_) :
      origin(*file_),
      fp(origin.position) {
      setup();
    }

    tokenizer_t::tokenizer_t(fileOrigin origin_) :
      origin(origin_),
      fp(origin.position) {
      setup();
    }

    tokenizer_t::tokenizer_t(const tokenizer_t &stream) :
      origin(stream.origin),
      fp(origin.position),
      stack(stream.stack) {
      setup();
    }

    tokenizer_t& tokenizer_t::operator = (const tokenizer_t &stream) {
      origin = stream.origin;
      stack  = stream.stack;
      return *this;
    }

    tokenizer_t::~tokenizer_t() {
      clear();
    }

    void tokenizer_t::setup() {
      getOperators(operators);
      operators.freeze();

      // Extract the first characters for all operators
      //   that don't conflict with identifier (e.g. sizeof)
      //   for the shallowPeek
      typedef std::map<char, bool> charMap;
      charMap charcodes;

      const int operatorCount = operators.size();
      for (int i = 0; i < operatorCount; ++i) {
        const operator_t &op = *(operators.values[i]);
        const char c = op.str[0];
        // Only store chars that don't conflict with identifiers
        // This check is done in the peekForIdentifier method
        if (!lex::inCharset(c, charcodes::identifierStart)) {
          charcodes[c] = true;
        }
      }

      // Store the unique char codes in operatorCharcodes
      operatorCharcodes = "";
      charMap::iterator it = charcodes.begin();
      while (it != charcodes.end()) {
        operatorCharcodes += (it->first);
        ++it;
      }
    }

    baseStream<token_t*>& tokenizer_t::clone() const {
      return *(new tokenizer_t(*this));
    }

    void* tokenizer_t::passMessageToInput(const occa::json &props) {
      const std::string inputName = props.get<std::string>("input_name");
      if (inputName == "tokenizer_t") {
        return (void*) this;
      }
      return NULL;
    }

    void tokenizer_t::set(const char *root) {
      clear();
      origin = fileOrigin(originSource::string,
                          filePosition(root));
    }

    void tokenizer_t::set(file_t *file_) {
      clear();
      origin = fileOrigin(*file_);
    }

    void tokenizer_t::clear() {
      lastTokenType = tokenType::none;
      lastNonNewlineTokenType = tokenType::none;

      errors   = 0;
      warnings = 0;

      stack.clear();
      origin.clear();

      tokenList::iterator it = outputCache.begin();
      while (it != outputCache.end()) {
        delete *it;
        ++it;
      }
      outputCache.clear();
    }

    void tokenizer_t::printError(const std::string &message) {
      origin.printError(message);
      ++errors;
    }

    void tokenizer_t::setLine(const int line) {
      fp.line = line;
    }

    bool tokenizer_t::reachedTheEnd() const {
      return ((*fp.start == '\0') &&
              !origin.up);
    }

    bool tokenizer_t::isEmpty() {
      while (!reachedTheEnd() &&
             outputCache.empty()) {
        token_t *token = getToken();

        lastTokenType = token_t::safeType(token);
        if (token) {
          if (lastTokenType != tokenType::newline) {
            lastNonNewlineTokenType = lastTokenType;
          }
          outputCache.push_back(token);
        }
      }
      return outputCache.empty();
    }

    void tokenizer_t::setNext(token_t *&out) {
      if (!isEmpty()) {
        out = outputCache.front();
        outputCache.pop_front();
      } else {
        out = NULL;
      }
    }

    void tokenizer_t::pushSource(const std::string &filename) {
      // Delete tokens and rewind
      if (outputCache.size()) {
        origin = outputCache.front()->origin;
        tokenList::iterator it = outputCache.begin();
        while (it != outputCache.end()) {
          delete *it;
          ++it;
        }
        outputCache.clear();
      }

      // TODO: Use a fileCache
      file_t *file = new file_t(filename);
      origin.push(true,
                  *file,
                  file->content.c_str());
    }

    void tokenizer_t::popSource() {
      OCCA_ERROR("Unable to call tokenizer_t::popSource",
                 origin.up);
      origin.pop();
    }

    void tokenizer_t::push() {
      stack.push_back(origin);
    }

    void tokenizer_t::pop(const bool rewind) {
      OCCA_ERROR("Unable to call pop()",
                 stack.size() > 0);
      OCCA_ERROR("Missed a push() from the previous source",
                 stack.back().up == origin.up);
      if (rewind) {
        origin = stack.back();
      }
      stack.pop_back();
    }

    void tokenizer_t::popAndRewind() {
      pop(true);
    }

    fileOrigin tokenizer_t::popTokenOrigin() {
      const size_t size = strSize();
      fileOrigin tokenOrigin = stack.back();
      tokenOrigin.position.end = tokenOrigin.position.start + size;
      pop();
      return tokenOrigin;
    }

    size_t tokenizer_t::strSize() {
      if (stack.size() == 0) {
        printError("Not able to strSize() without a stack");
        return 0;
      }
      fileOrigin last = stack.back();
      return (fp.start - last.position.start);
    }

    std::string tokenizer_t::str() {
      if (stack.size() == 0) {
        printError("Not able to str() without a stack");
        return "";
      }
      fileOrigin last = stack.back();
      return std::string(last.position.start,
                         fp.start - last.position.start);
    }

    void tokenizer_t::countSkippedLines() {
      if (stack.size() == 0) {
        printError("Not able to countSkippedLines() without a stack");
        return;
      }
      fileOrigin last = stack.back();
      if (last.file != origin.file) {
        printError("Trying to countSkippedLines() across different files");
        return;
      }
      const char *pos = last.position.start;
      const char *end = fp.start;
      while (pos < end) {
        if (*pos == '\\') {
          if (fp.start[1] == '\n') {
            fp.lineStart = fp.start + 2;
            ++fp.line;
          }
          pos += 1 + (pos[1] != '\0');
          continue;
        }
        if (*pos == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++pos;
      }
    }

    void tokenizer_t::skipTo(const char delimiter) {
      while (*fp.start != '\0') {
        if (*fp.start == '\\') {
          if (fp.start[1] == '\n') {
            fp.lineStart = fp.start + 2;
            ++fp.line;
          }
          fp.start += 1 + (fp.start[1] != '\0');
          continue;
        }
        if (*fp.start == delimiter) {
          return;
        }
        if (*fp.start == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++fp.start;
      }
    }

    void tokenizer_t::skipTo(const char *delimiters) {
      while (*fp.start != '\0') {
        if (*fp.start == '\\') {
          if (fp.start[1] == '\n') {
            fp.lineStart = fp.start + 2;
            ++fp.line;
          }
          fp.start += 1 + (fp.start[1] != '\0');
          continue;
        }
        if (lex::inCharset(*fp.start, delimiters)) {
          return;
        }
        if (*fp.start == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++fp.start;
      }
    }

    void tokenizer_t::skipFrom(const char *delimiters) {
      while (*fp.start != '\0') {
        if (*fp.start == '\\') {
          if (fp.start[1] == '\n') {
            fp.lineStart = fp.start + 2;
            ++fp.line;
          }
          fp.start += 1 + (fp.start[1] != '\0');
          continue;
        }
        if (!lex::inCharset(*fp.start, delimiters)) {
          break;
        }
        if (*fp.start == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++fp.start;
      }
    }

    void tokenizer_t::skipWhitespace() {
      skipFrom(charcodes::whitespaceNoNewline);
    }

    int tokenizer_t::peek() {
      int type = shallowPeek();
      if (type & tokenType::identifier) {
        type = peekForIdentifier();
      } else if (type & tokenType::op) {
        type = peekForOperator();
      }
      return type;
    }

    int tokenizer_t::shallowPeek() {
      skipWhitespace();

      const char c = *fp.start;
      if (c == '\0') {
        return tokenType::none;
      }

      const char *pos = fp.start;
      const bool isPrimitive = (
        primitive::load(pos, false).type != occa::primitiveType::none
      );

      // Primitive must be checked before identifiers and operators since:
      //   - true/false
      //   - Operators can start with a . (for example, .01)
      // However, make sure we aren't parsing an identifier:
      //   - true_var
      //   - false_case
      if (isPrimitive && !lex::inCharset(*pos, charcodes::identifierStart)) {
        return tokenType::primitive;
      }
      if (lex::inCharset(c, charcodes::identifierStart)) {
        return tokenType::identifier;
      }
      if (lex::inCharset(c, operatorCharcodes.c_str())) {
        return tokenType::op;
      }
      if (c == '\n') {
        return tokenType::newline;
      }
      if (c == '"') {
        return tokenType::string;
      }
      if (c == '\'') {
        return tokenType::char_;
      }
      return tokenType::none;
    }

    int tokenizer_t::peekForIdentifier() {
      push();

      // Go through the identifier keys
      ++fp.start;
      skipFrom(charcodes::identifier);

      const std::string identifier = str();

      int type = shallowPeek();
      popAndRewind();

      // sizeof, new, delete, throw
      if (operators.has(identifier)) {
        return tokenType::op;
      };

      // [u8R]"foo" or [u]'foo'
      if (type & tokenType::string) {
        const int encoding = getStringEncoding(identifier);
        if (encoding) {
          return (tokenType::string |
                  (encoding << tokenType::encodingShift));
        }
      }
      if (type & tokenType::char_) {
        const int encoding = getCharacterEncoding(identifier);
        if (encoding) {
          return (tokenType::char_ |
                  (encoding << tokenType::encodingShift));
        }
      }
      return tokenType::identifier;
    }

    int tokenizer_t::peekForOperator() {
      push();
      operatorTrie::result_t result = operators.getLongest(fp.start);
      if (!result.success()) {
        printError("Not able to parse operator");
        popAndRewind();
        return tokenType::none;
      }
      popAndRewind();
      return tokenType::op;
    }

    void tokenizer_t::getIdentifier(std::string &value) {
      if (!lex::inCharset(*fp.start, charcodes::identifierStart)) {
        return;
      }
      push();
      ++fp.start;
      skipFrom(charcodes::identifier);
      value = str();
      pop();
    }

    bool tokenizer_t::getString(std::string &value,
                                const int encoding) {
      if (encoding & encodingType::R) {
        getRawString(value);
        return true;
      }
      if (*fp.start != '"') {
        return false;
      }
      // Skip ["]
      ++fp.start;

      push();
      skipTo("\"\n");

      // Handle error outside of here
      if (*fp.start == '\n') {
        printError("Not able to find a closing \"");
        pop();
        return false;
      }

      value = unescape(str(), '"');
      pop();
      ++fp.start;

      return true;
    }

    void tokenizer_t::getRawString(std::string &value) {
      // TODO: Keep the raw delimiter(s)
      if (*fp.start != '"') {
        return;
      }
      push();
      ++fp.start; // Skip "
      push();

      // Find delimiter
      skipTo("(\n");
      if (*fp.start == '\n') {
        pop();
        popAndRewind();
        return;
      }

      // Find end pattern
      std::string end;
      end += ')';
      end += str();
      end += '"';
      pop();
      ++fp.start; // Skip (
      push();

      // Find end match
      const int chars = (int) end.size();
      const char *m   = end.c_str();
      int mi;
      while (*fp.start != '\0') {
        for (mi = 0; mi < chars; ++mi) {
          if (fp.start[mi] != m[mi]) {
            break;
          }
        }
        if (mi == chars) {
          break;
        }
        if (*fp.start == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++fp.start;
      }

      // Make sure we found delimiter
      if (*fp.start == '\0') {
        pop();
        popAndRewind();
        return;
      }
      value = str();
      fp.start += chars;
    }

    token_t* tokenizer_t::getToken() {
      if (reachedTheEnd()) {
        return NULL;
      }

      skipWhitespace();

      // Check if file finished
      bool finishedSource = (*fp.start == '\0');
      while ((*fp.start == '\0') &&
             origin.up) {
        popSource();
        skipWhitespace();
        finishedSource = true;
      }
      if (finishedSource) {
        push();
        return new newlineToken(popTokenOrigin());
      }

      int type = peek();
      if (type & tokenType::identifier) {
        return getIdentifierToken();
      }
      if (type & tokenType::primitive) {
        return getPrimitiveToken();
      }
      if (type & tokenType::op) {
        return getOperatorToken();
      }
      if (type & tokenType::newline) {
        push();
        ++fp.start;
        ++fp.line;
        fp.lineStart = fp.start;
        return new newlineToken(popTokenOrigin());
      }
      if (type & tokenType::char_) {
        return getCharToken(tokenType::getEncoding(type));
      }
      if (type & tokenType::string) {
        return getStringToken(tokenType::getEncoding(type));
      }

      push();
      ++fp.start;
      return new unknownToken(popTokenOrigin());
    }

    token_t* tokenizer_t::getIdentifierToken() {
      if (!lex::inCharset(*fp.start, charcodes::identifierStart)) {
        printError("Not able to parse identifier");
        return NULL;
      }

      push();
      std::string value;
      getIdentifier(value);

      return new identifierToken(popTokenOrigin(),
                                 value);
    }

    token_t* tokenizer_t::getPrimitiveToken() {
      push();
      primitive value = primitive::load(fp.start);
      if (value.isNaN()) {
        printError("Not able to parse primitive");
        popAndRewind();
        return NULL;
      }
      const std::string strValue = str();
      countSkippedLines();
      return new primitiveToken(popTokenOrigin(),
                                value,
                                strValue);
    }

    token_t* tokenizer_t::getOperatorToken() {
      push();
      operatorTrie::result_t result = operators.getLongest(fp.start);
      if (!result.success()) {
        printError("Not able to parse operator");
        return NULL;
      }

      const operator_t &op = *(result.value());

      if (op.opType & operatorType::comment) {
        if (op.opType == operatorType::lineComment) {
          return getLineCommentToken();
        }
        else if (op.opType == operatorType::blockCommentStart) {
          return getBlockCommentToken();
        }
      }

      fp.start += result.length; // Skip operator
      return new operatorToken(popTokenOrigin(),
                               op);
    }

    token_t* tokenizer_t::getLineCommentToken() {
      int spacingType = spacingType_t::none;

      if (
        // Don't double the newlines
        (lastNonNewlineTokenType != tokenType::comment)
        // Shift by 1 to undo the '/' operator peek
        && (1 < origin.emptyLinesBefore(fp.start - 1))
      ) {
        spacingType |= spacingType_t::left;
      }

      push();
      skipTo('\n');

      const std::string comment = str();

      pop();

      if (1 < origin.emptyLinesAfter(fp.start + 1)) {
            spacingType |= spacingType_t::right;
      }

      return new commentToken(popTokenOrigin(),
                              comment,
                              spacingType);
    }

    token_t* tokenizer_t::getBlockCommentToken() {
      int spacingType = spacingType_t::none;

      if (
        // Don't double the newlines
        (lastNonNewlineTokenType != tokenType::comment)
        // Shift by 2 to undo the '/*' operator peek
        && (1 < origin.emptyLinesBefore(fp.start - 2))
      ) {
        spacingType |= spacingType_t::left;
      }

      push();

      bool finishedComment = false;
      while (!finishedComment && *fp.start != '\0') {
        skipTo('*');
        if (*fp.start == '*') {
          ++fp.start;
          if (*fp.start == '/') {
            ++fp.start;
            finishedComment = true;
          }
        }
      }

      const std::string comment = str();
      pop();

      if (1 <= origin.emptyLinesAfter(fp.start)) {
        spacingType |= spacingType_t::right;
      }

      return new commentToken(popTokenOrigin(),
                              comment,
                              spacingType);
    }

    token_t* tokenizer_t::getStringToken(const int encoding) {
      push();

      if (encoding) {
        std::string encodingStr;
        getIdentifier(encodingStr);
      }

      if (*fp.start != '"') {
        printError("Not able to parse string");
        pop();
        return NULL;
      }

      std::string value, udf;
      if (!getString(value, encoding)) {
        return NULL;
      }

      if (*fp.start == '_') {
        getIdentifier(udf);
      }

      return new stringToken(popTokenOrigin(),
                             encoding, value, udf);
    }

    token_t* tokenizer_t::getCharToken(const int encoding) {
      push();

      if (encoding) {
        std::string encodingStr;
        getIdentifier(encodingStr);
      }
      if (*fp.start != '\'') {
        printError("Not able to parse char");
        pop();
        return NULL;
      }

      ++fp.start; // Skip '
      push();
      skipTo("'\n");
      if (*fp.start == '\n') {
        printError("Not able to find a closing '");
        popAndRewind();
        pop();
        return NULL;
      }
      const std::string value = unescape(str(), '\'');
      ++fp.start;
      pop();

      std::string udf;
      if (*fp.start == '_') {
        getIdentifier(udf);
      }

      return new charToken(popTokenOrigin(),
                           encoding, value, udf);
    }

    int tokenizer_t::peekForHeader() {
      int type = shallowPeek();
      if (type & tokenType::op) {
        push();
        operatorTrie::result_t result = operators.getLongest(fp.start);
        popAndRewind();
        if (result.success() &&
            (result.value()->opType & operatorType::lessThan)) {
          return tokenType::systemHeader;
        }
      }
      else if (type & tokenType::string) {
        return tokenType::header;
      }
      return tokenType::none;
    }

    bool tokenizer_t::loadingQuotedHeader() {
      // Assumes we are loading a header
      int type = shallowPeek();

      return (type & tokenType::string);
    }

    bool tokenizer_t::loadingAngleBracketHeader() {
      // Assumes we are loading a header
      int type = shallowPeek();

      return (type & tokenType::op);
    }

    std::string tokenizer_t::getHeader() {
      bool isQuoted = loadingQuotedHeader();
      bool isAngleBracket = loadingAngleBracketHeader();

      if (!isQuoted && !isAngleBracket) {
        printError("Not able to parse header");
        return NULL;
      }

      // Push after in case of whitespace
      push();

      // (Quoted) #include "..."
      if (isQuoted) {
        std::string value;
        getString(value);
        return value;
      }

      // (Angle bracket) #include <...>
      ++fp.start; // Skip <
      push();
      skipTo(">\n");
      if (*fp.start == '\n') {
        printError("Not able to find a closing >");
        pop();
        pop();
        return NULL;
      }
      const std::string header = str();
      pop();
      ++fp.start; // Skip >
      return header;
    }

    void tokenizer_t::setOrigin(const int line,
                                const std::string &filename) {
      // TODO: Needs to create a new file instance to avoid
      //         renaming all versions of *file
      origin.position.line  = line;
      origin.file->filename = filename;
      // Update all cached tokens
      tokenList::iterator it = outputCache.begin();
      while (it != outputCache.end()) {
        token_t *token = *it;
        token->origin.position.line  = line;
        token->origin.file->filename = filename;
        ++it;
      }
    }

    tokenVector tokenizer_t::tokenize(const std::string &source) {
      tokenVector tokens;
      fileOrigin origin_ = originSource::string;
      tokenize(tokens, origin_, source);
      return tokens;
    }

    void tokenizer_t::tokenize(tokenVector &tokens,
                               fileOrigin origin_,
                               const std::string &source) {
      // TODO: Make a string file_t
      fileOrigin fakeOrigin(*origin_.file,
                            source.c_str());

      tokenizer_t tstream(fakeOrigin);
      // Fill tokens
      token_t *token;
      while (!tstream.isEmpty()) {
        tstream.setNext(token);
        tokens.push_back(token);
      }
    }
  }
}
