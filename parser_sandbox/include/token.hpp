#ifndef OCCA_PARSER_TOKEN_HEADER2
#define OCCA_PARSER_TOKEN_HEADER2

/*
  Comments are replaced by a space ' '
*/

namespace occa {
  namespace lang {
    class token_t;

    class tokenStream {
      const char *start, *end;
      const char *ptr;

    public:
      tokenStream() :
        start(NULL),
        end(NULL),
        ptr(NULL) {}

      tokenStream(const char *start_,
                  const char *end_ = NULL) :
        start(start_),
        ptr(start) {
        end = ((end_ != NULL)
               ? end_
               : (start + strlen(start)));
      }

      tokenStream(const std::string &str) {
        const int chars = (int) str.size();
        if (chars == 0) {
          start = end = ptr = NULL;
          return;
        }
        start = new char[chars + 1];
        ptr = start;
        end = start + chars;
        ::memcpy(start, str.c_str(), chars + 1);
      }

      void clear() {
        ptr = start;
      }

      bool hasNext() {
        skipWhitespace(ptr);
        return (ptr < end);
      }

      void setNext(token_t &token) {
        tokenTrie_t::result_t result = tokenList.getFirst(ptr);
        if (result.success()) {

        }
      }

      void set(const char *c) {
        OCCA_ERROR("Must set pointer between bounds",
                   (start <= c) && (c < end));
        ptr = c;
      }
    };

    class tokenType {
      directive
    };

    class token_t {
    public:
      bool in(tokenStream &stream) {
        if (stream.hasNext()) {
          stream.setNext(*this);
          return true;
        }
        return false;
      };
    };
  }
}

#endif
