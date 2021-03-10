#ifndef OCCA_INTERNAL_LANG_PROCESSINGSTAGES_HEADER
#define OCCA_INTERNAL_LANG_PROCESSINGSTAGES_HEADER

#include <occa/internal/lang/stream.hpp>

namespace occa {
  namespace lang {
    class token_t;

    typedef streamFilter<token_t*>              tokenFilter;
    typedef streamMap<token_t*, token_t*>       tokenMap;
    typedef withInputCache<token_t*, token_t*>  tokenInputCacheMap;
    typedef withOutputCache<token_t*, token_t*> tokenOutputCacheMap;
    typedef withCache<token_t*, token_t*>       tokenCacheMap;

    class newlineTokenFilter : public tokenFilter {
    public:
      newlineTokenFilter();

      virtual tokenMap& clone_() const;
      virtual bool isValid(token_t * const &token);
    };

    class stringTokenMerger : public tokenOutputCacheMap {
    public:
      stringTokenMerger();
      stringTokenMerger(const stringTokenMerger &other);

      virtual tokenMap& clone_() const;
      virtual void fetchNext();
    };

    class externTokenMerger : public tokenCacheMap {
    public:
      externTokenMerger();
      externTokenMerger(const externTokenMerger &other);

      virtual tokenMap& clone_() const;
      virtual void fetchNext();
    };

    class unknownTokenFilter : public tokenFilter {
    public:
      bool printError;

      unknownTokenFilter(const bool printError_);

      tokenMap& clone_() const;

      void setPrintError(const bool printError_);

      virtual bool isValid(token_t * const &token);
    };
  }
}

#endif
