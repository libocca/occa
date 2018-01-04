#if 0
#ifndef OCCA_PARSER_TOKEN_HEADER2
#define OCCA_PARSER_TOKEN_HEADER2

#include <iostream>

#include "occa/tools/gc.hpp"

/*
  Comments are replaced by a space ' '

  \\n -> nothing

  \n is guaranteed by the end of a file
  \s -> one space

  "a" "b" -> "ab"

  Make tokens
*/

class fileInfo {
public:
  std::string path;
  std::string source;

  fileInfo(const std::string &path_);
};

class fileInfoDB {
private:
  std::map<std::string, int> pathToID;
  std::map<int, fileInfo*> idToPath;
  int currentID;

public:
  fileInfoDB();
  fileInfoDB();
  ~fileInfoDB();

  const std::string& get(const std::string &path);
  const std::string& get(const int id);
};

namespace occa {
  namespace lang {
    class tokenStream;

    class token_t {
    public:
    };

    class tokenStream {
    private:
      char *start, *end;
      char *ptr;

    public:
      tokenStream();

      tokenStream(const char *start_,
                  const char *end_ = NULL);

      tokenStream(const std::string &str);

      tokenStream(const tokenStream &stream);

      virtual void destructor();

      void load(const char *start_,
                const char *end_ = NULL);

      void clear();

      bool hasNext();

      bool get(token_t &token);
    };
  }
}

#endif
#endif