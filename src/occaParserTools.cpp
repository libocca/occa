#include "occaParserTools.hpp"

namespace occa {
  namespace parserNamespace {
    //---[ Helper Functions ]-----------------------
    std::string obfuscate(const std::string s1){
      return "__occa__variable__" + s1 + "__";
    }

    std::string obfuscate(const std::string s1, const std::string s2){
      return "__occa__variable__" + s1 + "__" + s2 + "__";
    }

    bool stringsAreEqual(const char *cStart, const size_t chars,
                         const char *c2){
      for(size_t c = 0; c < chars; ++c){
        if(cStart[c] != c2[c])
          return false;

        if(c2[c] == '\0')
          return false;
      }

      return true;
    }

    bool charIsIn(const char c, const char *delimeters){
      while((*delimeters) != '\0')
        if(c == *(delimeters++))
          return true;

      return false;
    }

    bool charIsIn2(const char *c, const char *delimeters){
      const char c0 = c[0];
      const char c1 = c[1];

      while((*delimeters) != '\0'){
        if((c0 == delimeters[0]) && (c1 == delimeters[1]))
          return true;

        delimeters += 2;
      }

      return false;
    }

    bool charIsIn3(const char *c, const char *delimeters){
      const char c0 = c[0];
      const char c1 = c[1];
      const char c2 = c[2];

      while((*delimeters) != '\0'){
        if((c0 == delimeters[0]) && (c1 == delimeters[1]) && (c2 == delimeters[2]))
          return true;

        delimeters += 3;
      }

      return false;
    }

    char upChar(const char c){
      if(('a' <= c) && (c <= 'z'))
        return ((c + 'A') - 'a');

      return c;
    }

    char downChar(const char c){
      if(('A' <= c) && (c <= 'Z'))
        return ((c + 'a') - 'A');

      return c;
    }

    std::string upString(const char *c, const int chars){
      std::string ret(c, chars);

      for(int i = 0; i < chars; ++i)
        ret[i] = upChar(ret[i]);

      return ret;
    }

    std::string upString(const std::string &s){
      return upString(s.c_str(), s.size());
    }

    bool upStringCheck(const std::string &a,
                       const std::string &b){
      const int aSize = a.size();
      const int bSize = b.size();

      if(aSize != bSize)
        return false;

      for(int i = 0; i < aSize; ++i){
        if(upChar(a[i]) != upChar(b[i]))
          return false;
      }

      return true;
    }

    bool isWhitespace(const char c){
      return charIsIn(c, whitespace);
    }

    void skipWhitespace(const char *&c){
      while(charIsIn(*c, whitespace) && (*c != '\0'))
        ++c;
    }

    void skipToWhitespace(const char *&c){
      while(!charIsIn(*c, whitespace) && (*c != '\0'))
        ++c;
    }

    bool isAString(const char *c){
      return ((*c == '\'') || (*c == '"'));
    }

    bool isAnInt(const char *c){
      const char *cEnd = c;
      skipToWhitespace(cEnd);

      while(c < cEnd){
        if(('0' > *c) || (*c > '9'))
          return false;

        ++c;
      }

      return true;
    }

    bool isAFloat(const char *c){
      if(('0' <= *c) && (*c <= '9'))
        return true;

      if(((c[0] == '+') || (c[0] == '-')) &&
         (c[1] != '\0') &&
         ((c[1] == '.') || (('0' <= c[1]) && (c[1] <= '9'))))
        return true;

      if((c[0] == '.')     &&
         (c[1] != '\0') &&
         ('0' <= c[1]) && (c[1] <= '9'))
        return true;

      return false;
    }

    bool isANumber(const char *c){
      return (isAnInt(c) || isAFloat(c));
    }

    void skipInt(const char *&c){
      while((*c != '\0') &&
            ('0' <= *c) && (*c <= '9'))
        ++c;
    }

    void skipNumber(const char *&c, const bool parsingC){
      if(!parsingC){
        skipFortranNumber(c);
        return;
      }

      if((*c == '+') || (*c == '-'))
        ++c;

      skipInt(c);

      if(*c == '.'){
        ++c;

        skipInt(c);
      }

      if(*c == 'e'){
        ++c;

        if((*c == '+') || (*c == '-'))
          ++c;

        skipInt(c);
      }

      if((*c == 'f') || (*c == 'F') ||
         (*c == 'l') || (*c == 'L'))
        ++c;
    }

    void skipFortranNumber(const char *&c){
      if((*c == '+') || (*c == '-'))
        ++c;

      skipInt(c);

      if(*c == '.'){
        ++c;

        skipInt(c);
      }

      const char nextChar = upChar(*c);

      if((nextChar == 'D') ||
         (nextChar == 'E')){
        ++c;

        if((*c == '+') || (*c == '-'))
          ++c;

        skipInt(c);
      }

      if(*c == '_')
        c += 2;
    }

    void skipString(const char *&c){
      if(!isAString(c))
        return;

      const char match = *(c++);

      while(*c != '\0'){
        if(*c == '\\')
          ++c;
        else if(*c == match){
          ++c;
          return;
        }

        ++c;
      }
    }

    char isAWordDelimeter(const char *c, const bool parsingC){
      if(!parsingC)
        return isAFortranWordDelimeter(c);

      if(charIsIn(c[0], cWordDelimeter)){
        if(charIsIn2(c, cWordDelimeter2)){
          if(charIsIn3(c, cWordDelimeter3))
            return 3;

          return 2;
        }

        return 1;
      }

      return 0;
    }

    char isAFortranWordDelimeter(const char *c){
      if(charIsIn(c[0], fortranWordDelimeter)){
        if(charIsIn2(c, fortranWordDelimeter2))
          return 2;

        if(c[0] == '.'){
          const char *c2 = (c + 1);

          while(*c2 != '.')
            ++c2;

          return (c2 - c + 1);
        }

        return 1;
      }

      return 0;
    }

    int skipWord(const char *&c, const bool parsingC){
      while(!charIsIn(*c, whitespace) && (*c != '\0')){
        const int delimeterChars = isAWordDelimeter(c, parsingC);

        if(delimeterChars == 0)
          ++c;
        else
          return delimeterChars;
      }

      return 0;
    }

    const char* readLine(const char *c, const bool parsingC){
      if(!parsingC)
        return readFortranLine(c);

      const char *c0 = c;
      bool breakNextLine = true;

      while(*c != '\0'){
        skipString(c);

        if(*c == '\0')
          break;

        if(*c == '\n'){
          if(breakNextLine)
            break;

          breakNextLine = false;
        }
        // Append next line
        else if((c[0] == '\\') && isWhitespace(c[1])){
          breakNextLine = true;
          ++c;
        }
        else if(c[0] == '/'){
          if(c[1] == '/'){
            while((*c != '\n') && (*c != '\0'))
              ++c;

            return c;
          }
          else if(c[1] == '*'){
            c += 2;

            while( !((c[0] == '*') && (c[1] == '/')) &&
                   (*c != '\0') )
              ++c;

            if(*c == '*')
              c += 2;

            return c;
          }
        }

        ++c;
      }

      return ((c[0] != '\0') ? (c + 1) : c);
    }

    const char* readFortranLine(const char *c){
      const char *c0 = c;
      bool breakNextLine = true;

      // Starting with [c] means line is a comment
      if(*c == 'c'){
        while((*c != '\n') &&
              (*c != '\0')){

          ++c;
        }

        return c;
      }

      while(*c != '\0'){
        skipString(c);

        if(*c == '\0')
          break;

        if(*c == '\n'){
          if(breakNextLine)
            break;

          breakNextLine = false;
        }
        // Append next line
        else if((c[0] == '&') && isWhitespace(c[1])){
          breakNextLine = true;
          ++c;
        }
        else if(c[0] == '!'){
          while((*c != '\n') && (*c != '\0'))
            ++c;

          return c;
        }

        ++c;
      }

      return ((c[0] != '\0') ? (c + 1) : c);
    }

    std::string compressWhitespace(const std::string &str){
      const size_t chars = str.size();
      std::string ret = str;

      const char *c = str.c_str();
      size_t pos = 0;

      while(*c != '\0'){
        if(isWhitespace(*c)){
          ret[pos++] = ' ';

          skipWhitespace(c);
        }
        else
          ret[pos++] = *(c++);
      }

      ret.resize(pos);

      return ret;
    }

    std::string strip(const char *c, const size_t chars){
      if(chars == 0)
        return "";

      const char *cLeft  = c;
      const char *cRight = c + (chars - 1);

      while(charIsIn(*cLeft , whitespace) && (cLeft <= cRight)) ++cLeft;
      while(charIsIn(*cRight, whitespace) && (cRight > cLeft)) --cRight;

      if(cLeft > cRight)
        return "";

      std::string ret = "";

      const char *cMid = cLeft;

      while(cMid < cRight){
        if((cMid[0] == '\\') && isWhitespace(cMid[1])){
          ret += strip(cLeft, cMid - cLeft);
          ret += ' ';

          ++cMid;

          cLeft = (cMid + 1);
        }

        ++cMid;

        if((cMid >= cRight) && ret.size())
          ret += strip(cLeft, (cMid - cLeft + 1));
      }

      if(ret.size() == 0)
        return compressWhitespace( std::string(cLeft, (cRight - cLeft + 1)) );

      return compressWhitespace(ret);
    }

    void strip(std::string &str){
      str = strip(str.c_str(), str.size());
    }

    char* cReadFile(const std::string &filename){
      // NBN: handle EOL chars on Windows
      FILE *fp = fopen(filename.c_str(), "r");

      if(!fp){
        printf("Failed to open: %s\n", filename.c_str());
        throw 1;
      }

      struct stat statbuf;
      stat(filename.c_str(), &statbuf);

      const size_t nchars = statbuf.st_size;

      char *buffer = (char*) calloc(nchars + 1, sizeof(char));
      size_t nread = fread(buffer, sizeof(char), nchars, fp);

      buffer[nread] = '\0';

      fclose(fp);

      return buffer;
    }

    int stripComments(std::string &line, const bool parsingC){
      if(!parsingC)
        return stripFortranComments(line);

      std::string line2  = line;
      line = "";

      const char *cLeft  = line2.c_str();
      const char *cRight = cLeft;

      int status = readingCode;

      while(*cRight != '\0'){
        skipString(cRight);

        if((*cRight == '\0') || (*cRight == '\n'))
          break;

        if((cRight[0] == '/') && (cRight[1] == '/')){
          line += std::string(cLeft, cRight - cLeft);
          return readingCode;
        }
        else if((cRight[0] == '/') && (cRight[1] == '*')){
          line += std::string(cLeft, cRight - cLeft);
          status = insideCommentBlock;
          cLeft = cRight + 2;
        }
        else if((cRight[0] == '*') && (cRight[1] == '/')){
          if(status == insideCommentBlock)
            status = readingCode;
          else
            status = finishedCommentBlock;
          cLeft = cRight + 2;
        }

        ++cRight;
      }

      if(cLeft != cRight)
        line += std::string(cLeft, cRight - cLeft);

      return status;
    }

    int stripFortranComments(std::string &line){
      std::string line2  = line;
      line = "";

      const char *cLeft  = line2.c_str();
      const char *cRight = cLeft;

      int status = readingCode;

      while(*cRight != '\0'){
        skipString(cRight);

        if((*cRight == '\0') || (*cRight == '\n'))
          break;

        if(*cRight == '!'){
          line += std::string(cLeft, cRight - cLeft);
          return readingCode;
        }

        ++cRight;
      }

      if(cLeft != cRight)
        line += std::string(cLeft, cRight - cLeft);

      return status;
    }

    char segmentPair(const char c){
      return ((')' * (c == '(')) +
              (']' * (c == '[')) +
              ('}' * (c == '{')) +
              ('(' * (c == ')')) +
              ('[' * (c == ']')) +
              ('{' * (c == '}')));
    }


    void skipPair(const char *&c){
      if(*c == '\0')
        return;

      const char pair = segmentPair(*c);

      if(pair == 0)
        return;

      ++c;

      while((*c != '\0') &&
            (*c != pair)){
        if(segmentPair(*c))
          skipPair(c);
        else
          ++c;
      }

      if(*c != '\0')
        ++c;
    }
    //==============================================

    std::string getBits(const int value){
      if(value == 0)
        return "0";

      std::stringstream ret;

      bool printedSomething = false;

      for(int i = 0; i < 32; ++i){
        if(value & (1 << i)){
          if(printedSomething)
            ret << ',';

          ret << i;

          printedSomething = true;
        }
      }

      return ret.str();
    }
  };
};
