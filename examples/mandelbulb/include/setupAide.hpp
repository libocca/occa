#ifndef OCCA_SETUPAIDE_HEADER
#define OCCA_SETUPAIDE_HEADER

#include <sys/stat.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <sstream>
#include <vector>
#include <map>

#include "occa.hpp"

using std::stringstream;
using std::string;
using std::vector;
using std::map;

class setupAide {
private:
  map<string,string> data;

public:
  setupAide();
  /// Setup from given file
  setupAide(string fileName);

  setupAide(const setupAide&);
  setupAide& operator=(const setupAide&);

  /// Parse through the read file
  void read(string setupFile);

  /// Get arguments for given string
  string getArgs(string key);

  template <class T>
  void getArgs(string key, T &arg);

  template <class T>
  void getArgs(string key, vector<T>& args);

  void getArgs(string key, vector<string>& args, string delimeter);
};

template <>
void setupAide::getArgs(string key, string &t);

static inline std::string strip(const std::string &str){
  const int chars = str.size();
  const char *c = str.c_str();
  int start, end;

  for(start = 0; start < chars; ++start)
    if((c[start] != ' ')  &&
       (c[start] != '\t') &&
       (c[start] != '\n') &&
       (c[start] != '\r'))
      break;

  for(end = (chars - 1); 0 <= end; --end)
    if((c[end] != ' ')  &&
       (c[end] != '\t') &&
       (c[end] != '\n') &&
       (c[end] != '\r'))
      break;

  return str.substr(start, end - start + 1);
}

#include "setupAide.tpp"

#endif
