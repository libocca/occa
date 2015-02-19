#include "setupAide.hpp"

setupAide::setupAide(){}

setupAide::setupAide(string setupFile){
  read(setupFile);
}

setupAide::setupAide(const setupAide& sa){
  *this = sa;
}

setupAide& setupAide::operator = (const setupAide& sa){
  data = sa.data;
  return *this;
}

void setupAide::read(string setupFile){
  vector<string> data2;
  vector<string> keyword2;

  string args = occa::readFile(setupFile);

  int size = args.length();
  string current = "";
  stringstream ss;
  char c;

  for(int i=0; i<size; i++){
    c = args[i];

    // Batch strings together
    if(c == '\'' || c == '"'){
      i++;

      while(i < size && args[i] != c)
        current += args[i++];

      if(i >= size)
        break;

      ++i;
    }

    // Batch comments
    else if(c == '/' && i < size && args[i+1] == '*'){
      i += 2;

      while( args[i] != '*' || (i < size && args[i+1] != '/') )
        i++;

      if(i >= size)
        break;

      i++;
    }

    // Removing # comments
    else if(c == '#'){
      i++;

      while(i < size && args[i] != '\n')
        i++;
    }

    // Change \[\] to []
    else if(c == '\\' && i < size && (args[i+1] == '[' || args[i+1] == ']')){
      current += args[i+1];
      i += 2;
    }

    // Split keywords []
    else if(c == '['){
      if(current != ""){
        data2.push_back(current);
        current = "";
      }

      i++;

      while(i < size && args[i] != ']')
        current += args[i++];

      keyword2.push_back(current);
      current = "";
    }

    // Else add the character
    else
      current += c;

    if(i >= (size-1) && current.length()){
      data2.push_back(current);
      current = "";
    }
  }

  if(current.length())
    data2.push_back(current);

  int argc = keyword2.size();

  for(int i=0; i<argc; i++)
    data[ keyword2[i] ] = data2[i];
}

string setupAide::getArgs(string key){
  return data[key];
}

template <>
void setupAide::getArgs(string key, string &t){
  t = data[key];
}

void setupAide::getArgs(string key, vector<string> &argv, string delimeter){
  string args, current;
  int size;

  args = getArgs(key);

  size = args.length();

  current = "";

  for(int i=0; i<size; i++){
    while( i < size && delimeter.find(args[i]) == string::npos )
      current += args[i++];

    if(current.length())
      argv.push_back(current);

    current = "";
  }
}
