#ifndef OCCA_TIMER_HEADER
#define OCCA_TIMER_HEADER

#include "occaBase.hpp"

#include<iostream>
#include<fstream>
#include<assert.h>
#include<vector>
#include<stack>
#include<map>
#include<iomanip>
#include<utility>
#include<algorithm>
#ifndef WIN32
#  include <sys/time.h>
#else
#  undef UNICODE
#  include <windows.h>
#endif

namespace occa {

  class timerTraits{
  public:
    double timeTaken;
    double selfTime;
    int    numCalls;
    double flopCount;
    double bandWidthCount;
    int treeDepth;
    std::vector<std::string> childs;

    timerTraits();
  };

  class timer{

    bool profileKernels;
    bool profileApplication;
    bool deviceInitialized;

    occa::device occaHandle;

  public:

    timer();

    void initTimer(const occa::device &deviceHandle);

    std::stack<std::string> keyStack;
    std::stack<double> timeStack;

    std::map<std::stack<std::string>, timerTraits> times;

    double getTime();

    void tic(std::string key);

    void toc(std::string key);

    void toc(std::string key, double flops);

    void toc(std::string key, occa::kernel &kernel);

    void toc(std::string key, occa::kernel &kernel, double flops);

    void toc(std::string key, double flops, double bw);

    void toc(std::string key, occa::kernel &kernel, double flops, double bw);

    double print_recursively(std::vector<std::string> &childs,
                             double parentTime,
                             double overallTime);



    // struct myclass {
    //   bool operator() (std::pair<std::string, timerTraits> &a,
    // 		     std::pair<std::string, timerTraits> &b){
    //     return(a.second.selfTime > b.second.selfTime);
    //   }
    // } compareSelfTimes;


    void printTimer();
  };


  extern timer globalTimer;

  extern double dataTransferred;

  void initTimer(const occa::device &deviceHandle);

  void tic(std::string key);

  void toc(std::string key);

  void toc(std::string key, occa::kernel &kernel);

  void toc(std::string key, double fp);

  void toc(std::string key, occa::kernel &kernel, double fp);

  void toc(std::string key, double fp, double bw);

  void toc(std::string key, occa::kernel &kernel, double fp, double bw);

  void printTimer();

}

#endif
