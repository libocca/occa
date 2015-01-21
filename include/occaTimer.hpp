#ifndef OCCA_TIMER_HEADER
#define OCCA_TIMER_HEADER

#include "occaBase.hpp"

#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <stack>
#include <map>
#include <iomanip>
#include <utility>
#include <algorithm>

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

    void tic(std::string key);

    double toc(std::string key);

    double toc(std::string key, double flops);

    double toc(std::string key, occa::kernel &kernel);

    double toc(std::string key, occa::kernel &kernel, double flops);

    double toc(std::string key, double flops, double bw);

    double toc(std::string key, occa::kernel &kernel, double flops, double bw);

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

  double toc(std::string key);

  double toc(std::string key, occa::kernel &kernel);

  double toc(std::string key, double fp);

  double toc(std::string key, occa::kernel &kernel, double fp);

  double toc(std::string key, double fp, double bw);

  double toc(std::string key, occa::kernel &kernel, double fp, double bw);

  void printTimer();
}

#endif
