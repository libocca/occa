#include "occa/timer.hpp"
#include "occa/tools.hpp"

namespace occa {
  timerTraits::timerTraits(){
    timeTaken      = 0.0;
    selfTime       = 0.0;
    numCalls       = 0;
    flopCount      = 0.0;
    bandWidthCount = 0.0;
    treeDepth      = 0;
  }

  timer::timer(){
    profileKernels     = false;
    deviceInitialized  = false;
    profileApplication = false;

    std::string profilerOn       = occa::env::var("OCCA_PROFILE");
    std::string kernelProfilerOn = occa::env::var("OCCA_KERNEL_PROFILE");

    if(profilerOn == "1")
      profileApplication = true;

    if(kernelProfilerOn == "1"){
      profileKernels     = true;
      profileApplication = true;
    }
  }

  void timer::initTimer(const occa::device &deviceHandle){
    deviceInitialized = true;

    occaHandle = deviceHandle;
  }

  void timer::tic(std::string key){

    if(profileApplication){
      keyStack.push(key);

      int treeDepth = keyStack.size() - 1;
      times[keyStack].treeDepth = treeDepth;

      double currentTime = occa::currentTime();

      timeStack.push(currentTime);


      if(treeDepth){
        keyStack.pop();

        // see if it was already in the child list
        std::vector<std::string>::iterator iter;
        std::vector<std::string> *childs = &(times[keyStack].childs);
        iter = std::find(childs->begin(), childs->end(), key);

        if(iter == childs->end())
          childs->push_back(key);

        keyStack.push(key);
      }
    }
  }


  double timer::toc(std::string key){

    double elapsedTime = 0.;

    if(profileApplication){
      assert(key == keyStack.top());

      OCCA_CHECK(key == keyStack.top(),
                 "Error in timer " << key << '\n');

      double currentTime = occa::currentTime();
      elapsedTime = (currentTime - timeStack.top());

      times[keyStack].timeTaken += elapsedTime;
      times[keyStack].numCalls++;

      keyStack.pop();
      timeStack.pop();
    }

    return elapsedTime;
  }

  double timer::toc(std::string key, occa::kernel &kernel){

    double elapsedTime = 0.;

    if(profileApplication){

      assert(key == keyStack.top());

      OCCA_CHECK(key == keyStack.top(),
                 "Error in timer " << key << '\n');

      if(profileKernels){
        if(deviceInitialized)
          occaHandle.finish();

        double currentTime = occa::currentTime();
        elapsedTime = (currentTime - timeStack.top());
        // times[keyStack].timeTaken += kernel.timeTaken();
        times[keyStack].timeTaken += elapsedTime;
        times[keyStack].numCalls++;
      }

      keyStack.pop();
      timeStack.pop();
    }

    return elapsedTime;
  }


  double timer::toc(std::string key, double flops){

    double elapsedTime = 0.;

    if(profileApplication){

      assert(key == keyStack.top());

      double currentTime = occa::currentTime();
      elapsedTime = (currentTime - timeStack.top());

      times[keyStack].timeTaken += elapsedTime;
      times[keyStack].numCalls++;
      times[keyStack].flopCount += flops;

      keyStack.pop();
      timeStack.pop();
    }

    return elapsedTime;
  }


  double timer::toc(std::string key, occa::kernel &kernel, double flops){

    double elapsedTime = 0.;

    if(profileApplication){

      assert(key == keyStack.top());

      if(profileKernels){
        occaHandle.finish();
        double currentTime = occa::currentTime();
        elapsedTime = (currentTime - timeStack.top());
        // times[keyStack].timeTaken += kernel.timeTaken();
        times[keyStack].timeTaken += elapsedTime;
        times[keyStack].numCalls++;
        times[keyStack].flopCount += flops;
      }

      keyStack.pop();
      timeStack.pop();
    }

    return elapsedTime;
  }

  double timer::toc(std::string key, double flops, double bw){

    double elapsedTime = 0.;

    if(profileApplication){

      assert(key == keyStack.top());

      double currentTime = occa::currentTime();
      elapsedTime = (currentTime - timeStack.top());

      times[keyStack].timeTaken += elapsedTime;
      times[keyStack].numCalls++;
      times[keyStack].flopCount += flops;
      times[keyStack].bandWidthCount += bw;

      dataTransferred += bw;

      keyStack.pop();
      timeStack.pop();
    }

    return elapsedTime;
  }


  double timer::toc(std::string key, occa::kernel &kernel,
                    double flops, double bw){

    double elapsedTime = 0.;

    if(profileApplication){

      assert(key == keyStack.top());

      if(profileKernels){
        occaHandle.finish();
        double currentTime = occa::currentTime();
        elapsedTime = (currentTime - timeStack.top());
        // times[keyStack].timeTaken += kernel.timeTaken();
        times[keyStack].timeTaken += elapsedTime;
        times[keyStack].numCalls++;
        times[keyStack].flopCount += flops;
        times[keyStack].bandWidthCount += bw;
      }

      dataTransferred += bw;

      keyStack.pop();
      timeStack.pop();
    }

    return elapsedTime;
  }

  double timer::print_recursively(std::vector<std::string> &childs,
                                  double parentTime,
                                  double overallTime){

    double sumChildrenTime = 0.0;

    for(size_t i = 0; i < childs.size(); ++i){

      keyStack.push(childs[i]);

      timerTraits *traits = &(times[keyStack]);

      std::string stringName = "  ";
      for(int j=0; j<traits->treeDepth; j++)	stringName.append(" ");

      stringName.append("*"); stringName.append(keyStack.top());

      double timeTaken = traits->timeTaken;

      sumChildrenTime += timeTaken;

      double invTimeTaken = (timeTaken > 1e-10) ? 1.0/timeTaken : 0.;

      std::cout << std::left << std::setw(30)<< stringName
                << std::right << std::setw(10)<< std::setprecision(3)<<timeTaken
                << std::right<<std::setw(10)<<traits->numCalls
                << std::right<<std::setw(10)<<std::setprecision(3)<<100.*timeTaken/parentTime
                << std::right<<std::setw(10)<<std::setprecision(3)<<100.*timeTaken/overallTime
                << std::right<<std::setw(10)<<std::setprecision(3)<<traits->flopCount*invTimeTaken/1e9
                << std::right<<std::setw(10)<<std::setprecision(3)<<traits->bandWidthCount*invTimeTaken/1e9
                << std::endl;

      traits->selfTime -= print_recursively(traits->childs, timeTaken, overallTime);

      keyStack.pop();
    }

    return sumChildrenTime;
  }


  static bool compareSelfTimes(std::pair<std::string, timerTraits> a,
                               std::pair<std::string, timerTraits> b){

    return (a.second.selfTime > b.second.selfTime);
  }

  void timer::printTimer(){

    if(profileApplication){
      std::map<std::stack<std::string>, timerTraits>::iterator iter;

      // compute overall time
      double overallTime = 0.;
      for(iter = times.begin(); iter != times.end(); iter++){
        iter->second.selfTime = iter->second.timeTaken;
        if(iter->second.treeDepth == 0){
          overallTime += iter->second.timeTaken;
        }
      }

      std::cout<<"********************************************************"
               <<"**********************************"<<std::endl;
      std::cout << "Profiling info: " << std::endl;
      std::cout << std::left<<std::setw(30)<<"Name"
                << std::right<<std::setw(10)<<"time spent"
                << std::right<<std::setw(10)<<"# calls"
                << std::right<<std::setw(10)<<"% time"
                << std::right<<std::setw(10)<<"% total"
                << std::right<<std::setw(10)<<"gflops "
                << std::right<<std::setw(10)<<"bwidth"
                << std::endl;

      std::cout<<"--------------------------------------------------------"
               <<"----------------------------------"<<std::endl;

      for(iter = times.begin(); iter != times.end(); iter++){
        if(iter->second.treeDepth == 0){

          keyStack = iter->first;

          timerTraits *traits = &(iter->second);

          std::string stringName = " *";
          stringName.append(keyStack.top());

          double timeTaken = traits->timeTaken;

          double invTimeTaken = (timeTaken > 1e-10) ? 1.0/timeTaken : 0.;

          std::cout << std::left << std::setw(30) << stringName
                    << std::right << std::setw(10) << std::setprecision(3)<<timeTaken
                    << std::right<<std::setw(10)<<traits->numCalls
                    << std::right<<std::setw(10)<<std::setprecision(3)<<100.0
                    << std::right<<std::setw(10)<<std::setprecision(3)<<100*timeTaken/overallTime
                    << std::right<<std::setw(10)<<std::setprecision(3)<<traits->flopCount*invTimeTaken/1e9
                    << std::right<<std::setw(10)<<std::setprecision(3)<<traits->bandWidthCount*invTimeTaken/1e9
                    << std::endl;

          traits->selfTime -= print_recursively(iter->second.childs, timeTaken, overallTime);

        }
      }


      std::map<std::string, timerTraits> flat;

      // flat profile
      for(iter=times.begin(); iter!=times.end(); iter++){

        std::string key = iter->first.top();

        timerTraits *traits = &(iter->second);

        timerTraits *targetTraits = &(flat[key]);

        targetTraits->timeTaken += traits->timeTaken;
        targetTraits->selfTime += traits->selfTime;
        targetTraits->numCalls += traits->numCalls;
        targetTraits->flopCount += traits->flopCount;
        targetTraits->bandWidthCount += traits->bandWidthCount;
      }


      std::vector<std::pair<std::string, timerTraits> > flatVec(flat.size());

      std::map<std::string, timerTraits>::iterator iter2 = flat.begin();

      for(size_t i = 0; i < flat.size(); ++i){
        flatVec[i].first = iter2->first;

        flatVec[i].second.timeTaken = iter2->second.timeTaken;
        flatVec[i].second.numCalls = iter2->second.numCalls;
        flatVec[i].second.selfTime = iter2->second.selfTime;
        flatVec[i].second.flopCount = iter2->second.flopCount;
        flatVec[i].second.bandWidthCount = iter2->second.bandWidthCount;

        iter2++;
      }

      // sort
      std::sort(flatVec.begin(), flatVec.end(), compareSelfTimes);


      // write the flat profiling info
      std::cout<<"********************************************************"
               <<"**********************************"<<std::endl;

      std::cout<<"Profiling summary: " << std::endl;


      std::cout << std::left<<std::setw(30)<<"Name"
                << std::right<<std::setw(10)<<"time"
                << std::right<<std::setw(10)<<"self time"
                << std::right<<std::setw(10)<<"# calls"
                << std::right<<std::setw(10)<<"% time"
                << std::right<<std::setw(10)<<"gflops "
                << std::right<<std::setw(10)<<"bwidth"
                << std::endl;

      std::cout<<"--------------------------------------------------------"
               <<"----------------------------------"<<std::endl;

      std::vector<std::pair<std::string, timerTraits> > ::iterator iter1;
      for(iter1=flatVec.begin(); iter1!=flatVec.end(); iter1++){

        timerTraits *traits = &(iter1->second);
        double timeTaken = traits->timeTaken;
        double invTimeTaken = (timeTaken > 1e-10) ? 1.0/timeTaken : 0.;
        std::cout << std::left<<std::setw(30) << iter1->first
                  << std::right<<std::setw(10) << std::setprecision(3)<<traits->timeTaken
                  << std::right<<std::setw(10) << std::setprecision(3)<<traits->selfTime
                  << std::right<<std::setw(10)<<traits->numCalls
                  << std::right<<std::setw(10)<<std::setprecision(3)<<100*traits->selfTime/overallTime
                  << std::right<<std::setw(10)<<std::setprecision(3)<<traits->flopCount*invTimeTaken/1e9
                  << std::right<<std::setw(10)<<std::setprecision(3)<<traits->bandWidthCount*invTimeTaken/1e9
                  << std::endl;

      }

      std::cout<<"********************************************************"
               <<"**********************************"<<std::endl;

    }
  }

  timer globalTimer;

  double dataTransferred = 0.;

  void initTimer(const occa::device &deviceHandle){
    globalTimer.initTimer(deviceHandle);
  }

  void tic(std::string key){
    globalTimer.tic(key);
  }

  double toc(std::string key){
    return globalTimer.toc(key);
  }

  double toc(std::string key, occa::kernel &kernel){
    return globalTimer.toc(key, kernel);
  }

  double toc(std::string key, double fp){
    return globalTimer.toc(key, fp);
  }

  double toc(std::string key, occa::kernel &kernel, double fp){
    return globalTimer.toc(key, kernel, fp);
  }

  double toc(std::string key, double fp, double bw){
    return globalTimer.toc(key, fp, bw);
  }

  double toc(std::string key, occa::kernel &kernel, double fp, double bw){
    return globalTimer.toc(key, kernel, fp, bw);
  }

  void printTimer(){
    globalTimer.printTimer();
  }
}
