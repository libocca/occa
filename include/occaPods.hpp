#ifndef OCCA_PODS_HEADER
#define OCCA_PODS_HEADER

#define OCCA_KERNEL_ARG_CONSTRUCTOR(TYPE)       \
  inline kernelArg(const TYPE &arg_){           \
    arg.TYPE##_ = arg_;                         \
    size = sizeof(TYPE);                        \
                                                \
    pointer = false;                            \
  }

namespace occa {
  class kernelArg {
  public:
    union arg_t {
      int int_;
      char char_;
      float float_;
      short short_;
      double double_;
      size_t size_t_;
      void* void_;
    } arg;

    size_t size;
    bool pointer;

    inline kernelArg(){
      arg.void_ = NULL;
    }

    inline kernelArg(const kernelArg &k) :
      arg(k.arg),
      size(k.size),
      pointer(k.pointer) {}

    inline kernelArg& operator = (const kernelArg &k){
      arg.void_ = k.arg.void_;
      size      = k.size;
      pointer   = k.pointer;

      return *this;
    }

    OCCA_KERNEL_ARG_CONSTRUCTOR(int);
    OCCA_KERNEL_ARG_CONSTRUCTOR(char);
    OCCA_KERNEL_ARG_CONSTRUCTOR(float);
    OCCA_KERNEL_ARG_CONSTRUCTOR(short);
    OCCA_KERNEL_ARG_CONSTRUCTOR(double);
    OCCA_KERNEL_ARG_CONSTRUCTOR(size_t);

    inline kernelArg(occa::memory &m){
      arg.void_ = m.mHandle->handle;
      size = sizeof(void*);

      pointer = true;
    }

    inline kernelArg(void *arg_){
      arg.void_ = arg_;
      size = sizeof(void*);

      pointer = true;
    }

    inline void* data() const {
      return pointer ? arg.void_ : (void*) &arg;
    }
  };
};

#endif
