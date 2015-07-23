namespace occa {
  template <class TM, const int idxType>
  array<TM,idxType>::array() :
    data_(NULL) {

    initSOrder();
  }


  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM,idxType>::array(const array<TM2,idxType2> &v){
    *this = v;
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM,idxType>& array<TM,idxType>::operator = (const array<TM2,idxType2> &v){
    device = v.device;
    memory = v.memory;

    data_ = v.data_;

    initSOrder(v.idxCount);

    for(int i = 0; i < idxCount; ++i){
      ks_[i]     = v.ks_[i];
      s_[i]      = v.s_[i];
      sOrder_[i] = v.sOrder_[i];
    }

    if(idxType == occa::useIdxOrder)
      updateFS(v.idxCount);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::initSOrder(int idxCount_){
    idxCount = idxCount_;

    if(idxType == occa::useIdxOrder){
      for(int i = 0; i < 6; ++i)
        sOrder_[i] = i;
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::free(){
    if(data_ == NULL)
      return;

    occa::free(data_);

    data_ = NULL;

    for(int i = 0; i < 6; ++i){
      ks_[i]     = 0;
      s_[i]      = 0;
      sOrder_[i] = i;
    }
  }

  //---[ Info ]-------------------------
  template <class TM, const int idxType>
  std::string array<TM,idxType>::idxOrderStr(){
    if(idxType == occa::useIdxOrder){
      std::string str(2*idxCount - 1, ',');

      for(int i = 0; i < idxCount; ++i)
        str[2*i] = ('0' + sOrder_[idxCount - i - 1]);

      return str;
    }
    else {
      OCCA_CHECK(false,
                 "Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()");
    }

    return "";
  }

  //---[ clone() ]----------------------
  template <class TM, const int idxType>
  array<TM,idxType> array<TM,idxType>::clone(const int copyOn){
    return cloneOn(device, copyOn);
  }

  template <class TM, const int idxType>
  array<TM,idxType> array<TM,idxType>::cloneOnCurrentDevice(const int copyOn){
    return cloneOn(occa::getCurrentDevice(), copyOn);
  }

  template <class TM, const int idxType>
  array<TM,idxType> array<TM,idxType>::cloneOn(occa::device device_, const int copyOn){
    return cloneOn<TM,idxType>(device_, copyOn);
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM2,idxType2> array<TM,idxType>::clone(const int copyOn){
    return cloneOn<TM2>(device, copyOn);
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM2,idxType2> array<TM,idxType>::cloneOnCurrentDevice(const int copyOn){
    return cloneOn<TM2>(occa::getCurrentDevice(), copyOn);
  }

  template <class TM, const int idxType>
  template <class TM2, const int idxType2>
  array<TM2,idxType2> array<TM,idxType>::cloneOn(occa::device device_, const int copyOn){
    array<TM2,idxType2> clone_ = *this;

    clone_.allocate(device_, idxCount, s_);

    occa::memcpy(clone_.data_,
                 data_,
                 bytes());

    return clone_;
  }

  //---[ array(...) ]------------------
  template <class TM, const int idxType>
  array<TM,idxType>::array(const int dim, const dim_t *d){
    initSOrder(dim);

    switch(dim){
    case 1: allocate(d[0]);                               break;
    case 2: allocate(d[1], d[0]);                         break;
    case 3: allocate(d[2], d[1], d[0]);                   break;
    case 4: allocate(d[3], d[2], d[1], d[0]);             break;
    case 5: allocate(d[4], d[3], d[2], d[1], d[0]);       break;
    case 6: allocate(d[5], d[4], d[3], d[2], d[1], d[0]); break;
    default:
      if(dim <= 0){
        OCCA_CHECK(false,
                   "Number of dimensions must be [1-6]");
      }
      else {
        OCCA_CHECK(false,
                   "occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const dim_t d0){
    initSOrder(1);

    allocate(d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const dim_t d1, const dim_t d0){
    initSOrder(2);

    allocate(d1, d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const dim_t d2, const dim_t d1, const dim_t d0){
    initSOrder(3);

    allocate(d2, d1, d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const dim_t d3,
                           const dim_t d2, const dim_t d1, const dim_t d0){
    initSOrder(4);

    allocate(d3,
             d2, d1, d0);

  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const dim_t d4, const dim_t d3,
                           const dim_t d2, const dim_t d1, const dim_t d0){
    initSOrder(5);

    allocate(d4, d3,
             d2, d1, d0);

  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(const dim_t d5, const dim_t d4, const dim_t d3,
                           const dim_t d2, const dim_t d1, const dim_t d0){
    initSOrder(6);

    allocate(d5, d4, d3,
             d2, d1, d0);
  }

  //---[ array(device, ...) ]----------
  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_, const int dim, const dim_t *d){
    switch(dim){
    case 1: allocate(device_, d[0]);                               break;
    case 2: allocate(device_, d[1], d[0]);                         break;
    case 3: allocate(device_, d[2], d[1], d[0]);                   break;
    case 4: allocate(device_, d[3], d[2], d[1], d[0]);             break;
    case 5: allocate(device_, d[4], d[3], d[2], d[1], d[0]);       break;
    case 6: allocate(device_, d[5], d[4], d[3], d[2], d[1], d[0]); break;
    default:
      if(dim <= 0){
        OCCA_CHECK(false,
                   "Number of dimensions must be [1-6]");
      }
      else {
        OCCA_CHECK(false,
                   "occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const dim_t d0){

    allocate(device_,
             d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const dim_t d1, const dim_t d0){

    allocate(device_,
             d1, d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const dim_t d2, const dim_t d1, const dim_t d0){

    allocate(device_,
             d2, d1, d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const dim_t d3,
                           const dim_t d2, const dim_t d1, const dim_t d0){

    allocate(device_,
             d3,
             d2, d1, d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const dim_t d4, const dim_t d3,
                           const dim_t d2, const dim_t d1, const dim_t d0){

    allocate(device_,
             d4, d3,
             d2, d1, d0);
  }

  template <class TM, const int idxType>
  array<TM,idxType>::array(occa::device device_,
                           const dim_t d5, const dim_t d4, const dim_t d3,
                           const dim_t d2, const dim_t d1, const dim_t d0){

    allocate(device_,
             d5, d4, d3,
             d2, d1, d0);
  }

  //---[ allocate(...) ]----------------
  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(){
    const dim_t entries = (s_[0] * s_[1] * s_[2] *
                           s_[3] * s_[4] * s_[5]);

    data_ = (TM*) device.managedAlloc(entries * sizeof(TM));

    memory = occa::memory(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const int dim, const dim_t *d){
    switch(dim){
    case 1: allocate(d[0]);                               break;
    case 2: allocate(d[1], d[0]);                         break;
    case 3: allocate(d[2], d[1], d[0]);                   break;
    case 4: allocate(d[3], d[2], d[1], d[0]);             break;
    case 5: allocate(d[4], d[3], d[2], d[1], d[0]);       break;
    case 6: allocate(d[5], d[4], d[3], d[2], d[1], d[0]); break;
    default:
      if(dim <= 0){
        OCCA_CHECK(false,
                   "Number of dimensions must be [1-6]");
      }
      else {
        OCCA_CHECK(false,
                   "occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const dim_t d0){
    allocate(occa::getCurrentDevice(),
             d0);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const dim_t d1, const dim_t d0){
    allocate(occa::getCurrentDevice(),
             d1, d0);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const dim_t d2, const dim_t d1, const dim_t d0){
    allocate(occa::getCurrentDevice(),
             d2, d1, d0);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const dim_t d3,
                                   const dim_t d2, const dim_t d1, const dim_t d0){

    allocate(occa::getCurrentDevice(),
             d3,
             d2, d1, d0);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const dim_t d4, const dim_t d3,
                                   const dim_t d2, const dim_t d1, const dim_t d0){

    allocate(occa::getCurrentDevice(),
             d4, d3,
             d2, d1, d0);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(const dim_t d5, const dim_t d4, const dim_t d3,
                                   const dim_t d2, const dim_t d1, const dim_t d0){

    allocate(occa::getCurrentDevice(),
             d5, d4, d3,
             d2, d1, d0);
  }

  //---[ allocate(device, ...) ]--------
  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_, const int dim, const dim_t *d){
    switch(dim){
    case 1: allocate(device_, d[0]);                               break;
    case 2: allocate(device_, d[1], d[0]);                         break;
    case 3: allocate(device_, d[2], d[1], d[0]);                   break;
    case 4: allocate(device_, d[3], d[2], d[1], d[0]);             break;
    case 5: allocate(device_, d[4], d[3], d[2], d[1], d[0]);       break;
    case 6: allocate(device_, d[5], d[4], d[3], d[2], d[1], d[0]); break;
    default:
      if(dim <= 0){
        OCCA_CHECK(false,
                   "Number of dimensions must be [1-6]");
      }
      else {
        OCCA_CHECK(false,
                   "occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const dim_t d0){

    device = device_;

    reshape(d0);

    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const dim_t d1, const dim_t d0){

    device = device_;

    reshape(d1, d0);

    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const dim_t d2, const dim_t d1, const dim_t d0){

    device = device_;

    reshape(d2, d1, d0);

    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const dim_t d3,
                                   const dim_t d2, const dim_t d1, const dim_t d0){

    device = device_;

    reshape(d3,
            d2, d1, d0);

    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const dim_t d4, const dim_t d3,
                                   const dim_t d2, const dim_t d1, const dim_t d0){

    device = device_;

    reshape(d4, d3,
            d2, d1, d0);

    allocate();
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::allocate(occa::device device_,
                                   const dim_t d5, const dim_t d4, const dim_t d3,
                                   const dim_t d2, const dim_t d1, const dim_t d0){

    device = device_;

    reshape(d5, d4, d3,
            d2, d1, d0);

    allocate();
  }

  //---[ reshape(...) ]-----------------
  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const int dim, const dim_t *d){
    switch(dim){
    case 1: reshape(d[0]);                               break;
    case 2: reshape(d[1], d[0]);                         break;
    case 3: reshape(d[2], d[1], d[0]);                   break;
    case 4: reshape(d[3], d[2], d[1], d[0]);             break;
    case 5: reshape(d[4], d[3], d[2], d[1], d[0]);       break;
    case 6: reshape(d[5], d[4], d[3], d[2], d[1], d[0]); break;
    default:
      if(dim <= 0){
        OCCA_CHECK(false,
                   "Number of dimensions must be [1-6]");
      }
      else {
        OCCA_CHECK(false,
                   "occa::array can only take up to 6 dimensions");
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const dim_t d0){

    s_[0] = d0; s_[1] =  1; s_[2] =  1;
    s_[3] =  1; s_[4] =  1; s_[5] =  1;

    updateFS(1);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const dim_t d1, const dim_t d0){

    s_[0] = d0; s_[1] = d1; s_[2] =  1;
    s_[3] =  1; s_[4] =  1; s_[5] =  1;

    updateFS(2);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const dim_t d2, const dim_t d1, const dim_t d0){

    s_[0] = d0; s_[1] = d1; s_[2] = d2;
    s_[3] =  1; s_[4] =  1; s_[5] =  1;

    updateFS(3);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const dim_t d3,
                                  const dim_t d2, const dim_t d1, const dim_t d0){

    s_[0] = d0; s_[1] = d1; s_[2] = d2;
    s_[3] = d3; s_[4] =  1; s_[5] =  1;

    updateFS(4);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const dim_t d4, const dim_t d3,
                                  const dim_t d2, const dim_t d1, const dim_t d0){

    s_[0] = d0; s_[1] = d1; s_[2] = d2;
    s_[3] = d3; s_[4] = d4; s_[5] =  1;

    updateFS(5);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::reshape(const dim_t d5, const dim_t d4, const dim_t d3,
                                  const dim_t d2, const dim_t d1, const dim_t d0){

    s_[0] = d0; s_[1] = d1; s_[2] = d2;
    s_[3] = d3; s_[4] = d4; s_[5] = d5;

    updateFS(6);
  }

  //---[ setIdxOrder(...) ]-------------
  template <class TM, const int idxType>
  void array<TM,idxType>::updateFS(const int idxCount_){
    idxCount = idxCount_;

    for(int i = 0; i < idxCount; ++i)
      ks_[i] = s_[i];

    if(idxType == occa::useIdxOrder){
      dim_t fs2[7];

      fs2[0] = 1;

      for(int i = 0; i < 6; ++i){
        const int i2 = (sOrder_[i] + 1);

        fs2[i2] = s_[i];
      }

      for(int i = 1; i < 7; ++i)
        fs2[i] *= fs2[i - 1];

      for(int i = 0; i < 6; ++i)
        fs_[i] = fs2[sOrder_[i]];
    }
  }

  //  |---[ useIdxOrder ]---------------
  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int dim, const int *o){
    if(idxType == occa::useIdxOrder){
      switch(dim){
      case 1:                                                  break;
      case 2: setIdxOrder(o[1], o[0]);                         break;
      case 3: setIdxOrder(o[2], o[1], o[0]);                   break;
      case 4: setIdxOrder(o[3], o[2], o[1], o[0]);             break;
      case 5: setIdxOrder(o[4], o[3], o[2], o[1], o[0]);       break;
      case 6: setIdxOrder(o[5], o[4], o[3], o[2], o[1], o[0]); break;
      default:
        if(dim <= 0){
          OCCA_CHECK(false,
                     "Number of dimensions must be [1-6]");
        }
        else {
          OCCA_CHECK(false,
                     "occa::array can only take up to 6 dimensions");
        }
      }
    }
    else {
      OCCA_CHECK(false,
                 "Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const std::string &default_,
                                      const std::string &given){

    const int dim = (int) default_.size();
    int o[6];

    OCCA_CHECK((dim == ((int) given.size())) &&
               (1 <= dim) && (dim <= 6),
               "occa::array::setIdxOrder(default, given) must have matching sized strings of size [1-6]");

    for(int j = 0; j < dim; ++j)
      o[j] = -1;

    for(int j_ = (dim - 1); 0 <= j_; --j_){
      const int j = (dim - j_ - 1);

      const char C = default_[j_];

      for(int i_ = (dim - 1); 0 <= i_; --i_){
        const int i = (dim - i_ - 1);

        if(C == given[i_]){
          OCCA_CHECK(o[j] == -1,
                     "occa::array::setIdxOrder(default, given) must have strings with unique characters");

          o[j] = i;
          break;
        }
      }
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o1, const int o0){
    if(idxType == occa::useIdxOrder){
      OCCA_CHECK((0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1),
                 "occa::array::setIdxOrder("
                 << o1 << ','
                 << o0 << ") has index out of bounds");

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] =  2;
      sOrder_[3] =  3; sOrder_[4] =  4; sOrder_[5] =  5;

      updateFS(2);
    }
    else {
      OCCA_CHECK(false,
                 "Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o2, const int o1, const int o0){
    if(idxType == occa::useIdxOrder){
      OCCA_CHECK((0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1) &&
                 (0 <= o2) && (o2 <= 1),
                 "occa::array::setIdxOrder("
                 << o2 << ','
                 << o1 << ','
                 << o0 << ") has index out of bounds");

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] = o2;
      sOrder_[3] =  3; sOrder_[4] =  4; sOrder_[5] =  5;

      updateFS(3);
    }
    else {
      OCCA_CHECK(false,
                 "Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o3,
                                      const int o2, const int o1, const int o0){
    if(idxType == occa::useIdxOrder){
      OCCA_CHECK((0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1) &&
                 (0 <= o2) && (o2 <= 1) &&
                 (0 <= o3) && (o3 <= 1),
                 "occa::array::setIdxOrder("
                 << o3 << ','
                 << o2 << ','
                 << o1 << ','
                 << o0 << ") has index out of bounds");

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] = o2;
      sOrder_[3] = o3; sOrder_[4] =  4; sOrder_[5] =  5;

      updateFS(4);
    }
    else {
      OCCA_CHECK(false,
                 "Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o4, const int o3,
                                      const int o2, const int o1, const int o0){
    if(idxType == occa::useIdxOrder){
      OCCA_CHECK((0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1) &&
                 (0 <= o2) && (o2 <= 1) &&
                 (0 <= o3) && (o3 <= 1) &&
                 (0 <= o4) && (o4 <= 1),
                 "occa::array::setIdxOrder("
                 << o4 << ','
                 << o3 << ','
                 << o2 << ','
                 << o1 << ','
                 << o0 << ") has index out of bounds");

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] = o2;
      sOrder_[3] = o3; sOrder_[4] = o4; sOrder_[5] =  5;

      updateFS(5);
    }
    else {
      OCCA_CHECK(false,
                 "Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()");
    }
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::setIdxOrder(const int o5, const int o4, const int o3,
                                      const int o2, const int o1, const int o0){
    if(idxType == occa::useIdxOrder){
      OCCA_CHECK((0 <= o0) && (o0 <= 1) &&
                 (0 <= o1) && (o1 <= 1) &&
                 (0 <= o2) && (o2 <= 1) &&
                 (0 <= o3) && (o3 <= 1) &&
                 (0 <= o4) && (o4 <= 1) &&
                 (0 <= o5) && (o5 <= 1),
                 "occa::array::setIdxOrder("
                 << o5 << ','
                 << o4 << ','
                 << o3 << ','
                 << o2 << ','
                 << o1 << ','
                 << o0 << ") has index out of bounds");

      sOrder_[0] = o0; sOrder_[1] = o1; sOrder_[2] = o2;
      sOrder_[3] = o3; sOrder_[4] = o4; sOrder_[5] = o5;

      updateFS(6);
    }
    else {
      OCCA_CHECK(false,
                 "Only occa::array<TM, occa::useIdxOrder> can use setIdxOrder()");
    }
  }

  //---[ Operators ]--------------------
  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator [] (const dim_t i0){

    return data_[i0];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const dim_t i0){

    return data_[i0];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const dim_t i1, const dim_t i0){
    if(idxType == occa::dontUseIdxOrder)
      return data_[i0 + s_[0]*i1];
    else
      return data_[fs_[0]*i0 + fs_[1]*i1];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const dim_t i2, const dim_t i1, const dim_t i0){
    if(idxType == occa::dontUseIdxOrder)
      return data_[i0 + s_[0]*(i1 + s_[1]*i2)];
    else
      return data_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const dim_t i3,
                                             const dim_t i2, const dim_t i1, const dim_t i0){
    if(idxType == occa::dontUseIdxOrder)
      return data_[i0 + s_[0]*(i1 + s_[1]*(i2 + s_[2]*i3))];
    else
      return data_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2 + fs_[3]*i3];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const dim_t i4, const dim_t i3,
                                             const dim_t i2, const dim_t i1, const dim_t i0){
    if(idxType == occa::dontUseIdxOrder)
      return data_[i0 + s_[0]*(i1 + s_[1]*(i2 + s_[2]*(i3 + s_[3]*i4)))];
    else
      return data_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2 + fs_[3]*i3 + fs_[4]*i4];
  }

  template <class TM, const int idxType>
  inline TM& array<TM,idxType>::operator () (const dim_t i5, const dim_t i4, const dim_t i3,
                                             const dim_t i2, const dim_t i1, const dim_t i0){
    if(idxType == occa::dontUseIdxOrder)
      return data_[i0 + s_[0]*(i1 + s_[1]*(i2 + s_[2]*(i3 + s_[3]*(i4 + s_[4]*i5))))];
    else
      return data_[fs_[0]*i0 + fs_[1]*i1 + fs_[2]*i2 + fs_[3]*i3 + fs_[4]*i4 + fs_[5]*i5];
  }

  //---[ Syncs ]------------------------
  template <class TM, const int idxType>
  void array<TM,idxType>::startManaging(){
    occa::startManaging(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::stopManaging(){
    occa::stopManaging(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::syncToDevice(){
    occa::syncToDevice(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::syncFromDevice(){
    occa::syncFromDevice(data_);
  }

  template <class TM, const int idxType>
  bool array<TM,idxType>::needsSync(){
    return occa::needsSync(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::sync(){
    occa::sync(data_);
  }

  template <class TM, const int idxType>
  void array<TM,idxType>::dontSync(){
    occa::dontSync(data_);
  }
}
