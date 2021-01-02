namespace occa {
  namespace lang {
    //---[ baseStream ]-------------------
    template <class output_t>
    baseStream<output_t>::~baseStream() {}

    template <class output_t>
    void baseStream<output_t>::clearCache() {}

    template <class output_t>
    void* baseStream<output_t>::passMessageToInput(const occa::json &props) {
      return NULL;
    }

    template <class output_t>
    void* baseStream<output_t>::getInput(const std::string &name) {
      occa::json props;
      props["input_name"] = name;
      return passMessageToInput(props);
    }

    template <class output_t>
    template <class newOutput_t>
    stream<newOutput_t> baseStream<output_t>::map(streamMap<output_t, newOutput_t> &smap) {
      return stream<output_t>(*this).map(smap);
    }

    template <class output_t>
    stream<output_t> baseStream<output_t>::filter(streamFilter<output_t> &sfilter) {
      return stream<output_t>(*this).filter(sfilter);
    }

    template <class output_t>
    baseStream<output_t>& baseStream<output_t>::operator >> (output_t &out) {
      setNext(out);
      return *this;
    }
    //====================================

    //---[ stream ]-----------------------
    template <class output_t>
    stream<output_t>::stream() :
      head(NULL) {}

    template <class output_t>
    stream<output_t>::stream(baseStream<output_t> &head_) :
      head(&head_) {}

    template <class output_t>
    stream<output_t>::stream(const stream &other) :
      head(other.head) {}

    template <class output_t>
    stream<output_t>::~stream() {}

    template <class output_t>
    stream<output_t>& stream<output_t>::operator = (const stream &other) {
      head = other.head;
      return *this;
    }

    template <class output_t>
    stream<output_t>& stream<output_t>::clone() const {
      if (!head) {
        return *(new stream());
      }
      return *(new stream(&(head->clone())));
    }

    template <class output_t>
    void stream<output_t>::clearCache() {
      head->clearCache();
    }

    template <class output_t>
    void* stream<output_t>::passMessageToInput(const occa::json &props) {
      return head->passMessageToInput(props);
    }

    template <class output_t>
    void* stream<output_t>::getInput(const std::string &name) {
      return head->getInput(name);
    }

    template <class output_t>
    bool stream<output_t>::isEmpty() {
      return (!head || head->isEmpty());
    }

    template <class output_t>
    template <class newOutput_t>
    stream<newOutput_t> stream<output_t>::map(streamMap<output_t, newOutput_t> &map_) {
      if (!head) {
        return stream<newOutput_t>();
      }

      typedef streamMap<output_t, newOutput_t> mapType;

      stream<newOutput_t> s(map_);
      mapType &sHead = *(static_cast<mapType*>(s.head));
      sHead.input = head;

      return s;
    }

    template <class output_t>
    stream<output_t> stream<output_t>::filter(streamFilter<output_t> &filter_) {
      return map(filter_);
    }

    template <class output_t>
    stream<output_t>& stream<output_t>::operator >> (output_t &out) {
      if (head && !head->isEmpty()) {
        head->setNext(out);
      }
      return *this;
    }
    //====================================


    //---[ streamMap ]--------------------
    template <class input_t, class output_t>
    streamMap<input_t, output_t>::streamMap() :
      input(NULL) {}

    template <class input_t, class output_t>
    streamMap<input_t, output_t>::~streamMap() {}

    template <class input_t, class output_t>
    bool streamMap<input_t, output_t>::inputIsEmpty() const {
      return (!input || input->isEmpty());
    }

    template <class input_t, class output_t>
    bool streamMap<input_t, output_t>::isEmpty() {
      return inputIsEmpty();
    }

    template <class input_t, class output_t>
    streamMap<input_t, output_t>& streamMap<input_t, output_t>::clone() const {
      streamMap<input_t, output_t>& smap = clone_();
      smap.input = (input
                    ? &(input->clone())
                    : NULL);
      return smap;
    }

    template <class input_t, class output_t>
    void streamMap<input_t, output_t>::clearCache() {
      if (input) {
        input->clearCache();
      }
    }

    template <class input_t, class output_t>
    void* streamMap<input_t, output_t>::passMessageToInput(const occa::json &props) {
      if (input) {
        return input->passMessageToInput(props);
      }
      return NULL;
    }

    template <class input_t, class output_t>
    streamMapFunc<input_t, output_t>::streamMapFunc(output_t (*func_)(const input_t &value)) :
      func(func_) {}

    template <class input_t, class output_t>
    streamMap<input_t, output_t>& streamMapFunc<input_t, output_t>::clone_() const {
      return *(new streamMapFunc<input_t, output_t>(func));
    }

    template <class input_t, class output_t>
    void streamMapFunc<input_t, output_t>::setNext(output_t &out) {
      input_t value;
      (*this->input) >> value;
      out = func(value);
    }
    //====================================


    //---[ streamFilter ]-----------------
    template <class input_t>
    streamFilter<input_t>::streamFilter() :
      streamMap<input_t, input_t>(),
      usedLastValue(true),
      isEmpty_(true) {}

    template <class input_t>
    bool streamFilter<input_t>::isEmpty() {
      if (!usedLastValue) {
        return isEmpty_;
      }

      isEmpty_ = true;

      while (!this->inputIsEmpty()) {
        (*this->input) >> lastValue;

        if (isValid(lastValue)) {
          usedLastValue = false;
          isEmpty_ = false;
          break;
        }
      }
      return isEmpty_;
    }

    template <class input_t>
    void streamFilter<input_t>::setNext(input_t &out) {
      if (!isEmpty()) {
        out = lastValue;
        usedLastValue = true;
      }
    }

    template <class input_t>
    void streamFilter<input_t>::clearCache() {
      streamMap<input_t, input_t>::clearCache();
      usedLastValue = true;
      isEmpty_ = true;
    }

    template <class input_t>
    streamFilterFunc<input_t>::streamFilterFunc(bool (*func_)(const input_t &value)) :
      func(func_) {}

    template <class input_t>
    streamMap<input_t, input_t>& streamFilterFunc<input_t>::clone_() const {
      return *(new streamFilterFunc<input_t>(func));
    }

    template <class input_t>
    bool streamFilterFunc<input_t>::isValid(const input_t &value) {
      return func(value);
    }
    //====================================


    //---[ Cache ]------------------------
    template <class input_t, class output_t>
    withInputCache<input_t, output_t>::withInputCache() :
      inputCache() {}

    template <class input_t, class output_t>
    withInputCache<input_t, output_t>::withInputCache(
      const withInputCache<input_t, output_t> &other
    ) :
      inputCache(other.inputCache) {}

    template <class input_t, class output_t>
    bool withInputCache<input_t, output_t>::inputIsEmpty() const {
      if (inputCache.size()) {
        return false;
      }
      return streamMap<input_t, output_t>::inputIsEmpty();
    }

    template <class input_t, class output_t>
    void withInputCache<input_t, output_t>::pushInput(const input_t &value) {
      inputCache.push_front(value);
    }

    template <class input_t, class output_t>
    void withInputCache<input_t, output_t>::getNextInput(input_t &value) {
      if (inputCache.size()) {
        value = inputCache.front();
        inputCache.pop_front();
      } else {
        (*this->input) >> value;
      }
    }

    template <class input_t, class output_t>
    void withInputCache<input_t, output_t>::clearCache() {
      streamMap<input_t, output_t>::clearCache();
      inputCache.clear();
    }

    template <class input_t, class output_t>
    withOutputCache<input_t, output_t>::withOutputCache() :
      outputCache() {}

    template <class input_t, class output_t>
    withOutputCache<input_t, output_t>::withOutputCache(
      const withOutputCache<input_t, output_t> &other
    ) :
      outputCache(other.outputCache) {}

    template <class input_t, class output_t>
    bool withOutputCache<input_t, output_t>::isEmpty() {
      while (!this->inputIsEmpty() &&
             outputCache.empty()) {
        this->fetchNext();
      }
      return outputCache.empty();
    }

    template <class input_t, class output_t>
    void withOutputCache<input_t, output_t>::setNext(output_t &out) {
      if (!isEmpty()) {
        out = outputCache.front();
        outputCache.pop_front();
      }
    }

    template <class input_t, class output_t>
    void withOutputCache<input_t, output_t>::pushOutput(const output_t &value) {
      outputCache.push_back(value);
    }

    template <class input_t, class output_t>
    void withOutputCache<input_t, output_t>::clearCache() {
      streamMap<input_t, output_t>::clearCache();
      outputCache.clear();
    }

    template <class input_t, class output_t>
    withCache<input_t, output_t>::withCache() {}

    template <class input_t, class output_t>
    withCache<input_t, output_t>::withCache(
      const withCache<input_t, output_t> &other
    ) :
      withInputCache<input_t, input_t>(other),
      withOutputCache<output_t, output_t>(other) {}

    template <class input_t, class output_t>
    void withCache<input_t, output_t>::clearCache() {
      withInputCache<input_t, input_t>::clearCache();
      withOutputCache<output_t, output_t>::clearCache();
    }
    //====================================
  }
}
