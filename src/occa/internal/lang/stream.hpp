#ifndef OCCA_INTERNAL_LANG_STREAM_HEADER
#define OCCA_INTERNAL_LANG_STREAM_HEADER

#include <cstddef>
#include <list>

#include <occa/types.hpp>
#include <occa/types/json.hpp>

namespace occa {
  namespace lang {
    template <class output_t>
    class stream;
    template <class input_t, class output_t>
    class streamMap;
    template <class input_t>
    class streamFilter;

    //---[ baseStream ]-------------------
    template <class output_t>
    class baseStream {
     public:
      virtual ~baseStream();

      virtual baseStream& clone() const = 0;

      virtual bool isEmpty() = 0;

      virtual void setNext(output_t &out) = 0;

      virtual void clearCache();
      virtual void* passMessageToInput(const occa::json &props);
      void* getInput(const std::string &name);

      template <class newOutput_t>
      stream<newOutput_t> map(streamMap<output_t, newOutput_t> &smap);

      stream<output_t> filter(streamFilter<output_t> &sfilter);

      baseStream& operator >> (output_t &out);
    };
    //====================================

    //---[ stream ]-----------------------
    template <class output_t>
    class stream {
      template<typename>
      friend class baseStream;

      template<typename>
      friend class stream;

     private:
      baseStream<output_t> *head;

     public:
      stream();
      stream(baseStream<output_t> &head_);
      stream(const stream &other);
      virtual ~stream();

      stream& operator = (const stream &other);

      stream& clone() const;

      void clearCache();
      void* passMessageToInput(const occa::json &props);
      void* getInput(const std::string &name);

      bool isEmpty();

      template <class newOutput_t>
      stream<newOutput_t> map(streamMap<output_t, newOutput_t> &map_);

      stream<output_t> filter(streamFilter<output_t> &filter_);

      stream& operator >> (output_t &out);
    };
    //====================================


    //---[ streamMap ]--------------------
    template <class input_t,
              class output_t>
    class streamMap : public baseStream<output_t> {
     public:
      baseStream<input_t> *input;

      streamMap();
      ~streamMap();

      virtual bool inputIsEmpty() const;
      virtual bool isEmpty();

      virtual streamMap& clone() const;
      virtual streamMap& clone_() const = 0;

      virtual void clearCache();
      virtual void* passMessageToInput(const occa::json &props);
    };

    template <class input_t,
              class output_t>
    class streamMapFunc : public streamMap<input_t, output_t> {
     public:
      output_t (*func)(const input_t &value);

      streamMapFunc(output_t (*func_)(const input_t &value));

      virtual streamMap<input_t, output_t>& clone_() const;

      virtual void setNext(output_t &out);
    };
    //====================================


    //---[ streamFilter ]-----------------
    template <class input_t>
    class streamFilter : public streamMap<input_t, input_t> {
     public:
      input_t lastValue;
      bool usedLastValue;
      bool isEmpty_;

      streamFilter();

      virtual bool isEmpty();

      virtual void setNext(input_t &out);
      virtual bool isValid(const input_t &value) = 0;

      virtual void clearCache();
    };

    template <class input_t>
    class streamFilterFunc : public streamFilter<input_t> {
     public:
      bool (*func)(const input_t &value);

      streamFilterFunc(bool (*func_)(const input_t &value));

      virtual streamMap<input_t, input_t>& clone_() const;

      virtual bool isValid(const input_t &value);
    };
    //====================================


    //---[ Cache ]------------------------
    template <class input_t, class output_t>
    class withInputCache : virtual public streamMap<input_t, output_t> {
     public:
      std::list<input_t> inputCache;

      withInputCache();
      withInputCache(const withInputCache<input_t, output_t> &other);

      virtual bool inputIsEmpty() const;

      void pushInput(const input_t &value);

      void getNextInput(input_t &value);

      virtual void clearCache();
    };

    template <class input_t, class output_t>
    class withOutputCache : virtual public streamMap<input_t, output_t> {
     public:
      std::list<output_t> outputCache;

      withOutputCache();
      withOutputCache(const withOutputCache<input_t, output_t> &other);

      virtual bool isEmpty();

      virtual void setNext(output_t &out);

      void pushOutput(const output_t &value);

      virtual void fetchNext() = 0;

      virtual void clearCache();
    };

    template <class input_t, class output_t>
    class withCache : public withInputCache<input_t, input_t>,
                      public withOutputCache<output_t, output_t> {
     public:
      withCache();
      withCache(const withCache<input_t, output_t> &other);

      virtual void clearCache();
    };
    //====================================
  }
}

#include "stream.tpp"

#endif
