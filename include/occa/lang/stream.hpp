/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#ifndef OCCA_LANG_STREAM_HEADER
#define OCCA_LANG_STREAM_HEADER

#include <cstddef>
#include <list>

#include <occa/types.hpp>
#include <occa/tools/properties.hpp>

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

      virtual void* passMessageToInput(const occa::properties &props);
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

      void* passMessageToInput(const occa::properties &props);
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

      virtual void* passMessageToInput(const occa::properties &props);
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
    };
    //====================================
  }
}

#include <occa/lang/stream.tpp>

#endif
