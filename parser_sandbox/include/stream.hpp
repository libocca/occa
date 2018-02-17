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
#ifndef OCCA_STREAM_HEADER2
#define OCCA_STREAM_HEADER2

#include <cstddef>
#include <queue>

#include "occa/types.hpp"

namespace occa {
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

    virtual bool isEmpty() = 0;

    virtual baseStream& clone() const = 0;

    virtual void setNext(output_t &out) = 0;

    // Map
    template <class newOutput_t>
    stream<newOutput_t> map(const streamMap<output_t, newOutput_t> &smap) const;

    template <class newOutput_t>
    stream<newOutput_t> map(newOutput_t (*func)(const output_t &value)) const;

    // Filter
    stream<output_t> filter(const streamFilter<output_t> &sfilter) const;

    stream<output_t> filter(bool (*func)(const output_t &value)) const;

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
    stream(const baseStream<output_t> &head_);
    stream(const stream &other);
    virtual ~stream();

    stream& operator = (const stream &other);

    bool isEmpty();

    stream& clone() const;

    // Map
    template <class newOutput_t>
    stream<newOutput_t> map(const streamMap<output_t, newOutput_t> &map_) const;

    template <class newOutput_t>
    stream<newOutput_t> map(newOutput_t (*func)(const output_t &value)) const;

    // Filter
    stream<output_t> filter(const streamFilter<output_t> &filter_) const;

    stream<output_t> filter(bool (*func)(const output_t &value)) const;

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

    bool inputIsEmpty() const;
    virtual bool isEmpty();

    virtual streamMap& clone() const;
    virtual streamMap& clone_() const = 0;
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


  //---[ cacheMap ]---------------------
  template <class input_t,
            class output_t>
  class cacheMap : public streamMap<input_t, output_t> {
  public:
    std::queue<output_t> cache;

    cacheMap();
    cacheMap(const cacheMap<input_t, output_t> &map);

    virtual bool isEmpty();

    virtual void setNext(output_t &out);

    void push(const output_t &value);

    virtual void pop() = 0;
  };
  //====================================
}

#include "stream.tpp"

#endif
