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

namespace occa {
  template <class output_t> class stream;
  template <class output_t> class streamSource;
  template <class input_t, class output_t> class streamMap;

  //---[ stream ]-----------------------
  template <class output_t>
  class stream {
    template<typename> friend class stream;
    template<typename> friend class stream;

  private:
    stream *head;

  public:
    stream(stream *head_ = NULL);
    stream(const stream &other);
    virtual ~stream();

    stream& operator = (const stream &other);

    virtual bool isContainer() const;
    virtual bool isEmpty() const;

    virtual stream& clone() const;

    template <class newOutput_t>
    stream<newOutput_t> map(const streamMap<output_t, newOutput_t> &smap) const;

    virtual stream& operator >> (output_t &out);
  };
  //====================================


  //---[ streamSource ]-----------------
  template <class output_t>
  class streamSource : public stream<output_t> {
  public:
    virtual bool isContainer() const;

    virtual streamSource<output_t>& operator >> (output_t &out) = 0;

    operator stream<output_t> ();
  };
  //====================================


  //---[ streamMap ]--------------------
  template <class input_t,
            class output_t>
  class streamMap : public stream<output_t> {
  public:
    stream<input_t> *input;

    streamMap();

    virtual bool isContainer() const;
    bool inputIsEmpty() const;
    virtual bool isEmpty() const;

    virtual streamMap& operator >> (output_t &out) = 0;

    virtual streamMap& clone() const;
    virtual streamMap& cloneMap() const = 0;
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

    virtual bool isEmpty() const;
    virtual streamMap<input_t, output_t>& operator >> (output_t &out);

    void push(const output_t &value);

    virtual output_t pop() = 0;
  };
  //====================================
}

#include "stream.tpp"

#endif
