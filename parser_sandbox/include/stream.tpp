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

namespace occa {
  //---[ baseStream ]-------------------
  template <class output_t>
  baseStream<output_t>::~baseStream() {}

  template <class output_t>
  template <class newOutput_t>
  stream<newOutput_t> baseStream<output_t>::map(
    const streamMap<output_t, newOutput_t> &smap
  ) const {

    stream<output_t> s(*this);
    return s.map(smap);
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
  stream<output_t>::stream(const baseStream<output_t> &head_) :
    head(&(head_.clone())) {}

  template <class output_t>
  stream<output_t>::stream(const stream &other) :
    head(NULL) {
    *this = other;
  }

  template <class output_t>
  stream<output_t>::~stream() {
    delete head;
  }

  template <class output_t>
  stream<output_t>& stream<output_t>::operator = (const stream &other) {
    delete this->head;

    this->head = (other.head
                  ? &(other.head->clone())
                  : NULL);

    return *this;
  }

  template <class output_t>
  bool stream<output_t>::isEmpty() {
    return (!head || head->isEmpty());
  }

  template <class output_t>
  stream<output_t>& stream<output_t>::clone() const {
    if (!head) {
      return *(new stream());
    }
    return *(new stream(*head));
  }

  template <class output_t>
  template <class newOutput_t>
  stream<newOutput_t> stream<output_t>::map(
    const streamMap<output_t, newOutput_t> &map_
  ) const {
    if (!head) {
      return stream<newOutput_t>();
    }

    typedef streamMap<output_t, newOutput_t> mapType;

    stream<newOutput_t> s(map_);
    mapType &sHead = *(static_cast<mapType*>(s.head));
    sHead.input = &(head->clone());

    return s;
  }

  template <class output_t>
  stream<output_t> stream<output_t>::filter(
    const streamFilter<output_t> &filter_
  ) const {

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
  streamMap<input_t, output_t>::~streamMap() {
    delete input;
  }

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
  //====================================


  //---[ cacheMap ]---------------------
  template <class input_t, class output_t>
  cacheMap<input_t, output_t>::cacheMap() :
    cache() {}

  template <class input_t, class output_t>
  cacheMap<input_t, output_t>::cacheMap(
    const cacheMap<input_t, output_t> &map
  ) :
    cache(map.cache) {}

  template <class input_t, class output_t>
  bool cacheMap<input_t, output_t>::isEmpty() {
    if (!cache.empty()) {
      return false;
    }
    if (this->inputIsEmpty()) {
      return true;
    }
    this->pop();
    return cache.empty();
  }

  template <class input_t, class output_t>
  void cacheMap<input_t, output_t>::setNext(output_t &out) {
    if (!isEmpty()) {
      out = cache.front();
      cache.pop();
    }
  }

  template <class input_t, class output_t>
  void cacheMap<input_t, output_t>::push(const output_t &value) {
    cache.push(value);
  }
  //====================================
}
