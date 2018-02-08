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
  baseStream<output_t>::baseStream(baseStream *head_) :
    head(head_) {}

  template <class output_t>
  baseStream<output_t>::baseStream(const baseStream &other) :
    head(other.head) {}

  template <class output_t>
  baseStream<output_t>::~baseStream() {
    delete head;
    head = NULL;
  }

  template <class output_t>
  bool baseStream<output_t>::isContainer() const {
    return true;
  }

  template <class output_t>
  bool baseStream<output_t>::isEmpty() const {
    return (!head || head->isEmpty());
  }

  template <class output_t>
  baseStream<output_t>& baseStream<output_t>::clone() const {
    return *(new baseStream(head
                            ? head->clone()
                            : NULL));
  }

  template <class output_t>
  template <class newOutput_t>
  stream<newOutput_t> baseStream<output_t>::map(
    streamMap<output_t, newOutput_t> *smap
  ) const {
    if (isContainer() && !head) {
      return stream<newOutput_t>(NULL);
    }

    if (isContainer()) {
      smap->input = &(head->clone());
    } else {
      smap->input = &clone();
    }
    return stream<newOutput_t>(smap);
  }

  template <class output_t>
  baseStream<output_t>& baseStream<output_t>::operator >> (output_t &out) {
    if (head) {
      (*head) >> out;
    }
    return *this;
  }
  //====================================


  //---[ stream ]-----------------------
  template <class output_t>
  stream<output_t>::stream(baseStream<output_t> *head_) :
    baseStream<output_t>(head_) {
  }

  template <class output_t>
  stream<output_t>::stream(baseStream<output_t> stream) {
    delete this->head;
    if (stream.isContainer()) {
      this->head = (stream.head
                    ? &(stream.head->clone())
                    : NULL);
    } else {
      this->head = &(stream.clone());
    }
  }

  template <class output_t>
  stream<output_t>& stream<output_t>::operator >> (output_t &out) {
    if (this->head) {
      *(this->head) >> out;
    }
    return *this;
  }
  //====================================


  //---[ streamSource ]-----------------
  template <class output_t>
  bool streamSource<output_t>::isContainer() const {
    return false;
  }

  template <class output_t>
  streamSource<output_t>::operator stream<output_t> () {
    return stream<output_t>(&this->clone());
  }
  //====================================


  //---[ streamMap ]--------------------
  template <class input_t, class output_t>
  bool streamMap<input_t, output_t>::isContainer() const {
    return false;
  }

  template <class input_t, class output_t>
  bool streamMap<input_t, output_t>::inputIsEmpty() const {
    return (!input || input->isEmpty());
  }

  template <class input_t, class output_t>
  bool streamMap<input_t, output_t>::isEmpty() const {
    return inputIsEmpty();
  }

  template <class input_t, class output_t>
  streamMap<input_t, output_t>& streamMap<input_t, output_t>::clone() const {
    streamMap<input_t, output_t>& smap = cloneMap();
    smap.input = (input
                  ? &(input->clone())
                  : NULL);
    return smap;
  }
  //====================================


  //---[ cacheMap ]---------------------
  template <class input_t, class output_t>
  cacheMap<input_t, output_t>::cacheMap() :
    cache() {}

  template <class input_t, class output_t>
  cacheMap<input_t, output_t>::cacheMap(
    const std::queue<output_t> &cache_
  ) :
    cache(cache_) {}

  template <class input_t, class output_t>
  bool cacheMap<input_t, output_t>::isEmpty() const {
    return (cache.empty() || this->inputIsEmpty());
  }

  template <class input_t, class output_t>
  streamMap<input_t, output_t>& cacheMap<input_t, output_t>::operator >> (output_t &out) {
    if (cache.empty()) {
      out = pop();
      // Prioritize values were cached during pop()
      if (!cache.empty()) {
        push(out);
      }
    }
    if (!cache.empty()) {
      out = cache.front();
      cache.pop();
    }
    return *this;
  }

  template <class input_t, class output_t>
  void cacheMap<input_t, output_t>::push(const output_t &value) {
    cache.push(value);
  }
  //====================================
}
