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
#include "occa/tools/testing.hpp"
#include "stream.hpp"

template <class output_t>
class vectorStream : public occa::streamSource<output_t> {
public:
  std::vector<output_t> values;
  int index;

  vectorStream() :
    values(),
    index(0) {}

  vectorStream(const std::vector<output_t> &values_,
               const int index_ = 0) :
    values(values_),
    index(index_) {}

  operator occa::stream<output_t> () {
    return occa::stream<output_t>(&clone());
  }

  virtual occa::baseStream<output_t>& clone() const {
    return *(new vectorStream(values, index));
  }

  virtual bool isEmpty() const {
    return (index >= (int) values.size());
  }

  virtual occa::streamSource<output_t>& operator >> (output_t &out) {
    const int size = (int) values.size();
    if (index < size) {
      out = values[index++];
    }
    return *this;
  }
};

template <class input_t,
          class output_t>
class multMap : public occa::streamMap<input_t, output_t> {
public:
  output_t factor;

  multMap(output_t factor_) :
    factor(factor_) {}

  virtual occa::streamMap<input_t, output_t>& cloneMap() const {
    return *(new multMap(factor));
  }

  virtual occa::streamMap<input_t, output_t>& operator >> (output_t &out) {
    if (!this->inputIsEmpty()) {
      input_t in;
      *(this->input) >> in;
      out = (in * factor);
    }
    return *this;
  }
};

template <class output_t>
class duplicateMap : public occa::cacheMap<output_t, output_t> {
public:
  duplicateMap() {}

  virtual occa::streamMap<output_t, output_t>& cloneMap() const {
    return *(new duplicateMap());
  }

  virtual output_t pop() {
    output_t value;
    *(this->input) >> value;
    push(value);
    return value;
  }
};

int main(const int argc, const char **argv) {
  std::vector<int> values;
  for (int i = 0; i < 3; ++i) {
    values.push_back(i);
  }

  occa::stream<int> s          = vectorStream<int>(values);
  occa::stream<double> sTimes4 = s.map(new multMap<int, double>(4));
  occa::stream<double> sTimes1 = sTimes4.map(new multMap<double, double>(0.25));
  occa::stream<int> s2         = s.map(new duplicateMap());

  double value;

  // Test source
  for (int i = 0; i < 3; ++i) {
    s >> value;
    OCCA_ASSERT_EQUAL(i, value);
  }
  OCCA_ASSERT_TRUE(s.isEmpty());
  OCCA_ASSERT_FALSE(sTimes4.isEmpty());
  OCCA_ASSERT_FALSE(sTimes1.isEmpty());

  // Test map
  for (int i = 0; i < 3; ++i) {
    s >> value;
    OCCA_ASSERT_EQUAL(4.0 * i, value);
  }
  OCCA_ASSERT_TRUE(sTimes4.isEmpty());
  OCCA_ASSERT_FALSE(sTimes1.isEmpty());

  // Test map composition
  for (int i = 0; i < 3; ++i) {
    s >> value;
    OCCA_ASSERT_EQUAL((i * 4.0) * 0.25, value);
  }
  OCCA_ASSERT_TRUE(sTimes1.isEmpty());

  // Test cache map
  for (int i = 0; i < 6; ++i) {
    s >> value;
    OCCA_ASSERT_EQUAL(i / 2, value);
  }
  OCCA_ASSERT_TRUE(s2.isEmpty());

  return 0;
}