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
class vectorStream : public occa::baseStream<output_t> {
public:
  std::vector<output_t> values;
  int index;

  vectorStream() :
    occa::baseStream<output_t>(),
    values(),
    index(0) {}

  vectorStream(const std::vector<output_t> &values_,
               const int index_ = 0) :
    values(values_),
    index(index_) {}

  virtual occa::baseStream<output_t>& clone() const {
    return *(new vectorStream(values, index));
  }

  virtual bool isEmpty() {
    return (index >= (int) values.size());
  }

  virtual void setNext(output_t &out) {
    const int size = (int) values.size();
    if (index < size) {
      out = values[index++];
    }
  }
};

template <class input_t,
          class output_t>
class multMap : public occa::streamMap<input_t, output_t> {
public:
  output_t factor;

  multMap(output_t factor_) :
    occa::streamMap<input_t, output_t>(),
    factor(factor_) {}

  virtual occa::streamMap<input_t, output_t>& clone_() const {
    return *(new multMap(factor));
  }

  virtual void setNext(output_t &out) {
    input_t in;
    *(this->input) >> in;
    out = (in * factor);
  }
};

template <class input_t, class output_t>
class addHalfMap : public occa::cacheMap<input_t, output_t> {
public:
  addHalfMap() {}

  virtual occa::streamMap<input_t, output_t>& clone_() const {
    return *(new addHalfMap());
  }

  virtual void pop() {
    input_t value;
    *(this->input) >> value;
    this->push(value);
    this->push(value + 0.5);
  }
};

template <class input_t>
class oddFilter : public occa::streamFilter<input_t> {
public:
  oddFilter() {}

  virtual occa::streamMap<input_t, input_t>& clone_() const {
    return *(new oddFilter());
  }

  virtual bool isValid(const input_t &value) {
    return !(value % 2);
  }
};

double mult2(const double &value) {
  return 2.0 * value;
}

bool isEven(const int &value) {
  return !(value % 2);
}

int main(const int argc, const char **argv) {
  std::vector<int> values;
  const int N = 4;
  for (int i = 0; i < N; ++i) {
    values.push_back(i);
  }

  occa::stream<int> s            = vectorStream<int>(values);
  occa::stream<double> sTimes4   = s.map(multMap<int, double>(4));
  occa::stream<double> sTimes1   = sTimes4.map(multMap<double, double>(0.25));
  occa::stream<double> sTimes2   = sTimes1.map(mult2);
  occa::stream<double> sPlusHalf = s.map(addHalfMap<int, double>());
  occa::stream<int> sEven        = s.filter(oddFilter<int>());
  occa::stream<int> sEvenFunc    = s.filter(isEven);

  // Test source
  for (int i = 0; i < N; ++i) {
    int value = -1;
    s >> value;
    OCCA_ASSERT_EQUAL(i, value);
  }
  OCCA_ASSERT_TRUE(s.isEmpty());
  OCCA_ASSERT_FALSE(sTimes4.isEmpty());
  OCCA_ASSERT_FALSE(sTimes1.isEmpty());

  // Test map
  for (int i = 0; i < N; ++i) {
    double value;
    sTimes4 >> value;
    OCCA_ASSERT_EQUAL(4.0 * i, value);
  }
  OCCA_ASSERT_TRUE(sTimes4.isEmpty());
  OCCA_ASSERT_FALSE(sTimes1.isEmpty());

  // Test map composition
  for (int i = 0; i < N; ++i) {
    double value;
    sTimes1 >> value;
    OCCA_ASSERT_EQUAL((i * 4.0) * 0.25, value);
  }
  OCCA_ASSERT_TRUE(sTimes1.isEmpty());

  // Test function map
  for (int i = 0; i < N; ++i) {
    double value;
    sTimes2 >> value;
    OCCA_ASSERT_EQUAL(((i * 4.0) * 0.25) * 2.0, value);
  }
  OCCA_ASSERT_TRUE(sTimes2.isEmpty());

  // Test cache map
  for (int i = 0; i < (2 * N); ++i) {
    double value;
    sPlusHalf >> value;
    OCCA_ASSERT_EQUAL(i * 0.5, value);
  }
  OCCA_ASSERT_TRUE(sPlusHalf.isEmpty());

  // Test filter
  for (int i = 0; i < (N / 2); ++i) {
    int value;
    sEven >> value;
    OCCA_ASSERT_EQUAL(2 * i, value);
  }
  OCCA_ASSERT_TRUE(sEven.isEmpty());

  // Test function filter
  for (int i = 0; i < (N / 2); ++i) {
    int value;
    sEvenFunc >> value;
    OCCA_ASSERT_EQUAL(2 * i, value);
  }
  OCCA_ASSERT_TRUE(sEvenFunc.isEmpty());

  return 0;
}
