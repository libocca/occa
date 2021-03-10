#include <occa/internal/utils/testing.hpp>
#include <occa/internal/lang/stream.hpp>

template <class output_t>
class vectorStream : public occa::lang::baseStream<output_t> {
public:
  std::vector<output_t> values;
  int index;

  vectorStream() :
    occa::lang::baseStream<output_t>(),
    values(),
    index(0) {}

  vectorStream(const std::vector<output_t> &values_,
               const int index_ = 0) :
    values(values_),
    index(index_) {}

  virtual occa::lang::baseStream<output_t>& clone() const {
    return *(new vectorStream(values, index));
  }

  void set(const std::vector<output_t> &values_) {
    values = values_;
    index = 0;
  }

  virtual void* passMessageToInput(const occa::json &props) {
    const std::string inputName = props.get<std::string>("input_name");
    if (inputName == "vectorStream") {
      return (void*) this;
    }
    return NULL;
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
class multMap : public occa::lang::streamMap<input_t, output_t> {
public:
  output_t factor;

  multMap(const output_t &factor_) :
    occa::lang::streamMap<input_t, output_t>(),
    factor(factor_) {}

  virtual occa::lang::streamMap<input_t, output_t>& clone_() const {
    return *(new multMap(factor));
  }

  void set(const output_t factor_) {
    factor = factor_;
  }

  virtual void setNext(output_t &out) {
    input_t in;
    *(this->input) >> in;
    out = (in * factor);
  }
};

template <class input_t, class output_t>
class addHalfMap : public occa::lang::withOutputCache<input_t, output_t> {
public:
  addHalfMap() {}

  virtual occa::lang::streamMap<input_t, output_t>& clone_() const {
    return *(new addHalfMap());
  }

  virtual void fetchNext() {
    input_t value;
    *(this->input) >> value;
    this->pushOutput(value);
    this->pushOutput(value + 0.5);
  }
};

template <class input_t>
class oddNumberFilter : public occa::lang::streamFilter<input_t> {
public:
  oddNumberFilter() {}

  virtual occa::lang::streamMap<input_t, input_t>& clone_() const {
    return *(new oddNumberFilter());
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

  vectorStream<int> source(values);
  occa::lang::stream<int> sInt;
  occa::lang::stream<double> sDouble;

  multMap<int, double> times4(4);
  multMap<double, double> timesQ(0.25);
  addHalfMap<int, double> addHalf;
  oddNumberFilter<int> oddFilter;
  occa::lang::streamMapFunc<double, double> times2(mult2);
  occa::lang::streamFilterFunc<int> evenFilter(isEven);

  sDouble = source.map(times4);

  // Test passMessage
  ASSERT_EQ((void*) NULL,
            sDouble.passMessageToInput({{"input_name", "test"}}));
  ASSERT_EQ((void*) NULL,
            sDouble.getInput("test"));

  vectorStream<int> *vs = (vectorStream<int>*) sDouble.passMessageToInput({{"input_name", "vectorStream"}});
  ASSERT_NEQ((void*) NULL,
             (void*) vs);
  ASSERT_EQ(4, (int) vs->values.size());

  vs = (vectorStream<int>*) sDouble.getInput("vectorStream");
  ASSERT_NEQ((void*) NULL,
             (void*) vs);
  ASSERT_EQ(4, (int) vs->values.size());

  // Test source
  for (int i = 0; i < N; ++i) {
    int value = -1;
    source >> value;
    ASSERT_EQ(i, value);
  }
  ASSERT_TRUE(source.isEmpty());

  // Test map
  source.set(values);
  sDouble = source.map(times4);
  for (int i = 0; i < N; ++i) {
    double value;
    sDouble >> value;
    ASSERT_EQ(4.0 * i, value);
  }

  // Test map composition
  source.set(values);
  sDouble = sDouble.map(timesQ);
  for (int i = 0; i < N; ++i) {
    double value;
    sDouble >> value;
    ASSERT_EQ((i * 4.0) * 0.25, value);
  }

  // Test function map
  source.set(values);
  sDouble = sDouble.map(times2);
  for (int i = 0; i < N; ++i) {
    double value;
    sDouble >> value;
    ASSERT_EQ(((i * 4.0) * 0.25) * 2.0, value);
  }

  // Test cache map
  source.set(values);
  sDouble = source.map(addHalf);
  for (int i = 0; i < (2 * N); ++i) {
    double value;
    sDouble >> value;
    ASSERT_EQ(i * 0.5, value);
  }

  // Test filter
  source.set(values);
  sInt = source.filter(oddFilter);
  for (int i = 0; i < (N / 2); ++i) {
    int value;
    sInt >> value;
    ASSERT_EQ(2 * i, value);
  }

  // Test function filter
  source.set(values);
  sInt = source.filter(evenFilter);
  for (int i = 0; i < (N / 2); ++i) {
    int value;
    sInt >> value;
    ASSERT_EQ(2 * i, value);
  }

  return 0;
}
