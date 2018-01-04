#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "occa/tools/gc.hpp"

class test : public occa::withRef {
public:
  static int count;

  test(const int idx) {}

  test(const test &other) :
    occa::withRef(other) {}

  test& operator = (const test &other) {
    changeRef(other);
    return *this;
  }

  ~test() {
    removeRef();
  }

  virtual void destructor() {
    --count;
  }
};

int test::count = 2;

int main(const int argc, const char **argv) {
  {
    test a1(1), a2(a1), a3(a1), a4(a1), a5(a1), a6(a4);
    test b1(2), b2(b1), b3(b1), b4(b1), b5(b1), b6(b4);

    a1 = b5;
    a2 = b2;
    a3 = b2;
    a4 = b3;
    a5 = b3;

    b1 = a5;
    b2 = a4;
    b3 = a3;
    b4 = a2;
    b5 = a1;
    OCCA_ERROR("Oh oh... some died: " << (2 - test::count),
               test::count == 2);
  }

  OCCA_ERROR("Oh oh... left alive: " << test::count,
             test::count == 0);
}
