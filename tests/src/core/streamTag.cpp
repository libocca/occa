#include <occa.hpp>
#include <occa/internal/utils/testing.hpp>

void testTagStreamAndWait();
void testTimeBetween();
void testUnwrap();

int main(const int argc, const char **argv) {
  testTagStreamAndWait();
  testTimeBetween();
  testUnwrap();

  return 0;
}

void testTagStreamAndWait() {
 occa::device occa_device({
    {"mode", "Serial"}
  });

  occa::streamTag occa_stream_tag = occa_device.tagStream();

  ASSERT_TRUE(occa_stream_tag.getDevice() == occa_device);

  occa_stream_tag.wait();
}

void testTimeBetween() {
  occa::device occa_device({
    {"mode", "Serial"}
  });

  occa::streamTag start_tag = occa_device.tagStream();
  occa::streamTag finish_tag = occa_device.tagStream();

  ASSERT_EQ(0.0,occa_device.timeBetween(start_tag,start_tag));
  ASSERT_EQ(0.0,occa_device.timeBetween(finish_tag,finish_tag));
  ASSERT_TRUE(0.0 < occa_device.timeBetween(start_tag,finish_tag));
}

void testUnwrap() {
  occa::device occa_device({
    {"mode","Serial"}
  });

  occa::streamTag occa_stream_tag;

  // Unwrapping an uninitialized streamTag is undefined
  ASSERT_THROW(occa::unwrap(occa_stream_tag););

  occa_stream_tag = occa_device.tagStream();

  // Unwrapping a serial mode streamTag is undefined
  ASSERT_THROW(occa::unwrap(occa_stream_tag););
}
