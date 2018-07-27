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
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa.hpp>
#include <occa/c/base.h>
#include <occa/c/types.hpp>
#include <occa/tools/sys.hpp>
#include <occa/tools/testing.hpp>

void testGlobals();
void testDevice();
void testStream();

int main(const int argc, const char **argv) {
  srand(time(NULL));

  testGlobals();
  testDevice();
  testStream();
}

void testGlobals() {
  ASSERT_EQ(&occa::c::properties(occaSettings()),
            &occa::settings());
}

void testDevice() {
  ASSERT_EQ(occa::c::device(occaHost()),
            occa::host());

  ASSERT_EQ(occa::c::device(occaGetDevice()),
            occa::getDevice());

  occa::device fakeDevice("mode: 'Serial',"
                          "key: 'value'");
  occaSetDevice(occa::c::newOccaType(fakeDevice));
  ASSERT_EQ(occa::getDevice(),
            fakeDevice);

  occaSetDeviceFromInfo("mode: 'Serial',"
                        "key: 'value'");
  occa::properties &fakeProps = occa::c::properties(occaDeviceProperties());
  ASSERT_EQ((std::string) fakeProps["key"],
            "value");

  occaLoadKernels("lib");

  occaFinish();
}

void testStream() {
  occaStream cStream = occaCreateStream();
  occa::stream stream = occa::c::stream(cStream);

  occaSetStream(cStream);

  ASSERT_EQ(stream.getHandle(),
            occa::getStream().getHandle());

  ASSERT_EQ(stream.getHandle(),
            occa::c::stream(occaGetStream()).getHandle());

  // Start tagging
  double outerStart = occa::sys::currentTime();
  occaStreamTag startTag = occaTagStream();
  double innerStart = occa::sys::currentTime();

  // Wait 0.3 - 0.5 seconds
  ::usleep((3 + (rand() % 3)) * 100000);

  // End tagging
  double innerEnd = occa::sys::currentTime();
  occaStreamTag endTag = occaTagStream();
  occaWaitFor(endTag);
  double outerEnd = occa::sys::currentTime();

  double tagTime = occaTimeBetween(startTag, endTag);

  ASSERT_GE(outerEnd - outerStart,
            tagTime);
  ASSERT_LE(innerEnd - innerStart,
            tagTime);
}
