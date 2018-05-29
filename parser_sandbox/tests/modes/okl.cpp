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
#include "../parserUtils.hpp"

void testLoops();
void testTypes();
void testLoopSkips();

int main(const int argc, const char **argv) {
  parser.addAttribute<dummy>();
  parser.addAttribute<attributes::kernel>();
  parser.addAttribute<attributes::outer>();
  parser.addAttribute<attributes::inner>();
  parser.addAttribute<attributes::shared>();
  parser.addAttribute<attributes::exclusive>();

  testLoops();
  testTypes();
  testLoopSkips();

  return 0;
}

//---[ Loop ]---------------------------
void testOKLLoopExists();
void testProperOKLLoops();
void testInnerInsideOuter();
void testSameInnerLoopCount();

void testLoops() {
  testOKLLoopExists();
  testProperOKLLoops();
  testInnerInsideOuter();
  testSameInnerLoopCount();
}

void testOKLLoopExists() {
  // @outer + @inner exist
}

void testProperOKLLoops() {
  // Proper loops (decl, update, inc)
}

void testInnerInsideOuter() {
  // @outer > @inner
}

void testSameInnerLoopCount() {
  // Same # of @inner in each @outer
}

//======================================

//---[ Types ]--------------------------
void testSharedLocation();
void testExclusiveLocation();
void testValidSharedArray();

void testTypes() {
  testSharedLocation();
  testExclusiveLocation();
  testValidSharedArray();
}

void testSharedLocation() {
  // @outer > @shared > @inner
}

void testExclusiveLocation() {
  // @outer > @exclusive > @inner
}

void testValidSharedArray() {
  // @shared has an array with evaluable sizes
}
//======================================

//---[ Loop Skips ]---------------------
void testValidBreaks();
void testValidContinues();

void testLoopSkips() {
  testValidBreaks();
  testValidContinues();
}

void testValidBreaks() {
  // No break in @outer/@inner (ok inside regular loops inside @outer/@inner)
}

void testValidContinues() {
  // No continue in @inner (ok inside regular loops inside @outer/@inner)
}
//======================================
