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

#include <occa/c/types.hpp>
#include <occa/c/properties.h>

OCCA_START_EXTERN_C

occaType OCCA_RFUNC occaCreateProperties() {
  return occa::c::newOccaType(*(new occa::properties()),
                              true);
}

occaType OCCA_RFUNC occaCreatePropertiesFromString(const char *c) {
  return occa::c::newOccaType(*(new occa::properties(c)),
                              true);
}

occaType OCCA_RFUNC occaPropertiesGet(occaProperties props,
                                      const char *key,
                                      occaType defaultValue) {
  occa::properties& props_ = occa::c::properties(props);
  if (props_.has(key)) {
    return occa::c::newOccaType(props_[key], false);
  }
  return defaultValue;
}

void OCCA_RFUNC occaPropertiesSet(occaProperties props,
                                  const char *key,
                                  occaType value) {
  occa::properties& props_ = occa::c::properties(props);
  props_[key] = occa::c::inferJson(value);
}

int OCCA_RFUNC occaPropertiesHas(occaProperties props,
                                  const char *key) {
  occa::properties& props_ = occa::c::properties(props);
  return props_.has(key);
}

OCCA_END_EXTERN_C
