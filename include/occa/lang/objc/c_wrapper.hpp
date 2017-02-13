/* The MIT License (MIT)
 * 
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#ifndef OCCA_OBJC_BASE_HEADER
#define OCCA_OBJC_BASE_HEADER

#import <Foundation/NSString.h>

#include "occaCBase.hpp"

@interface OCCAType : NSObject {
  occaType handle;
}

- (id) initWithInt:(int)           value;
- (id) initWithUInt:(unsigned int) value;

- (id) initWithChar:(char)           value;
- (id) initWithUChar:(unsigned char) value;

- (id) initWithShort:(short)           value;
- (id) initWithUShort:(unsigned short) value;

- (id) initWithLong:(long)           value;
- (id) initWithULong:(unsigned long) value;

- (id) initWithFloat:(float)   value;
- (id) initWithDouble:(double) value;

- (id) initWithStruct:(void*)     value
           withSizeOf:(udim_t) bytes;

- (id) initWithString:(const char*) str;
@end

@interface OCCA : NSObject {
  + (void) printAvailableDevices();
  +
}
@end

@interface OCCAKernel : NSObject {
  occaKernel handle;
}
@end

@interface OCCADevice : NSObject {
  occaDevice handle;
}
@end

@interface OCCAMemory : NSObject {
  occaMemory handle;
}
@end

#endif