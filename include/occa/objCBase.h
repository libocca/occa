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
           withSizeOf:(uintptr_t) bytes;

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