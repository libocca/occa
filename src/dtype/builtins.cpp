#include <occa/dtype/builtins.hpp>

namespace occa {
  namespace dtypes {
    dtype none("none", 0);

    dtype void_("void", 0);
    dtype byte("byte", 1);

    dtype bool_("bool", sizeof(bool));
    dtype char_("char", sizeof(char));
    dtype short_("short", sizeof(short));
    dtype int_("int", sizeof(int));
    dtype long_("long", sizeof(long));
    dtype float_("float", sizeof(float));
    dtype double_("double", sizeof(double));

    dtype int8("int8", 1);
    dtype uint8("uint8", 1);
    dtype int16("int16", 2);
    dtype uint16("uint16", 2);
    dtype int32("int32", 4);
    dtype uint32("uint32", 4);
    dtype int64("int64", 8);
    dtype uint64("uint64", 8);
    dtype float32("float32", 4);
    dtype float64("float64", 8);
  }
}
