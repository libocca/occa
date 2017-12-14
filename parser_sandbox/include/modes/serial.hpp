#if 0
#ifndef OCCA_PARSER_MODES_SERIAL_HEADER2
#define OCCA_PARSER_MODES_SERIAL_HEADER2

namespace occa {
  namespace lang {
    class serialBackend : public backend {
    public:
      virtual void transform(statement_t &root,
                             const properties &props = "");
    };
  }
}

#endif
#endif
