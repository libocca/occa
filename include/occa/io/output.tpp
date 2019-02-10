namespace occa {
  namespace io {
    template <class TM>
    output& output::operator << (const TM &t) {
      if (!overrideOut) {
        out << t;
      } else {
        ss << t;
        const std::string str = ss.str();
        ss.str("");
        overrideOut(str.c_str());
      }

      return *this;
    }
  }
}
