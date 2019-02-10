namespace occa {
  namespace io {
    template <class TM>
    output& output::operator << (const TM &t) {
      if (!customOut) {
        out << t;
      } else {
        ss << t;
        const std::string str = ss.str();
        ss.str("");
        customOut(str.c_str());
      }

      return *this;
    }
  }
}
