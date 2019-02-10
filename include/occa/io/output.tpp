namespace occa {
  namespace io {
    template <class TM>
    output_t& output_t::operator << (const TM &t) {
      if (!customOut) {
        out << t;
      } else {
        // Cast value to string only once
        ss << t;
        const std::string str = ss.str();
        ss.str("");

        out << str;
        customOut(str.c_str());
      }

      return *this;
    }
  }
}
