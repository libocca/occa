namespace occa {
  namespace testing {
    template <class TM>
    void compare(const TM &a, const TM &b) {
      OCCA_ERROR("Comparing Failed",
                 a == b);
    }

    template <>
    void compare<float>(const float &a, const float &b);

    template <>
    void compare<double>(const double &a, const double &b);
  }
}
