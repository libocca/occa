namespace occa {
  namespace testing {
    template <class TM1, class TM2>
    void compare(const TM1 &a, const TM2 &b) {
      OCCA_ERROR("Comparing Failed",
                 a == b);
    }

    template <>
    void compare<float, float>(const float &a, const float &b);

    template <>
    void compare<double, float>(const double &a, const float &b);

    template <>
    void compare<float, double>(const float &a, const double &b);

    template <>
    void compare<double, double>(const double &a, const double &b);
  }
}
