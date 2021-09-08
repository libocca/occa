extern "C" void addVectors(const int &entries,
                           const float *a,
                           const float *b,
                           float *ab) {

  for (int i = 0; i < entries; ++i) {
    ab[i] = a[i] + b[i];
  }
}
