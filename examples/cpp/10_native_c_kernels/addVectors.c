void addVectors(const int *entriesPtr,
                const float *a,
                const float *b,
                float *ab) {
  const int entries = *entriesPtr;

  for (int i = 0; i < entries; ++i) {
    ab[i] = a[i] + b[i];
  }
}
