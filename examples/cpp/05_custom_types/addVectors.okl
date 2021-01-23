struct myFloat {
  float value;
};

typedef struct myFloat2_t {
  float x, y;
} myFloat2;

typedef struct {
  float values[4];
} myFloat4;

@kernel void addVectors(const int entries,
                        const myFloat *a,
                        const myFloat2 *b,
                        myFloat4 *ab) {
  for (int i = 0; i < (entries / 4); ++i; @tile(16, @outer, @inner)) {
    ab[i].values[0] = a[4*i + 0].value + b[2*i + 0].x;
    ab[i].values[1] = a[4*i + 1].value + b[2*i + 0].y;
    ab[i].values[2] = a[4*i + 2].value + b[2*i + 1].x;
    ab[i].values[3] = a[4*i + 3].value + b[2*i + 1].y;
  }
}
