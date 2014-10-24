const struct a_t {
  int a1;
  struct b {
    int b1, c1;
  };
} a[2], b[3];

static const int* main(int arg1[1 + 1]){
  int **a, *a2;
  *a;

  int &b = a[0][0];
  /* (int**) 3; */
  return 0;
}
