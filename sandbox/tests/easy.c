static const int* main(int arg1[1 + 1]){
  int **a;
  *a;

  int &b = a[0][0];
  /* (int**) 3; */
  return 0;
}
