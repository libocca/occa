template <class T>
void setupAide::getArgs(string key, T &t){
  vector<T> argv;

  getArgs(key, argv);

  if(argv.size())
    t = argv[0];

  return;
}

template <class T>
void setupAide::getArgs(string key, vector<T> &argv){
  stringstream args;
  T input;

  args.str( getArgs(key) );

  while(args >> input)
    argv.push_back(input);

  return;
}
