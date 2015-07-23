switch(argc){
  case 1:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0]);
  }
  break;
  case 2:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1]);
  }
  break;
  case 3:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2]);
  }
  break;
  case 4:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3]);
  }
  break;
  case 5:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4]);
  }
  break;
  case 6:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5]);
  }
  break;
  case 7:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
  }
  break;
  case 8:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
  }
  break;
  case 9:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
  }
  break;
  case 10:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]);
  }
  break;
  case 11:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]);
  }
  break;
  case 12:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]);
  }
  break;
  case 13:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]);
  }
  break;
  case 14:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13]);
  }
  break;
  case 15:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14]);
  }
  break;
  case 16:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15]);
  }
  break;
  case 17:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16]);
  }
  break;
  case 18:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17]);
  }
  break;
  case 19:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18]);
  }
  break;
  case 20:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19]);
  }
  break;
  case 21:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20]);
  }
  break;
  case 22:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21]);
  }
  break;
  case 23:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22]);
  }
  break;
  case 24:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23]);
  }
  break;
  case 25:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24]);
  }
  break;
  case 26:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25]);
  }
  break;
  case 27:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26]);
  }
  break;
  case 28:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27]);
  }
  break;
  case 29:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28]);
  }
  break;
  case 30:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29]);
  }
  break;
  case 31:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30]);
  }
  break;
  case 32:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31]);
  }
  break;
  case 33:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32]);
  }
  break;
  case 34:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33]);
  }
  break;
  case 35:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34]);
  }
  break;
  case 36:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35]);
  }
  break;
  case 37:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36]);
  }
  break;
  case 38:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37]);
  }
  break;
  case 39:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38]);
  }
  break;
  case 40:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39]);
  }
  break;
  case 41:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40]);
  }
  break;
  case 42:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41]);
  }
  break;
  case 43:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42]);
  }
  break;
  case 44:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43]);
  }
  break;
  case 45:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44]);
  }
  break;
  case 46:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45]);
  }
  break;
  case 47:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45], args[46]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45], args[46]);
  }
  break;
  case 48:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45], args[46], args[47]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45], args[46], args[47]);
  }
  break;
  case 49:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45], args[46], args[47], args[48]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45], args[46], args[47], args[48]);
  }
  break;
  case 50:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45], args[46], args[47], args[48], args[49]);
  }
  break;}
