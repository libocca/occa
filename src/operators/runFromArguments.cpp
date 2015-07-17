switch(argumentCount){
  case 1:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0]);
  }
  break;
  case 2:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1]);
  }
  break;
  case 3:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2]);
  }
  break;
  case 4:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3]);
  }
  break;
  case 5:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]);
  }
  break;
  case 6:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
  }
  break;
  case 7:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6]);
  }
  break;
  case 8:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7]);
  }
  break;
  case 9:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8]);
  }
  break;
  case 10:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9]);
  }
  break;
  case 11:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10]);
  }
  break;
  case 12:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11]);
  }
  break;
  case 13:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12]);
  }
  break;
  case 14:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13]);
  }
  break;
  case 15:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14]);
  }
  break;
  case 16:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15]);
  }
  break;
  case 17:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16]);
  }
  break;
  case 18:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17]);
  }
  break;
  case 19:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18]);
  }
  break;
  case 20:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19]);
  }
  break;
  case 21:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20]);
  }
  break;
  case 22:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21]);
  }
  break;
  case 23:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22]);
  }
  break;
  case 24:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23]);
  }
  break;
  case 25:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24]);
  }
  break;
  case 26:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25]);
  }
  break;
  case 27:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26]);
  }
  break;
  case 28:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27]);
  }
  break;
  case 29:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28]);
  }
  break;
  case 30:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29]);
  }
  break;
  case 31:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30]);
  }
  break;
  case 32:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31]);
  }
  break;
  case 33:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32]);
  }
  break;
  case 34:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33]);
  }
  break;
  case 35:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34]);
  }
  break;
  case 36:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35]);
  }
  break;
  case 37:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36]);
  }
  break;
  case 38:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37]);
  }
  break;
  case 39:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38]);
  }
  break;
  case 40:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39]);
  }
  break;
  case 41:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40]);
  }
  break;
  case 42:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41]);
  }
  break;
  case 43:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42]);
  }
  break;
  case 44:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43]);
  }
  break;
  case 45:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44]);
  }
  break;
  case 46:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45]);
  }
  break;
  case 47:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46]);
  }
  break;
  case 48:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47]);
  }
  break;
  case 49:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48]);
  }
  else{
    (*kHandle)(kHandle->nestedKernels, arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48]);
  }
  break;
  case 50:
  if(kHandle->nestedKernelCount == 0){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48], arguments[49]);
  }
  break;}
