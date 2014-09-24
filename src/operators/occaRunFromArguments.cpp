switch(argumentCount){
  case 1:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0]);
      (*nestedKernels[k])(arguments[0]);
    }
  }
  break;
  case 2:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1]);
      (*nestedKernels[k])(arguments[0], arguments[1]);
    }
  }
  break;
  case 3:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2]);
    }
  }
  break;
  case 4:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3]);
    }
  }
  break;
  case 5:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]);
    }
  }
  break;
  case 6:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
    }
  }
  break;
  case 7:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6]);
    }
  }
  break;
  case 8:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7]);
    }
  }
  break;
  case 9:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8]);
    }
  }
  break;
  case 10:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9]);
    }
  }
  break;
  case 11:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10]);
    }
  }
  break;
  case 12:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11]);
    }
  }
  break;
  case 13:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12]);
    }
  }
  break;
  case 14:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13]);
    }
  }
  break;
  case 15:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14]);
    }
  }
  break;
  case 16:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15]);
    }
  }
  break;
  case 17:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16]);
    }
  }
  break;
  case 18:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17]);
    }
  }
  break;
  case 19:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18]);
    }
  }
  break;
  case 20:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19]);
    }
  }
  break;
  case 21:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20]);
    }
  }
  break;
  case 22:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21]);
    }
  }
  break;
  case 23:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22]);
    }
  }
  break;
  case 24:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23]);
    }
  }
  break;
  case 25:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24]);
    }
  }
  break;
  case 26:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25]);
    }
  }
  break;
  case 27:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26]);
    }
  }
  break;
  case 28:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27]);
    }
  }
  break;
  case 29:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28]);
    }
  }
  break;
  case 30:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29]);
    }
  }
  break;
  case 31:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30]);
    }
  }
  break;
  case 32:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31]);
    }
  }
  break;
  case 33:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32]);
    }
  }
  break;
  case 34:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33]);
    }
  }
  break;
  case 35:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34]);
    }
  }
  break;
  case 36:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35]);
    }
  }
  break;
  case 37:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36]);
    }
  }
  break;
  case 38:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37]);
    }
  }
  break;
  case 39:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38]);
    }
  }
  break;
  case 40:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39]);
    }
  }
  break;
  case 41:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40]);
    }
  }
  break;
  case 42:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41]);
    }
  }
  break;
  case 43:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42]);
    }
  }
  break;
  case 44:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43]);
    }
  }
  break;
  case 45:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44]);
    }
  }
  break;
  case 46:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45]);
    }
  }
  break;
  case 47:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46]);
    }
  }
  break;
  case 48:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47]);
    }
  }
  break;
  case 49:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48]);
    }
  }
  break;
  case 50:
  if(nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48], arguments[49]);
  }
  else{
    for(int k = 0; k < nestedKernelCount; ++k){
      (*setDimsKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48], arguments[49]);
      (*nestedKernels[k])(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48], arguments[49]);
    }
  }
  break;}