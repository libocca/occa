switch(argumentCount){
  case 1:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0]); */
      (*(kHandle->nestedKernels[k]))(arguments[0]);
    }
  }
  break;
  case 2:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1]);
    }
  }
  break;
  case 3:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2]);
    }
  }
  break;
  case 4:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3]);
    }
  }
  break;
  case 5:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]);
    }
  }
  break;
  case 6:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
    }
  }
  break;
  case 7:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6]);
    }
  }
  break;
  case 8:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7]);
    }
  }
  break;
  case 9:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8]);
    }
  }
  break;
  case 10:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9]);
    }
  }
  break;
  case 11:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10]);
    }
  }
  break;
  case 12:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11]);
    }
  }
  break;
  case 13:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12]);
    }
  }
  break;
  case 14:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13]);
    }
  }
  break;
  case 15:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14]);
    }
  }
  break;
  case 16:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15]);
    }
  }
  break;
  case 17:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16]);
    }
  }
  break;
  case 18:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17]);
    }
  }
  break;
  case 19:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18]);
    }
  }
  break;
  case 20:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19]);
    }
  }
  break;
  case 21:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20]);
    }
  }
  break;
  case 22:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21]);
    }
  }
  break;
  case 23:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22]);
    }
  }
  break;
  case 24:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23]);
    }
  }
  break;
  case 25:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24]);
    }
  }
  break;
  case 26:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25]);
    }
  }
  break;
  case 27:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26]);
    }
  }
  break;
  case 28:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27]);
    }
  }
  break;
  case 29:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28]);
    }
  }
  break;
  case 30:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29]);
    }
  }
  break;
  case 31:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30]);
    }
  }
  break;
  case 32:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31]);
    }
  }
  break;
  case 33:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32]);
    }
  }
  break;
  case 34:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33]);
    }
  }
  break;
  case 35:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34]);
    }
  }
  break;
  case 36:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35]);
    }
  }
  break;
  case 37:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36]);
    }
  }
  break;
  case 38:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37]);
    }
  }
  break;
  case 39:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38]);
    }
  }
  break;
  case 40:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39]);
    }
  }
  break;
  case 41:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40]);
    }
  }
  break;
  case 42:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41]);
    }
  }
  break;
  case 43:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42]);
    }
  }
  break;
  case 44:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43]);
    }
  }
  break;
  case 45:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44]);
    }
  }
  break;
  case 46:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45]);
    }
  }
  break;
  case 47:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46]);
    }
  }
  break;
  case 48:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47]);
    }
  }
  break;
  case 49:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48]);
    }
  }
  break;
  case 50:
  if(kHandle->nestedKernelCount == 1){
    (*kHandle)(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48], arguments[49]);
  }
  else{
    for(int k = 0; k < kHandle->nestedKernelCount; ++k){
      /* (*(kHandle->setDimsKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48], arguments[49]); */
      (*(kHandle->nestedKernels[k]))(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5], arguments[6], arguments[7], arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13], arguments[14], arguments[15], arguments[16], arguments[17], arguments[18], arguments[19], arguments[20], arguments[21], arguments[22], arguments[23], arguments[24], arguments[25], arguments[26], arguments[27], arguments[28], arguments[29], arguments[30], arguments[31], arguments[32], arguments[33], arguments[34], arguments[35], arguments[36], arguments[37], arguments[38], arguments[39], arguments[40], arguments[41], arguments[42], arguments[43], arguments[44], arguments[45], arguments[46], arguments[47], arguments[48], arguments[49]);
    }
  }
  break;}