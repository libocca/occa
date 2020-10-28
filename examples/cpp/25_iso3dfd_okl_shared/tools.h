/***************************************************************************
 * “INTEL CONFIDENTIAL
 * Copyright (2012)2 (03-2014)3 Intel Corporation All Rights Reserved.
 *
 * The source code contained or described herein and all documents related to 
 * the source code ("Material") are owned by Intel Corporation or its suppliers 
 * or licensors. Title to the Material remains with Intel Corporation or its 
 * suppliers and licensors. The Material contains trade secrets and proprietary 
 * and confidential information of Intel or its suppliers and licensors. The 
 * Material is protected by worldwide copyright and trade secret laws and 
 * treaty provisions. No part of the Material may be used, copied, reproduced, 
 * modified, published, uploaded, posted, transmitted, distributed, or disclosed 
 * in any way without Intel’s prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other intellectual 
 * property right is granted to or conferred upon you by disclosure or delivery 
 * of the Materials, either expressly, by implication, inducement, estoppel or 
 * otherwise. Any license under such intellectual property rights must be express 
 * and approved by Intel in writing.
 * ***************************************************************************/

/*****************************************************************************
 * ! Content:
 * ! Implementation example of ISO-3DFD implementation for 
 * !   Intel(R) Xeon Phi(TM) and Intel(R) Xeon.
 * ! Version 00
 * ! leonardo.borges@intel.com
 * ! cedric.andreolli@intel.com
 * !****************************************************************************/

#ifndef _TOOLS_INCLUDE
#define _TOOLS_INCLUDE

#include <stddef.h>
#include <sys/time.h>
#include <time.h>


// NOTE: the use of clock_gettime() below requires you to link
// with -lrt (the real-time clock)
double walltime() // seconds
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    return ((double)(ts.tv_sec) + 
            1e-09 * (double)(ts.tv_nsec));
}


#if defined(VERIFY_RESULTS)
#include <math.h>
void init_data(float *data, const int dimx, const int dimy, const int dimz)
{
  for(int iz=0; iz<dimz; iz++)
    for(int iy=0; iy<dimy; iy++)
      for(int ix=0; ix<dimx; ix++) {
	*data = (float)iz;
	++data;
      }
}

// naive and slow implementation
void reference_implementation(float *nexti, float *previ, float *coeff, 
		  float *veli,
		  const int n1, const int n2, const int n3, const int half_length){
  int n1n2 = (n1+2*half_length)*n2;
  float* next = &nexti[half_length];  
  float* prev = &previ[half_length];  
  float* vel  = &veli[half_length];  
  for(int iz=half_length; iz<n3-half_length; iz++) {
    for(int iy=half_length; iy<n2-half_length; iy++) {
      for(int ix=0; ix<n1; ix++) {
	  int offset = iz*n1n2 + iy*(n1+2*half_length) + ix;
	  float res = prev[offset]*coeff[0];
	  for(int ir=1; ir<=half_length; ir++) {
	    res += coeff[ir] * (prev[offset + ir] + prev[offset - ir]);	      // horizontal
	    res += coeff[ir] * (prev[offset + ir*(n1+2*half_length)] + prev[offset - ir*(n1+2*half_length)]);   // vertical
	    res += coeff[ir] * (prev[offset + ir*n1n2] + prev[offset - ir*n1n2]); // in front / behind
	  }
	  next[offset] = 2.0f* prev[offset] - next[offset] + res * vel[offset];
      }
    }
  }
}

bool within_epsilon(float* output, float *reference, const int dimx, const int dimy, const int dimz, const int radius, const int zadjust=0, const float delta=0.0001f )
{
  bool retval = true;
  float abs_delta = fabsf(delta);
  for(int iz=0; iz<dimz; iz++) {
    for(int iy=0; iy<dimy; iy++) {
      for(int ix=0; ix<dimx; ix++) {
	if( ix>=radius && ix<(dimx-radius) && iy>=radius && iy<(dimy-radius) && iz>=radius && iz<(dimz-radius+zadjust) ) {
	  float difference = fabsf( *reference - *output);
	  if( difference > delta ) {
	    retval = false;
	    printf(" ERROR: (%d,%d,%d)\t%4.3e instead of %4.3e \n", ix,iy,iz, *output, *reference);
	    return false;
	  }
	}
	++output;
	++reference;
      }
    }
  }
  return retval;
}


#endif /* VERIFY_RESULTS */


#endif /*_TOOLS_INCLUDE */


