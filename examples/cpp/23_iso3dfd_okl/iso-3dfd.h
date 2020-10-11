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
! Content:
! Implementation example of ISO-3DFD implementation for 
!   Intel(R) Xeon Phi(TM) and Intel(R) Xeon.
! Version 00
! leonardo.borges@intel.com
! cedric.andreolli@intel.com
!****************************************************************************/



#ifndef _ISO_3DFD_INCLUDE
#define _ISO_3DFD_INCLUDE

#include <occa.hpp>
#include <occa/types/fp.hpp>

#define HALF_LENGTH 8

/*** verify if stencil half lenght is properly defined */
#if !( (HALF_LENGTH == 4) || (HALF_LENGTH == 8) ) 
#error "HALF_LENGTH must be defined (either 4 or 8)"
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEILING(X) (X-(int)(X) > 0 ? (int)(X+1) : (int)(X))

void iso_3dfd_stencil_BLK(float *ptr_next,  float *ptr_prev,  float *ptr_vel, float *coeff,
			  const int i1_idx, const int i2_idx, const int i3_idx,
			  const int n1,     const int n2,     const int n3,
			  const int n1_Tb,  const int n2_Tb,  const int n3_Tb);

void iso_3dfd(float *next,  float *prev,  float *vel,   float *coeff,
	      const int n1, const int n2, const int n3, const int num_threads, const int nreps,
	      const int n1_Tblock, const int n2_Tblock, const int n3_Tblock);


#endif /* _ISO_3DFD_INCLUDE */
