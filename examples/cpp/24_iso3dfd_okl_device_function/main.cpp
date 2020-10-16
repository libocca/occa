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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "iso-3dfd.h"
#include "tools.h"


occa::json parseArgs(int argc, const char **argv);



typedef struct{
	int n1;   	// First dimension
	int n2;   	// Second dimension
	int n3;   	// Third dimension
	int num_threads;
	int nreps;     	// number of time-steps, over which performance is averaged
	int n1_Tblock;	// Thread blocking on 1st dimension
	int n2_Tblock;	// Thread blocking on 2nd dimension
	int n3_Tblock;	// Thread blocking on 3rd dimension
	float *prev;	
	float *next;
	float *vel;
}Parameters; 

//Function used for initialization
void initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, Parameters* p, size_t nbytes){
        memset(ptr_prev, 0.0f, nbytes);
        memset(ptr_next, 0.0f, nbytes);
        memset(ptr_vel, 2250000.0f*DT*DT, nbytes);
	//Then we add a source
        float val = 1.1f;
        for(int s=5; s>=0; s--){
                for(int i=p->n3/2-s; i<p->n3/2+s;i++){
                        for(int j=p->n2/4-s; j<p->n2/4+s;j++){
                                for(int k=p->n1/4-s; k<p->n1/4+s;k++){
                                        ptr_prev[i*p->n1*p->n2 + j*p->n1 + k] = val;
                                }
                        }
                }
                val *= 1.2f;
       }
}

int main(int argc, const char** argv)
{
	// Defaults
	occa::json args = parseArgs(argc, argv);

	Parameters p;
	p.n1 = 256;   // First dimension
  	p.n2 = 300;   // Second dimension
  	p.n3 = 300;   // Third dimension
  	p.num_threads = 4;
  	p.nreps = 100;     // number of time-steps, over which performance is averaged
  	p.n1_Tblock=256;       // Thread blocking on 1st dimension
  	p.n2_Tblock=4;       // Thread blocking on 2nd dimension
  	p.n3_Tblock=2;       // Thread blocking on 3rd dimension
# define N2_TBLOCK 1   // Default thread blocking on 2nd dimension: 1
# define N3_TBLOCK 124 // Default thread blocking on 3rd dimension: 124
  
  	if( (argc > 1) && (argc < 4) ) {
    		printf(" usage: [n1 n2 n3] [# threads] [# iterations] [thread block n1] [thread block n2] [thread block n3]\n");
    		exit(1);
  	}
  	// [n1 n2 n3]
  	if( argc >= 4 ) {
    		p.n1 = atoi(argv[1]);
    		p.n2 = atoi(argv[2]);
    		p.n3 = atoi(argv[3]);
  	}
  	//  [# iterations]
  	if( argc >= 5)
    		p.nreps = atoi(argv[4]);
  	//  [thread block n1] [thread block n2] [thread block n3]
  	if( argc >= 6) {
    		p.n1_Tblock = atoi(argv[5]);
  	} else {
    		p.n1_Tblock = p.n1; // Default: no blocking on 1st dim
  	}
  	if( argc >= 7) {
    		p.n2_Tblock = atoi(argv[6]);
  	} else {
    		p.n2_Tblock =  N2_TBLOCK;
  	}
  	if( argc >= 8) {
    		p.n3_Tblock = atoi(argv[7]);
  	} else {
    		p.n3_Tblock = N3_TBLOCK;
  	}
  
  	// Make sure n1 and n1_Tblock are multiple of 16 (to support 64B alignment)
  	if ((p.n1%16)!=0) {
    		printf("Parameter n1=%d must be a multiple of 16\n",p.n1);
    		exit(1);
  	}
  	if ((p.n1_Tblock%16)!=0) {
    		printf("Parameter n1_Tblock=%d must be a multiple of 16\n",p.n1_Tblock);
    		exit(1);
  	}
  	// Make sure nreps is rouded up to next even number (to support swap)
  	p.nreps = ((p.nreps+1)/2)*2;


  	printf("n1=%d n2=%d n3=%d nreps=%d num_threads=%d HALF_LENGTH=%d\n",p.n1,p.n2,p.n3,p.nreps,p.num_threads,HALF_LENGTH);
  	printf("n1_thrd_block=%d n2_thrd_block=%d n3_thrd_block=%d\n", p.n1_Tblock, p.n2_Tblock, p.n3_Tblock);


        float coeff[HALF_LENGTH+1] = {
                        -3.0548446,
                        +1.7777778,
                        -3.1111111e-1,
                        +7.572087e-2,
                        -1.76767677e-2,
                        +3.480962e-3,
                        -5.180005e-4,
                        +5.074287e-5,
                        -2.42812e-6};
	//Apply the DX DY and DZ to coefficients
	coeff[0] = (3.0f*coeff[0]) / (DXYZ*DXYZ);
	for(int i=1; i<= HALF_LENGTH; i++){
		coeff[i] = coeff[i] / (DXYZ*DXYZ);
	}



  	// Data Arrays
  	p.prev=NULL, p.next=NULL, p.vel=NULL;

  	// variables for measuring performance
  	double wstart, wstop;
  	float elapsed_time=0.0f, throughput_mpoints=0.0f, mflops=0.0f;
    
  	// allocate dat memory
  	size_t nsize = p.n1*p.n2*p.n3;
  	size_t nbytes = nsize*sizeof(float);

  	printf("allocating prev, next and vel: total %g Mbytes\n",(3.0*(nbytes))/(1024*1024));fflush(NULL);

  	float *prev_base = (float*)_mm_malloc(nsize*sizeof(float) + 32*sizeof(float), 64);
  	float *next_base = (float*)_mm_malloc(nsize*sizeof(float) + 32*sizeof(float), 64);
  	float *vel_base  = (float*)_mm_malloc(nsize*sizeof(float) + 32*sizeof(float), 64);

  	if( prev_base==NULL || next_base==NULL || vel_base==NULL ){
    		printf("couldn't allocate CPU memory prev_base=%p next=_base%p vel_base=%p\n",prev_base, next_base, vel_base);
    		printf("  TEST FAILED!\n"); fflush(NULL);
    		exit(-1);
  	}

  	p.prev = &prev_base[8];
  	p.next = &next_base[8];
  	p.vel  = &vel_base[8];

	//Starting OCCA implementation
	// Setup the platform and device IDs
	occa::properties deviceProps;
	deviceProps["mode"] = "dpcpp";
  	deviceProps["platform_id"] = (int) args["options/platform-id"];
  	deviceProps["device_id"] = (int) args["options/device-id"];

  	occa::device device(deviceProps);
  	// Allocate memory on the device
	int entries = p.n1 * p.n2 * p.n3;
  	occa::memory o_prev = device.malloc<float>(entries);
  	occa::memory o_prev2 = device.malloc<float>(entries);
  	occa::memory o_next = device.malloc<float>(entries);
  	occa::memory o_next2 = device.malloc<float>(entries);
  	occa::memory o_vel = device.malloc<float>(entries);
  	occa::memory o_coeff = device.malloc<float>(HALF_LENGTH+1);

  	// Compile a regular DPCPP kernel at run-time
  	occa::properties kernelProps;
  	kernelProps["okl/enabled"] = true;
  	kernelProps["compiler"] = "dpcpp";
  	kernelProps["compiler_linker_flags"] = "-shared -fPIC";
	
	occa::kernel iso3dfdkernel = device.buildKernel("isokernel.okl",
                                               "iso_kernel",
                                               kernelProps);
	initialize(p.prev, p.next, p.vel, &p, nbytes);

	occa::dim inner(p.n1, p.n2, p.n3);
	occa::dim outer(p.n1_Tblock, p.n2_Tblock, p.n3_Tblock);
	iso3dfdkernel.setRunDims(outer, inner);

  	wstart = walltime();
  	o_prev.copyFrom(p.prev);
	o_next.copyFrom(p.next);
	o_vel.copyFrom(p.vel);
	o_coeff.copyFrom(coeff);

	//time iteration loop
	for(int it=0; it<p.nreps; it+=2){
		iso3dfdkernel(o_next, o_prev, o_vel, o_coeff, p.n1, p.n2, p.n3);
		iso3dfdkernel(o_prev, o_next, o_vel, o_coeff, p.n1, p.n2, p.n3);
  	} // time loop
	o_prev.copyTo(p.prev);
	o_next.copyTo(p.next);
  	wstop =  walltime();
	
  	// report time
  	elapsed_time = wstop - wstart;
  	float normalized_time = elapsed_time/p.nreps;   
  	throughput_mpoints = ((p.n1-2*HALF_LENGTH)*(p.n2-2*HALF_LENGTH)*(p.n3-2*HALF_LENGTH))/(normalized_time*1e6f);
  	mflops = (7.0f*HALF_LENGTH + 5.0f)* throughput_mpoints;

  	printf("-------------------------------\n");
  	printf("time:       %8.2f sec\n", elapsed_time );
  	printf("throughput: %8.2f MPoints/s\n", throughput_mpoints );
  	printf("flops:      %8.2f GFlops\n", mflops/1e3f );

#if defined(VERIFY_RESULTS)
        printf("\n-------------------------------\n");
        printf("comparing one iteration to reference implementation result...\n");

        initialize(p.prev, p.next, p.vel, &p, nbytes);

	o_prev2.copyFrom(p.prev);
        o_next2.copyFrom(p.next);
        o_vel.copyFrom(p.vel);
        o_coeff.copyFrom(coeff);

        //time iteration loop
	for(int it=0; it<2; it++){
                iso3dfdkernel(o_next2, o_prev2, o_vel, o_coeff, p.n1, p.n2, p.n3);
                iso3dfdkernel(o_prev2, o_next2, o_vel, o_coeff, p.n1, p.n2, p.n3);
        } // time loop

	o_prev2.copyTo(p.prev);
        o_next2.copyTo(p.next);


        float *p_ref = (float*)malloc(p.n1*p.n2*p.n3*sizeof(float));
        if( p_ref==NULL ){
                printf("couldn't allocate memory for p_ref\n");
                printf("  TEST FAILED!\n"); fflush(NULL);
                exit(-1);
        }

        initialize(p.prev, p_ref, p.vel, &p, nbytes);
        for(int it=0; it<2; it++){
	        reference_implementation( p_ref, p.prev, coeff, p.vel, p.n1, p.n2, p.n3, HALF_LENGTH );
	        reference_implementation( p.prev, p_ref, coeff, p.vel, p.n1, p.n2, p.n3, HALF_LENGTH );
	}
	o_next2.copyTo(p.next);
        if( within_epsilon( p.next, p_ref, p.n1, p.n2, p.n3, HALF_LENGTH, 0, 1e-45f ) ) {
                printf("  Result within epsilon\n");
                printf("  TEST PASSED!\n");
        } else {
                printf("  Incorrect result\n");
                printf("  TEST FAILED!\n");
        }
        free(p_ref);

#endif /* VERIFY_RESULTS */

  	_mm_free(prev_base);
  	_mm_free(next_base);
  	_mm_free(vel_base);
}


occa::json parseArgs(int argc, const char **argv) {
  // Note:
  //   occa::cli is not supported yet, please don't rely on it
  //   outside of the occa examples
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example of using a regular OpenCL kernel instead of an OCCA kernel"
    )
    .addOption(
      occa::cli::option('p', "platform-id",
                        "OpenCL platform ID (default: 0)")
      .withArg()
      .withDefaultValue(0)
    )
    .addOption(
      occa::cli::option('d', "device-id",
                        "OpenCL device ID (default: 0)")
      .withArg()
      .withDefaultValue(0)
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}

