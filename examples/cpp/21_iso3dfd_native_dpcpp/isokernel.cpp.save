#include <CL/sycl.hpp>

#define HALF_LENGTH 8

extern "C" void iso_kernel(sycl::queue* q, sycl::nd_range<3>* ndrange, float* ptr_next, float* ptr_prev, float* ptr_vel, float* coeff,
                        int& n1, int& n2, int& n3){
//	std::cout<<"pnext = "<<n1<<std::endl;
	int dimn1n2 = n1 * n2;
	q->submit([&](::sycl::handler &h){
                h.parallel_for(*ndrange, [=] (::sycl::nd_item<3> i){
                        int ix = i.get_global_id(0);
                        int iy = i.get_global_id(1);
                        int iz = i.get_global_id(2);
			if(ix >= HALF_LENGTH && ix < (n1-HALF_LENGTH) && iy >= HALF_LENGTH && iy < (n2-HALF_LENGTH) && iz >= HALF_LENGTH && iz < (n3-HALF_LENGTH)){
				int offset = iz*dimn1n2 + iy * n1 + ix;
                                float value = 0.0;
                                value += ptr_prev[offset]*coeff[0];
                                for(int ir=1; ir<=HALF_LENGTH; ir++) {
                                        value += coeff[ir] * (ptr_prev[offset + ir] + ptr_prev[offset - ir]);// horizontal
                                        value += coeff[ir] * (ptr_prev[offset + ir * n1] + ptr_prev[offset - ir * n1]);// vertical
                                        value += coeff[ir] * (ptr_prev[offset + ir*dimn1n2] + ptr_prev[offset - ir*dimn1n2]); // in front / behind
                                }
                                ptr_next[offset] = 2.0f* ptr_prev[offset] - ptr_next[offset] + value*ptr_vel[offset];

			}	
		});
	});
	q->wait();
}
