#include <CL/sycl.hpp>

extern "C" void addVectors_it(::sycl::queue* q, ::sycl::nd_range<3> *ndrange, int* oa, int* ob, int* oc){
	q->submit([&](::sycl::handler &h){

                h.parallel_for(*ndrange, [=] (::sycl::nd_item<3> i){
                        oc[i.get_global_id(0)] = oa[i.get_global_id(0)] + ob[i.get_global_id(0)];
			
                });
        });
	q->wait();
}

