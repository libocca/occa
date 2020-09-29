#include <CL/sycl.hpp>
#include <iostream>

//extern "C" void addVectors_it(::sycl::nd_range<3> *ndrange, int* oa, int* ob, int* oc, ::sycl::queue* q){
extern "C" void addVectors_it(::sycl::queue* q, ::sycl::nd_range<3> *ndrange, int* oa, int* ob, int* oc){
	std::cout<<"pointers = "<<(void*) q<<" "<<(void*) ndrange<<" "<<oa<<" "<<ob<<" "<<oc<<std::endl;
	int* bb = (int*) malloc_shared(15*sizeof(int), q->get_device(), q->get_context());
	int* cc = (int*) malloc_shared(15*sizeof(int), q->get_device(), q->get_context());
	for(int i=0; i<15;i++)
		bb[i] = i;
	q->wait();
	q->submit([&](::sycl::handler &h){

                h.parallel_for(sycl::range<1>{15}, [=] (::sycl::item<1> i){
                        //oc[i.get_global_id(0)] = oa[i.get_global_id(0)] + ob[i.get_global_id(0)];
                        cc[i] = bb[i];
			
                });
        });
	q->wait();
	for(int i=0; i<15;i++)
		std::cout<<cc[i]<<" ";
	std::cout<<std::endl;


}

