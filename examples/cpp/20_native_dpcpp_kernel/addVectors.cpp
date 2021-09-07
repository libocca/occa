#include <CL/sycl.hpp>

extern "C" void addVectors(::sycl::queue *q,
                           ::sycl::nd_range<3> *ndrange, 
                           int &entries, 
                           int *a, 
                           int *b, 
                           int *c)
{
  q->submit([&](::sycl::handler &h) {
    h.parallel_for(*ndrange, [=](::sycl::nd_item<3> i) {
      int ii = i.get_global_id(0) + i.get_global_id(1) * i.get_global_range(0) + i.get_global_id(2) * i.get_global_range(0) * i.get_global_range(1);
      if (ii < entries)
        c[ii] = a[ii] + b[ii];
    });
  });
}
