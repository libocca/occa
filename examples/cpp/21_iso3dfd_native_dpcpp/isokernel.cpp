#include <CL/sycl.hpp>
extern "C" void iso_kernel(sycl::queue* q,
                                                   sycl::nd_range<3>* ndrange,
                                                   float * ptr_next,
                                                   float * ptr_prev,
                                                   float * ptr_vel,
                                                   float * coeff,
                                                   int & n1,
                                                   int & n2,
                                                   int & n3) {
  q->submit([&](sycl::handler &h){
                         h.parallel_for(*ndrange, [=] (sycl::nd_item<3> i_dpcpp_iterator){
  const int iz = i_dpcpp_iterator.get_global_id(2);
  const int iy = i_dpcpp_iterator.get_global_id(1);
  const int ix = i_dpcpp_iterator.get_global_id(0);
  const int dimn1n2 = n1 * n2;
    if(ix>8 && ix<n1-8)
    if(iy>8 && iy<n2-8)
    if(iz>8 && iz<n3-8){
      int offset = iz * dimn1n2 + iy * n1 + ix;
      float value = 0.0f;
      value += ptr_prev[offset] * coeff[0];
      for (int ir = 1; ir <= 8; ir++) {
        value += coeff[ir] * (ptr_prev[offset + ir] + ptr_prev[offset - ir]);
        // horizontal
        value += coeff[ir] * (ptr_prev[offset + ir * n1] + ptr_prev[offset - ir * n1]);
        // vertical
        value += coeff[ir] * (ptr_prev[offset + ir * dimn1n2] + ptr_prev[offset - ir * dimn1n2]);
        // in front / behind
      }
      ptr_next[offset] = 2.0f * ptr_prev[offset] - ptr_next[offset] + value * ptr_vel[offset];
    }
  });
});
q->wait();
}

