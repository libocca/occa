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
      auto _occa_accessor_ptrs = sycl::accessor<char, 1, sycl::access::mode::read_write, sycl::access::target::local>(sizeof(float)*256 * 256 * 256*1, h);
      h.parallel_for(*ndrange, [=] (sycl::nd_item<3> i_dpcpp_iterator){
          float ptrs = *((_occa_accessor_ptrs.get_pointer()));
  {
    int iz = 0 + i_dpcpp_iterator.get_global_id(2);
    float ptrs[256 * 256 * 256];
    {
      int _occa_tiled_iy = 0 + (4 * i_dpcpp_iterator.get_global_id(1));
      for (int iy = _occa_tiled_iy; iy < (_occa_tiled_iy + 4); ++iy) {
        if (iy < n2) {
          {
            int ix = 0 + i_dpcpp_iterator.get_global_id(0);
            const int dimn1n2 = n1 * n2;
            if (ix >= 8 && ix < (n1 - 8) && iy >= 8 && iy < (n2 - 8) && iz >= 8 && iz < (n3 - 8)) {
              int offset = iz * dimn1n2 + iy * n1 + ix;
              float value = 0.00000000e+00f;
              value += ptr_prev[offset] * coeff[0];
              for (int ir = 1; ir <= 8; ir++) {
                value += coeff[ir] * (ptr_prev[offset + ir] + ptr_prev[offset - ir]);
                // horizontal
                value += coeff[ir] * (ptr_prev[offset + ir * n1] + ptr_prev[offset - ir * n1]);
                // vertical
                value += coeff[ir] * (ptr_prev[offset + ir * dimn1n2] + ptr_prev[offset - ir * dimn1n2]);
                // in front / behind
              }
              ptr_next[offset] = 2.00000000e+00f * ptr_prev[offset] - ptr_next[offset] + value * ptr_vel[offset];
            }
          }
        }
      }
    }
  }
  });
});
q->wait();
}


