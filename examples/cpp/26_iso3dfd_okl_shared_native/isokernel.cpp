#include <CL/sycl.hpp>

extern "C" void iso_kernel(sycl::queue* q,
                                                   sycl::nd_range<3>* ndrange,
                                                   float * ptr_next,
                                                   float * ptr_prev,
                                                   float * ptr_vel,
                                                   float * coeff,
                                                   int & n1,
                                                   int & n2,
                                                   int & n3,
                                                   int & halfpad) {
  q->submit([&](sycl::handler &h){
      auto _occa_accessor_xs = sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local>(sizeof(float)*(256 + 16*1), h);
      h.parallel_for(*ndrange, [=] (sycl::nd_item<3> i_dpcpp_iterator){
          float* xs = ((_occa_accessor_xs.get_pointer()));
  {
    int iz = 0 + i_dpcpp_iterator.get_global_id(0);
    //Compute local index for item in x dimension
    {
      int _occa_tiled_iy = 0 + (4 * i_dpcpp_iterator.get_global_id(1));
      for (int iy = _occa_tiled_iy; iy < (_occa_tiled_iy + 4); ++iy) {
        if (iy < n2) {
          if (iy >= 8 && iy < (n2 - 8) && iz >= 8 && iz < (n3 - 8)) {
            float * ptr_nextp = &ptr_next[halfpad];
            float * ptr_prevp = &ptr_prev[halfpad];
            float * ptr_velp = &ptr_vel[halfpad];
            const int dimn1n2 = (n1 + 2 * halfpad) * n2;
            //Handle copy of data in the local array on X
            {
              int ix = 0 + i_dpcpp_iterator.get_global_id(2);
              int offset = iz * dimn1n2 + iy * (n1 + 2 * halfpad) + ix;
              int xloc = ix % 256;
              //check if we need to get few elements before/after in the local array
              if (xloc == 0) {
                for (int i = -8; i <= 0; i++) {
                  xs[xloc + i + 8] = ptr_prevp[offset + i];
                }
              }
              else if (xloc == 256 - 1) {
                for (int i = 0; i <= 8; i++) {
                  xs[xloc + i + 8] = ptr_prevp[offset + i];
                }
              }
              else {
                xs[xloc + 8] = ptr_prevp[offset];
              }
            }
            i_dpcpp_iterator.barrier(sycl::access::fence_space::local_space);
            {
              int ix = 0 + i_dpcpp_iterator.get_global_id(2);
              int offset = iz * dimn1n2 + iy * (n1 + 2 * halfpad) + ix;
              int xloc = ix % 256;
              float value = 0.00000000e+00f;
              value += ptr_prevp[offset] * coeff[0];
              for (int ir = 1; ir <= 8; ir++) {
                value += coeff[ir] * (xs[xloc + ir + 8] + xs[xloc - ir + 8]);
                // horizontal
                value += coeff[ir] * (ptr_prevp[offset + ir * (n1 + 2 * halfpad)] + ptr_prevp[offset - ir * (n1 + 2 * halfpad)]);
                // vertical
                value += coeff[ir] * (ptr_prevp[offset + ir * dimn1n2] + ptr_prevp[offset - ir * dimn1n2]);
                // in front / behind
              }
              ptr_nextp[offset] = 2.00000000e+00f * ptr_prevp[offset] - ptr_nextp[offset] + value * ptr_velp[offset];
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

