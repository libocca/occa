#include <iostream>

#include <occa.hpp>
#include <occa/mpi.hpp>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (occa::mpi::size() != 2) {
    std::cerr << "Example expects to run with 2 processes\n";
    return 1;
  }

  occa::json config = occa::json::parse(
    "["
    "  { mode: 'OpenMP' },"
    "  { mode: 'CUDA', device_id: 0 },"
    "]"
  );

  const int mpiID = occa::mpi::id();
  occa::setDevice(config[mpiID]);

  int entries = 16;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = 10 + i;
    b[i]  = mpiID - a[i];
    ab[i] = 0;
  }

  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  o_a  = occa::malloc(entries, occa::dtype::float_);
  o_b  = occa::malloc(entries, occa::dtype::float_);
  o_ab = occa::malloc(entries, occa::dtype::float_);

  addVectors = occa::buildKernel("addVectors.okl",
                                 "addVectors");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  const int otherID     = (mpiID + 1) % 2;
  const int halfEntries = entries / 2;
  const int offsetBytes = halfEntries * sizeof(float);

  occa::mpi::tags tags;
  tags += occa::mpi::send<float>(otherID,
                                 o_ab + (mpiID * offsetBytes),
                                 halfEntries);

  tags += occa::mpi::get<float>(otherID,
                                o_ab + (otherID * offsetBytes),
                                halfEntries);
  tags.wait();

  o_ab.copyTo(ab);

  for (int id = 0; id < 2; ++id) {
    if (mpiID == id) {
      for (int i = 0; i < entries; ++i) {
        std::cout << '[' << mpiID << "] " << i << ": " << ab[i] << '\n';
      }
    }
    occa::mpi::barrier();
  }

  delete [] a;
  delete [] b;
  delete [] ab;
}
