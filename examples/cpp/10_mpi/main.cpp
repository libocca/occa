#include <iostream>

#include <occa.hpp>
#include <occa/experimental/mpi.hpp>

#if OCCA_MPI_ENABLED

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (occa::mpi::size() != 2) {
    std::cerr << "Example expects to run with 2 processes\n";
    return 1;
  }

  occa::json config = occa::json::parse(
    "["
    "  { mode: 'OpenMP' },"
    "  { mode: 'Serial' },"
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

  o_a  = occa::malloc<float>(entries);
  o_b  = occa::malloc<float>(entries);
  o_ab = occa::malloc<float>(entries);

  addVectors = occa::buildKernel("addVectors.okl",
                                 "addVectors");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  const int otherID     = (mpiID + 1) % 2;
  const int halfEntries = entries / 2;

  occa::mpi::tags tags;
  tags += occa::mpi::send<float>(otherID,
                                 o_ab + (mpiID * halfEntries),
                                 halfEntries);

  tags += occa::mpi::get<float>(otherID,
                                o_ab + (otherID * halfEntries),
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

  MPI_Finalize();
}

#else

int main(int argc, char **argv) {
  std::cout << "Warning: OCCA_MPI_ENABLED is set to false" << std::endl;
  return 0;
}

#endif
