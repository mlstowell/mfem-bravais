#include "mfem.hpp"
#include "../common/pfem_extras.hpp"
#include<fstream>
#include<iostream>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;

int main(int argc, char ** argv)
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   const char *a0_file   = "A0.mat";
   // const char *a1_file   = "A1.mat";
   const char *dkz0_file = "DKZ0.mat";
   double beta = M_PI;

   OptionsParser args(argc, argv);
   args.AddOption(&a0_file, "-a0", "--a0-matrix","");
   // args.AddOption(&a1_file, "-a1", "--a1-matrix","");
   args.AddOption(&dkz0_file, "-dkz0", "--dkz0-matrix","");
   args.AddOption(&beta, "-b", "--beta","");

   HypreParMatrix A0, DKZ0;
   // HypreParMatrix A1;

   A0.Read(comm,a0_file);
   // A1.Read(comm,a1_file);
   DKZ0.Read(comm,dkz0_file);

   Array<int> block_trueOffsets0;
   block_trueOffsets0.SetSize(3);
   block_trueOffsets0[0] = 0;
   block_trueOffsets0[1] = A0.Height();
   block_trueOffsets0[2] = A0.Height();
   block_trueOffsets0.PartialSum();

   BlockOperator S0(block_trueOffsets0);
   S0.SetDiagonalBlock(0,&A0,1.0);
   S0.SetDiagonalBlock(1,&A0,1.0);
   S0.SetBlock(0,1,&DKZ0,-beta);
   S0.SetBlock(1,0,&DKZ0, beta);
   S0.owns_blocks = 0;

   MINRESSolver minres(comm);
   minres.SetOperator(S0);
   minres.SetRelTol(1e-13);
   minres.SetMaxIter(3000);
   minres.SetPrintLevel(1);

   BlockVector x0(block_trueOffsets0);
   BlockVector y0(block_trueOffsets0);
   BlockVector b0(block_trueOffsets0);

   y0.Randomize(123);
   S0.Mult(y0,b0);

   x0 = 0.0;
   tic();
   minres.Mult(b0,x0);
   cout << "Time for MINRES solve of A0:  " << toc() << endl;
   y0 -= x0;
   cout << "Absolute error, ||x-y||_2 = " << y0.Norml2() << endl;

   MPI_Finalize();
}
