//                       MFEM Example 11 - Parallel Version
//
// Compile with: make ex11p
//
// Sample runs:  mpirun -np 4 ex11p -m ../data/square-disc.mesh
//               mpirun -np 4 ex11p -m ../data/star.mesh
//               mpirun -np 4 ex11p -m ../data/escher.mesh
//               mpirun -np 4 ex11p -m ../data/fichera.mesh
//               mpirun -np 4 ex11p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex11p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex11p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex11p -m ../data/disc-nurbs.mesh -o -1 -n 20
//               mpirun -np 4 ex11p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex11p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex11p -m ../data/star-surf.mesh
//               mpirun -np 4 ex11p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex11p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex11p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex11p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex11p -m ../data/mobius-strip.mesh -n 8
//               mpirun -np 4 ex11p -m ../data/klein-bottle.mesh -n 10
//
// Description:  This example code demonstrates the use of MFEM to solve the
//               eigenvalue problem -Delta u = lambda u with homogeneous
//               Dirichlet boundary conditions.
//
//               We compute a number of the lowest eigenmodes by discretizing
//               the Laplacian and Mass operators using a FE space of the
//               specified order, or an isoparametric/isogeometric space if
//               order < 1 (quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of the LOBPCG eigenvalue solver
//               together with the BoomerAMG preconditioner in HYPRE, as well as
//               optionally the SuperLU parallel direct solver. Reusing a single
//               GLVis visualization window for multiple eigenfunctions is also
//               illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include "linalg/abs_val_op.hpp"
#include "linalg/eigensolvers.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void PokeIntoAMG(HypreBoomerAMG &amg, HypreParMatrix & A);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   int av_o = 2;
   int nev = 5;
   int seed = 75;
   double eps = 0.0;
   bool slu_solver  = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&av_o, "-avo", "--abs-val-order",
                  "ORder of absolute value approximation");
   args.AddOption(&nev, "-n", "--num-eigs",
                  "Number of desired eigenmodes.");
   args.AddOption(&seed, "-s", "--seed",
                  "Random seed used to initialize LOBPCG.");
   args.AddOption(&eps, "-e", "--epsilon",
                  "PLHR shift = lambda + epsilon.");
#ifdef MFEM_USE_SUPERLU
   args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement (2 by default, or
   //    specified on the command line with -rs).
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution (1 time by
   //    default, or specified on the command line with -rp). Once the parallel
   //    mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   double h_min, h_max, kappa_min, kappa_max;
   pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }

   // 7. Set up the parallel bilinear forms a(.,.) and m(.,.) on the finite
   //    element space. The first corresponds to the Laplacian operator -Delta,
   //    while the second is a simple mass matrix needed on the right hand side
   //    of the generalized eigenvalue problem below. The boundary conditions
   //    are implemented by elimination with special values on the diagonal to
   //    shift the Dirichlet eigenvalues out of the computational range. After
   //    serial and parallel assembly we extract the corresponding parallel
   //    matrices A and M.
   ConstantCoefficient one(1.0);
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
   }

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   if (pmesh->bdr_attributes.Size() == 0)
   {
      // Add a mass term if the mesh has no boundary, e.g. periodic mesh or
      // closed surface.
      a->AddDomainIntegrator(new MassIntegrator(one));
   }
   a->Assemble();
   a->EliminateEssentialBCDiag(ess_bdr, 1.0);
   a->Finalize();

   ParBilinearForm *m = new ParBilinearForm(fespace);
   m->AddDomainIntegrator(new MassIntegrator(one));
   m->Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   m->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
   m->Finalize();

   HypreParMatrix *A = a->ParallelAssemble();
   HypreParMatrix *M = m->ParallelAssemble();

#ifdef MFEM_USE_SUPERLU
   Operator * Arow = NULL;
   if (slu_solver)
   {
      Arow = new SuperLURowLocMatrix(*A);
   }
#endif

   delete a;
   delete m;

   // 8. Define and configure the LOBPCG eigensolver and the BoomerAMG
   //    preconditioner for A to be used within the solver. Set the matrices
   //    which define the generalized eigenproblem A x = lambda M x.
   Solver * precond = NULL;
   if (!slu_solver)
   {
      HypreBoomerAMG * amg = new HypreBoomerAMG(*A);
      amg->SetPrintLevel(0);
      precond = amg;
   }
#ifdef MFEM_USE_SUPERLU
   else
   {
      SuperLUSolver * superlu = new SuperLUSolver(MPI_COMM_WORLD);
      superlu->SetPrintStatistics(false);
      superlu->SetSymmetricPattern(true);
      superlu->SetColumnPermutation(superlu::PARMETIS);
      superlu->SetOperator(*Arow);
      precond = superlu;
   }
#endif

   HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);
   lobpcg->SetNumModes(nev);
   lobpcg->SetRandomSeed(seed);
   lobpcg->SetPreconditioner(*precond);
   lobpcg->SetMaxIter(200);
   lobpcg->SetTol(1e-8);
   lobpcg->SetPrecondUsageMode(1);
   lobpcg->SetPrintLevel(1);
   lobpcg->SetMassMatrix(*M);
   lobpcg->SetOperator(*A);

   BPLHRSolver * bplhr = new BPLHRSolver(MPI_COMM_WORLD);
   bplhr->SetNumModes(nev);
   bplhr->SetMaxIter(15);
   bplhr->SetRelTol(1e-8);
   bplhr->SetOperator(*A, *M);

   // 9. Compute the eigenmodes and extract the array of eigenvalues. Define a
   //    parallel grid function to represent each of the eigenmodes returned by
   //    the solver.
   Array<double> eigenvalues;
   lobpcg->Solve();
   lobpcg->GetEigenvalues(eigenvalues);
   ParGridFunction x(fespace);
   Vector X;

   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);
      mode_sock.precision(8);

      double shift = eigenvalues[nev-1] + eps;

      HypreParMatrix * C = Add(1.0, *A, -shift, *M);

      AbsoluteValueOperator AbsC(*C, av_o, -shift,
                                 pow(M_PI / h_min, 2.0) - shift);

      // HypreBoomerAMG * amg = new HypreBoomerAMG(*C);

      // PokeIntoAMG(*amg, *C);
      /*
      MINRESSolver minres(MPI_COMM_WORLD);
      minres.SetMaxIter(10);
      minres.SetRelTol(1.0e-3);
      minres.SetOperator(AbsC);

      plhr->SetPreconditioner(minres);
      */
      AbsoluteValuePrecond avpc(*A, *M, shift);
      avpc.SetEigenvalueEstimates(0.0, pow(M_PI / h_min, 2.0));
      avpc.SetOrderLimits(1, 10);
      avpc.SetNumSmootherSteps(1, 5);
      // avpc.SetOperator(*C);
      avpc.Setup();

      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetMaxIter(100);
      gmres.SetRelTol(1.0e-6);
      gmres.SetOperator(AbsC);

      // bplhr->SetPreconditioner(gmres);
      bplhr->SetPreconditioner(avpc);
      bplhr->SetShift(shift);
      //bplhr->Reinitialize();

      Array<double> eig_bplhr;
      Vectors XV(C->Height(), nev);

      bplhr->GetEigenPairs(eig_bplhr, XV);

      delete C;

      for (int i=0; i<nev; i++)
      {
         if ( myid == 0 )
         {
            cout << i << ":\t" << eigenvalues[i] << "\t" << eig_bplhr[i] << endl;
         }

         x.Distribute(XV[i]);

         mode_sock << "parallel " << num_procs << " " << myid << "\n"
                   << "solution\n" << *pmesh << x << flush
                   << "window_title 'Eigenmode " << i+1 << '/' << nev
                   << ", Lambda = " << eig_bplhr[i] << "'" << endl;

         char c;
         if (myid == 0)
         {
            cout << "press (q)uit or (c)ontinue --> " << flush;
            cin >> c;
         }
         MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

         if (c != 'c')
         {
            break;
         }
      }
      mode_sock.close();
   }

   // 10. Save the refined mesh and the modes in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g mode".
   {
      ostringstream mesh_name, mode_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      for (int i=0; i<nev; i++)
      {
         // convert eigenvector from HypreParVector to ParGridFunction
         x = lobpcg->GetEigenvector(i);

         mode_name << "mode_" << setfill('0') << setw(2) << i << "."
                   << setfill('0') << setw(6) << myid;

         ofstream mode_ofs(mode_name.str().c_str());
         mode_ofs.precision(8);
         x.Save(mode_ofs);
         mode_name.str("");
      }
   }

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mode_sock(vishost, visport);
      mode_sock.precision(8);

      for (int i=0; i<nev; i++)
      {
         if ( myid == 0 )
         {
            cout << "Eigenmode " << i+1 << '/' << nev
                 << ", Lambda = " << eigenvalues[i] << endl;
         }

         // convert eigenvector from HypreParVector to ParGridFunction
         x = lobpcg->GetEigenvector(i);

         mode_sock << "parallel " << num_procs << " " << myid << "\n"
                   << "solution\n" << *pmesh << x << flush
                   << "window_title 'Eigenmode " << i+1 << '/' << nev
                   << ", Lambda = " << eigenvalues[i] << "'" << endl;

         char c;
         if (myid == 0)
         {
            cout << "press (q)uit or (c)ontinue --> " << flush;
            cin >> c;
         }
         MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

         if (c != 'c')
         {
            break;
         }
      }
      mode_sock.close();
   }

   // 12. Free the used memory.
   delete lobpcg;
   delete precond;
   delete M;
   delete A;
#ifdef MFEM_USE_SUPERLU
   delete Arow;
#endif

   delete fespace;
   if (order > 0)
   {
      delete fec;
   }
   delete pmesh;

   MPI_Finalize();

   return 0;
}

void PokeIntoAMG(HypreBoomerAMG &amg, HypreParMatrix & A)
{
   HYPRE_Solver sol_pre = (HYPRE_Solver)amg;
   hypre_ParAMGData * data_pre = (hypre_ParAMGData*)sol_pre;

   cout << "Num Levels: " << hypre_ParAMGDataNumLevels(data_pre) << endl;

   HypreParVector X(A);  HypreParVector B(A);
   X = 0.0;
   B = 1.0;

   amg.SetupFcn()(amg, A, B, X);

   HYPRE_Solver sol_post = (HYPRE_Solver)amg;
   hypre_ParAMGData * data_post = (hypre_ParAMGData*)sol_post;

   cout << "Num Levels: " << hypre_ParAMGDataNumLevels(data_post) << endl;

   HYPRE_Solver sol = (HYPRE_Solver)amg;
   hypre_ParAMGData * data = (hypre_ParAMGData*)sol;

   hypre_GaussElimSetup(data, data->num_levels - 1, 0);

   //  hypre_ParAMGData * data = data_post;
   cout << endl;
   cout << "/* setup params */" << endl;
   cout << "max_levels:        " << data->max_levels << endl;
   cout << "strong_threshold:  " << data->strong_threshold << endl;
   cout << "max_row_sum:       " << data->max_row_sum << endl;
   cout << "agg_num_levels:    " << data->agg_num_levels << endl;
   cout << "max_coarse_size:   " << data->max_coarse_size << endl;
   cout << "min_coarse_size:   " << data->min_coarse_size << endl;
   cout << endl;
   cout << "/* solve params */" << endl;
   cout << "max_iter:          " << data->max_iter << endl;
   cout << "min_iter:          " << data->min_iter << endl;
   cout << "cycle_type:        " << data->cycle_type << endl;
   cout << endl;
   cout << "/* data generated in the setup phase */" << endl;
   cout << "num_levels:        " << data->num_levels << endl;
   cout << endl;
   cout << "/* data for more complex smoothers */" << endl;
   cout << "smooth_num_levels: " << data->smooth_num_levels << endl;
   cout << "smooth_type:       " << data->smooth_type << endl;

   for (int l=0; l<data->num_levels; l++)
   {
      hypre_ParCSRMatrix * Al = data->A_array[l];
      ostringstream ossA; ossA << "A_" << l << ".mat";
      hypre_ParCSRMatrixPrint(Al, ossA.str().c_str());

      if ( l < data->num_levels - 1 )
      {
         hypre_ParCSRMatrix * Pl = data->P_array[l];
         ostringstream ossP; ossP << "P_" << l << ".mat";
         hypre_ParCSRMatrixPrint(Pl, ossP.str().c_str());

         hypre_ParCSRMatrix * Rl = data->R_array[l];
         ostringstream ossR; ossR << "R_" << l << ".mat";
         hypre_ParCSRMatrixPrint(Rl, ossR.str().c_str());
      }

      if ( data->F_array != NULL )
      {
         hypre_ParVector    * Fl = data->F_array[l];
         ostringstream ossF; ossF << "F_" << l << ".vec";
         if ( Fl != NULL )
         {
            hypre_ParVectorPrint(Fl, ossF.str().c_str());
         }
         else
         {
            cout << "F_array[" << l << "] is NULL" << endl;
         }
      }
      else
      {
         cout << "F_array is NULL" << endl;
      }
      if ( data->U_array != NULL )
      {
         hypre_ParVector    * Ul = data->U_array[l];
         ostringstream ossU; ossU << "U_" << l << ".vec";
         if ( Ul != NULL )
         {
            hypre_ParVectorPrint(Ul, ossU.str().c_str());
         }
         else
         {
            cout << "U_array[" << l << "] is NULL" << endl;
         }
      }
      else
      {
         cout << "U_array is NULL" << endl;
      }
   }
   if (  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(
                                   data->A_array[data->num_levels - 1])) )
   {
      int myid = -1;
      MPI_Comm_rank(A.GetComm(), &myid);

      ostringstream oss; oss << "a0_" << myid << ".mat";
      ofstream ofs(oss.str().c_str());

      int n_global = hypre_ParCSRMatrixGlobalNumRows(data->A_array[data->num_levels -
                                                                   1]);

      for (int i=0; i<n_global; i++)
      {
         for (int j=0; j<n_global; j++)
         {
            ofs << data->A_mat[i*n_global+j] << " ";
         }
         ofs << endl;
      }
   }
}
