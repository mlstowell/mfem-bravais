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
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double
PowerMethod(const Operator &A, HypreParVector &x, HypreParVector &y);

double
PowerMethod(const Operator &A, const Operator &MInv,
            HypreParVector &x, HypreParVector &y, HypreParVector &z);
/*
double
PowerMethod(const HypreParMatrix &A, const HypreParMatrix &M,
       HypreParVector &x, HypreParVector &y);
*/
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
   int av_order = 10;
   int nev = 5;
   int seed = 75;
   bool gen_eig = true;
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
   args.AddOption(&av_order, "-avo", "--abs-val-order",
                  "Order of the absolute value operator.");
   args.AddOption(&gen_eig, "-g", "--gen-eig","-no-g","--no-gen-eig",
                  "Generalized Eigenvalue problem.");
   args.AddOption(&nev, "-n", "--num-eigs",
                  "Number of desired eigenmodes.");
   args.AddOption(&seed, "-s", "--seed",
                  "Random seed used to initialize LOBPCG.");
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

   if ( myid == 0 )
   {
      cout << "Mesh Characteristics:" << endl
           << "  h_min:     " << h_min << endl
           << "  h_max:     " << h_max << endl
           << "  kappa_min: " << kappa_min << endl
           << "  kappa_max: " << kappa_max << endl;
   }

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

   ParBilinearForm *m0 = new ParBilinearForm(fespace);
   m0->AddDomainIntegrator(new MassIntegrator(one));
   m0->Assemble();
   // shift the eigenvalue corresponding to eliminated dofs to a large value
   // m0->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
   m0->Finalize();

   ParBilinearForm *m = new ParBilinearForm(fespace);
   m->AddDomainIntegrator(new MassIntegrator(one));
   m->Assemble();
   m->Finalize();

   HypreParMatrix *A  = a->ParallelAssemble();
   HypreParMatrix *M0 = m0->ParallelAssemble();
   HypreParMatrix *M  = m->ParallelAssemble();

#ifdef MFEM_USE_SUPERLU
   Operator * Arow = NULL;
   if (slu_solver)
   {
      Arow = new SuperLURowLocMatrix(*A);
   }
#endif

   delete a;
   delete m;
   delete m0;

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
   if ( gen_eig )
   {
      lobpcg->SetMassMatrix(*M);
   }
   lobpcg->SetOperator(*A);

   // 9. Compute the eigenmodes and extract the array of eigenvalues. Define a
   //    parallel grid function to represent each of the eigenmodes returned by
   //    the solver.
   Array<double> eigenvalues;
   lobpcg->Solve();
   lobpcg->GetEigenvalues(eigenvalues);
   ParGridFunction x(fespace);

   HyprePCG pcg(*M);
   pcg.SetTol(1.0e-10);
   pcg.SetMaxIter(500);
   pcg.SetPrintLevel(0);

   // HypreParVector X(*A);
   HypreParVector X_av(*A);
   HypreParVector X_b(*A);
   HypreParVector Y_av(*A);
   ParGridFunction x_av(fespace);
   ParGridFunction x_b(fespace);

   /*
   double  pm = PowerMethod(*A, X_av, Y_av);
   double gpm = PowerMethod(*A, *M, X_av, Y_av);
   cout << "Lambda max from PM:  " << pm << endl;
   cout << "Lambda max from GPM: " << gpm << endl;
   */
   /*
   double Amax = PowerMethod(*A, X_av, Y_av);
   double Mmax = PowerMethod(*M, X_av, Y_av);
   double Mmin = 1.0 / PowerMethod(pcg, X_av, Y_av);

   cout << "Amax: " << Amax << endl;
   cout << "Mmax: " << Mmax << endl;
   cout << "Mmin: " << Mmin << endl;
   */
   /*
   double lambda_max = pow(M_PI, 2.0);
   // if ( gen_eig ) lambda_max /= pow(h_min, 2.0);
    if ( gen_eig ) lambda_max = 6250.;
   */
   double lambda_max = 0.0;
   if ( gen_eig )
   {
      //lambda_max = 2.0 * PowerMethod(*A, *M, X_av, Y_av);
      // lambda_max = Amax / Mmin;
      lambda_max = PowerMethod(*A, pcg, X_av, Y_av, X_b);
   }
   else
   {
      lambda_max = PowerMethod(*A, X_av, Y_av);
   }

   // double shift = 0.5 * (eigenvalues[0] + eigenvalues[nev-1]);
   double shift = 0.9 * eigenvalues[nev-1];

   if ( myid == 0 )
   {
      cout << "Shift: " << shift << endl;
   }

   HypreParMatrix *B = NULL;
   if ( gen_eig )
   {
      B = Add(1.0, *A, -shift, *M);
   }
   else
   {
      SparseMatrix Isp(A->GetNumRows(), A->GetNumRows(), 1);
      Array<int> col(1);
      Vector val(1); val = 1.0;
      for (int i=0; i<A->GetNumRows(); i++)
      {
         col[0] = i;
         Isp.SetRow(i,col,val);
      }
      HypreParMatrix I0(A->GetComm(), A->M(), A->GetRowStarts(), &Isp);

      B = Add(1.0, *A, -shift, I0);
   }

   AbsoluteValueOperator * avop = NULL;
   if ( gen_eig )
   {
      avop = new AbsoluteValueOperator(*B, *M, av_order,
                                       0.0 - shift, lambda_max - shift);
   }
   else
   {
      avop = new AbsoluteValueOperator(*B, av_order,
                                       0.0 - shift, lambda_max - shift);
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
      socketstream bx_sock(vishost, visport);
      bx_sock.precision(8);
      socketstream diff_sock(vishost, visport);
      diff_sock.precision(8);

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

         B->Mult(lobpcg->GetEigenvector(i), X_b);
         avop->Mult(lobpcg->GetEigenvector(i), X_av);

         cout << "Norm X_av: " << X_av.Norml2() << endl;

         if ( gen_eig )
         {
            Y_av = 0.0; pcg.Mult(X_b, Y_av);
            x_b  = Y_av;
            // Y_av = 0.0; pcg.Mult(X_av, Y_av);
            // x_av = Y_av;
            x_av = X_av;
         }
         else
         {
            x_b  = X_b;
            x_av = X_av;
         }

         x_b /= eigenvalues[i] - shift;
         x_av /= fabs(eigenvalues[i] - shift);
         //x_av -= x;
         // x_av = X_av;
         // x_av.Add(fabs(eigenvalues[i]-shift), x);

         bx_sock << "parallel " << num_procs << " " << myid << "\n"
                 << "solution\n" << *pmesh << x_b << flush
                 << "window_title 'B x " << i+1 << '/' << nev
                 << ", Lambda-Shift = " << eigenvalues[i]-shift << "'"
                 << endl;

         diff_sock << "parallel " << num_procs << " " << myid << "\n"
                   << "solution\n" << *pmesh << x_av << flush
                   << "window_title 'Difference " << i+1 << '/' << nev
                   << ", |Lambda-Shift| = " << fabs(eigenvalues[i]-shift) << "'"
                   << endl;

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
   delete M0;
   delete M;
   delete A;
   delete avop;
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

double
PowerMethod(const Operator &A, HypreParVector &x, HypreParVector &y)
{
   x.Randomize(123);
   double mu0 = InnerProduct(x, x);
   double mu1 = 0.0;
   double rel_diff = 1.0;
   x /= sqrt(mu0);

   A.Mult(x, y);
   mu0 = InnerProduct(x, y);
   x.Set(1.0/sqrt(mu0), y);

   double tol = 1.0e-3;
   int maxit = 200;
   int it = 0;

   while ( it < maxit && rel_diff > tol )
   {
      A.Mult(x, y);
      mu1 = InnerProduct(x,y);

      rel_diff = 2.0 * fabs((mu1 - mu0) / (mu1 + mu0));

      x.Set(1.0/sqrt(mu1), y);

      mu0 = mu1;
      it++;
      cout << it << ": " << mu0  << "\t" << rel_diff << endl;
   }
   return mu1;
}

double
PowerMethod(const Operator &A, const Operator &MInv,
            HypreParVector &x, HypreParVector &y, HypreParVector &z)
{
   x.Randomize(123);
   double mu0 = InnerProduct(x, x);
   double mu1 = 0.0;
   double rel_diff = 1.0;
   x /= sqrt(mu0);

   A.Mult(x, y);
   z = 0.0; MInv.Mult(y, z);
   mu0 = InnerProduct(x, z);
   x.Set(1.0/sqrt(mu0), z);

   double tol = 1.0e-3;
   int maxit = 200;
   int it = 0;

   while ( it < maxit && rel_diff > tol )
   {
      A.Mult(x, y);
      MInv.Mult(y, z);
      mu1 = InnerProduct(x,z);

      rel_diff = 2.0 * fabs((mu1 - mu0) / (mu1 + mu0));

      x.Set(1.0/sqrt(mu1), z);

      mu0 = mu1;
      it++;
      cout << it << ": " << mu0  << "\t" << rel_diff << endl;
   }
   return mu1;
}
/*
double
PowerMethod(const HypreParMatrix &A, const HypreParMatrix &M,
       HypreParVector &x, HypreParVector &y)
{
  x.Randomize(123);
  M.Mult(x, y);
  double nrm = InnerProduct(x, y);
  A.Mult(x, y);
  double mu0 = InnerProduct(x, y) / nrm;
  double mu1 = 0.0;
  double rel_diff = 1.0;

  x.Set(1.0 / sqrt(nrm), y);

  double tol = 1.0e-3;
  int maxit = 200;
  int it = 0;

  while ( it < maxit && rel_diff > tol )
  {
    M.Mult(x, y);
    nrm = InnerProduct(x, y);
    // x /= sqrt(nrm);

    A.Mult(x, y);
    mu1 = InnerProduct(x, y) / nrm;

    rel_diff = 2.0 * fabs((mu1 - mu0) / (mu1 + mu0));

    x.Set(1.0 / sqrt(nrm), y);

    mu0 = mu1;
    it++;
    cout << it << ": " << mu0  << "\t" << rel_diff << endl;
  }
  return mu1;
}
*/
