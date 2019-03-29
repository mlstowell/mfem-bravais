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
//               mpirun -np 4 ex11p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex11p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex11p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex11p -m ../data/star-surf.mesh
//               mpirun -np 4 ex11p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex11p -m ../data/inline-segment.mesh
//
// Description:  This example code demonstrates the use of MFEM to solve a
//               generalized eigenvalue problem
//                 -Delta u = lambda u
//               with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize the Laplacian operator using a
//               FE space of the specified order, or if order < 1 using an
//               isoparametric/isogeometric space (i.e. quadratic for
//               quadratic curvilinear mesh, NURBS for NURBS mesh, etc.)
//
//               The example is a modification of example 1 which highlights
//               the use of the LOBPCG eigenvalue solver in HYPRE.
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class SturmLiouville
{
public:
   enum EQ_TYPE {BESSEL_EQ=1,CHEBYSHEV_EQ=2,CHEBYSHEV2_EQ=3,
                 LAPLACE_EQ=4,LEGENDRE_EQ=5,EX6_EQ
                };

   SturmLiouville(MPI_Comm comm, EQ_TYPE e);
   ~SturmLiouville();

   FunctionCoefficient & p() { return *pCoef; }
   FunctionCoefficient & q() { return *qCoef; }
   FunctionCoefficient & w() { return *wCoef; }

private:

   FunctionCoefficient * pCoef;
   FunctionCoefficient * qCoef;
   FunctionCoefficient * wCoef;

   static double BesselP(Vector & x);
   static double BesselQ(Vector & x);
   static double BesselW(Vector & x);

   static double LaplaceP(Vector & x);
   static double LaplaceQ(Vector & x);
   static double LaplaceW(Vector & x);

   static double LegendreP(Vector & x);
   static double LegendreQ(Vector & x);
   static double LegendreW(Vector & x);

   static double ChebyshevP(Vector & x);
   static double ChebyshevQ(Vector & x);
   static double ChebyshevW(Vector & x);

   static double Chebyshev2P(Vector & x);
   static double Chebyshev2Q(Vector & x);
   static double Chebyshev2W(Vector & x);

   static double Ex6P(Vector & x);
   static double Ex6Q(Vector & x);
   static double Ex6W(Vector & x);

};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-segment.mesh";
   int order = 1;
   int nev = 15;
   int sr = 5, pr = 2;
   SturmLiouville::EQ_TYPE e = SturmLiouville::LAPLACE_EQ;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&nev, "-n", "--num-eigs",
                  "Number of desired eigenmodes.");
   args.AddOption(&sr, "-sr", "--serial-ref-levels",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-pr", "--parallel-ref-levels",
                  "Number of parallel refinement levels.");
   args.AddOption((int*)&e, "-e", "--eq",
                  "1 - Bessel, 2 - Chebyshev, 3 - Chebyshev 2nd, 4 - Laplace, 5 - Legendre.");
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

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = pr;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
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

   // 8. Set up the parallel bilinear forms a(.,.) and m(.,.) on the
   //    finite element space.  The first corresponds to the Laplacian
   //    operator -Delta, by adding the Diffusion domain integrator and
   //    imposing homogeneous Dirichlet boundary conditions. The boundary
   //    conditions are implemented by marking all the boundary attributes
   //    from the mesh as essential.  The second is a simple mass matrix
   //    needed on the right hand side of the generalized eigenvalue problem.
   //    After serial and parallel assembly we extract the corresponding
   //    parallel matrix A.
   SturmLiouville SL(MPI_COMM_WORLD, e);

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr[0] = 1;
   ess_bdr[1] = 1;
   cout << "num bdr attrib:  " << pmesh->bdr_attributes.Max() << endl;

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(SL.p()));
   a->AddDomainIntegrator(new MassIntegrator(SL.q()));
   a->Assemble();
   a->EliminateEssentialBCDiag(ess_bdr,1000.0);
   a->Finalize();

   ParBilinearForm *m = new ParBilinearForm(fespace);
   m->AddDomainIntegrator(new MassIntegrator(SL.w()));
   m->Assemble();
   m->EliminateEssentialBCDiag(ess_bdr,1.0);
   m->Finalize();

   HypreParMatrix *A = a->ParallelAssemble();
   HypreParMatrix *M = m->ParallelAssemble();

   A->Print("A.mat");
   M->Print("M.mat");

   delete a;
   delete m;

   // 9. Define and configure the LOBPCG eigensolver and a BoomerAMG
   //    preconditioner to be used within the solver.
   HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);
   HypreSolver *    amg = new HypreBoomerAMG(*A);

   lobpcg->SetNumModes(nev);
   lobpcg->SetPrecond(*amg);
   lobpcg->SetMaxIter(100);
   lobpcg->SetTol(1e-8);
   lobpcg->SetPrecondUsageMode(1);
   lobpcg->SetPrintLevel(1);

   // Set the matrices which define the linear system
   lobpcg->SetB(*M);
   lobpcg->SetA(*A);

   // Obtain the eigenvalues and eigenvectors
   Array<double> eigenvalues;

   lobpcg->Solve();
   lobpcg->GetEigenvalues(eigenvalues);

   // 10. Define a parallel grid function to approximate each of the
   //     eigenmodes returned by the solver.  Use this as a template to
   //     create a special multi-vector object needed by the eigensolver
   //     which is then initialized with random values.
   ParGridFunction x(fespace);
   x = 0.0;

   // 11. Save the refined mesh and the modes in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g mode".
   {
      ostringstream mesh_name, mode_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      for (int i=0; i<nev; i++)
      {
         x = lobpcg->GetEigenvector(i);

         mode_name << "mode_" << setfill('0') << setw(2) << i << "."
                   << setfill('0') << setw(6) << myid;

         ofstream mode_ofs(mode_name.str().c_str());
         mode_ofs.precision(8);
         x.Save(mode_ofs);
         mode_name.str("");
      }
   }

   // 12. Send the solution by socket to a GLVis server.

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
            cout << "Lambda = " << eigenvalues[i] << endl;
         }

         x = lobpcg->GetEigenvector(i);

         mode_sock << "parallel " << num_procs << " " << myid << "\n";
         mode_sock << "solution\n" << *pmesh << x << flush;

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

   // 13. Free the used memory.
   delete amg;
   delete lobpcg;
   delete M;
   delete A;

   delete fespace;
   if (order > 0)
   {
      delete fec;
   }
   delete pmesh;

   MPI_Finalize();

   return 0;
}

SturmLiouville::SturmLiouville(MPI_Comm comm, EQ_TYPE e)
{
   switch (e)
   {
      case BESSEL_EQ:
         pCoef = new FunctionCoefficient(BesselP);
         qCoef = new FunctionCoefficient(BesselQ);
         wCoef = new FunctionCoefficient(BesselW);
         break;
      case CHEBYSHEV_EQ:
         pCoef = new FunctionCoefficient(ChebyshevP);
         qCoef = new FunctionCoefficient(ChebyshevQ);
         wCoef = new FunctionCoefficient(ChebyshevW);
         break;
      case CHEBYSHEV2_EQ:
         pCoef = new FunctionCoefficient(Chebyshev2P);
         qCoef = new FunctionCoefficient(Chebyshev2Q);
         wCoef = new FunctionCoefficient(Chebyshev2W);
         break;
      case LAPLACE_EQ:
         pCoef = new FunctionCoefficient(LaplaceP);
         qCoef = new FunctionCoefficient(LaplaceQ);
         wCoef = new FunctionCoefficient(LaplaceW);
         break;
      case LEGENDRE_EQ:
         pCoef = new FunctionCoefficient(LegendreP);
         qCoef = new FunctionCoefficient(LegendreQ);
         wCoef = new FunctionCoefficient(LegendreW);
         break;
      case EX6_EQ:
         pCoef = new FunctionCoefficient(Ex6P);
         qCoef = new FunctionCoefficient(Ex6Q);
         wCoef = new FunctionCoefficient(Ex6W);
         break;
      default:
         pCoef = new FunctionCoefficient(LaplaceP);
         qCoef = new FunctionCoefficient(LaplaceQ);
         wCoef = new FunctionCoefficient(LaplaceW);
   }
}

SturmLiouville::~SturmLiouville()
{
   if ( pCoef != NULL ) { delete pCoef; }
   if ( qCoef != NULL ) { delete qCoef; }
   if ( wCoef != NULL ) { delete wCoef; }
}

double SturmLiouville::BesselP(Vector & x) { return x(0); }
double SturmLiouville::BesselQ(Vector & x) { return x(0); }
double SturmLiouville::BesselW(Vector & x) { return 1.0/x(0); }

double SturmLiouville::LaplaceP(Vector & x) { return 1.0; }
double SturmLiouville::LaplaceQ(Vector & x) { return 0.0; }
double SturmLiouville::LaplaceW(Vector & x) { return 1.0; }

double SturmLiouville::LegendreP(Vector & x) { return x(0)*(1.0-x(0)); }
double SturmLiouville::LegendreQ(Vector & x) { return 0.0; }
double SturmLiouville::LegendreW(Vector & x) { return 1.0; }

double SturmLiouville::ChebyshevP(Vector & x)
{ return sqrt((1.0-x(0))*x(0)); }
double SturmLiouville::ChebyshevQ(Vector & x)
{ return 0.0; }
double SturmLiouville::ChebyshevW(Vector & x)
{ return 1.0/sqrt((1.0-x(0))*x(0)); }

double SturmLiouville::Chebyshev2P(Vector & x)
{ return (1.0-x(0)*x(0))*sqrt(1.0-x(0)*x(0)); }
double SturmLiouville::Chebyshev2Q(Vector & x)
{ return 0.0; }
double SturmLiouville::Chebyshev2W(Vector & x)
{ return sqrt(1.0-x(0)*x(0)); }

double SturmLiouville::Ex6P(Vector & x) { return (1.0+x(0)*(M_E*M_E-1.0))/((M_E*M_E-1.0)*(M_E*M_E-1.0)); }
double SturmLiouville::Ex6Q(Vector & x) { return 0.0; }
double SturmLiouville::Ex6W(Vector & x) { return 1.0/(1.0+x(0)*(M_E*M_E-1.0)); }

