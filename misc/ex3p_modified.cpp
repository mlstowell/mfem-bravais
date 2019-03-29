//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p -m ../data/star.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex3p -m ../data/escher.mesh
//               mpirun -np 4 ex3p -m ../data/fichera.mesh
//               mpirun -np 4 ex3p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex3p -m ../data/fichera-q3.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/amr-quad.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/amr-hex.mesh
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include "pfem_extras.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/// This class computes the irrotational portion of a vector field.
/// This vector field must be discretized using Nedelec basis
/// functions.
class IrrotationalProjector : public Operator
{
public:
   IrrotationalProjector(ParFiniteElementSpace & HCurlFESpace,
                         ParFiniteElementSpace & H1FESpace);
   virtual ~IrrotationalProjector();

   // Given a vector 'x' of Nedelec DoFs for an arbitrary vector field,
   // compute the Nedelec DoFs of the irrotational portion, 'y', of
   // this vector field.  The resulting vector will satisfy Curl y = 0
   // to machine precision.
   virtual void Mult(const Vector &x, Vector &y) const;

private:
   HypreBoomerAMG * amg_;
   HypreParMatrix * S0_;
   HypreParMatrix * M1_;
   ParDiscreteInterpolationOperator * Grad_;
   HypreParVector * yPot_;
   HypreParVector * xDiv_;
};

class DirectionalProjector : public Operator
{
public:
   DirectionalProjector(ParFiniteElementSpace & HCurlFESpace,
                        HypreParMatrix & M1, const Vector & zeta,
                        Coefficient * c = NULL);
   virtual ~DirectionalProjector();

   virtual void Mult(const Vector &x, Vector &y) const;

private:
   HyprePCG       * pcg_;
   HypreParMatrix * M1_;
   HypreParMatrix * M1zoz_;
   HypreParVector * xDual_;
};

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/periodic-cube.mesh";
   int order = 1;
   bool visualization = 1;
   double alpha_a = 0.0, alpha_i = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&alpha_a, "-az", "--azimuth",
                  "Azimuth in degrees");
   args.AddOption(&alpha_i, "-inc", "--inclination",
                  "Inclination in degrees");
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
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   pmesh->ReorientTetMesh();

   Vector zeta(dim);

   zeta[0] = cos(alpha_i*M_PI/180.0)*cos(alpha_a*M_PI/180.0);
   zeta[1] = cos(alpha_i*M_PI/180.0)*sin(alpha_a*M_PI/180.0);
   zeta[2] = sin(alpha_i*M_PI/180.0);

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   ParFiniteElementSpace *h1_fespace = new ParFiniteElementSpace(pmesh, h1_fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << size << endl;
   }
   /*
   // 7. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient f(dim, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();
   */
   // 8. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   ParGridFunction x_irr(fespace);
   ParGridFunction x_dir(fespace);
   VectorFunctionCoefficient E(dim, E_exact);
   x.ProjectCoefficient(E);
   HypreParVector *X = x.ParallelProject();
   HypreParVector *X_Irr = x_irr.ParallelProject();
   HypreParVector *X_Dir = x_dir.ParallelProject();

   ParBilinearForm *m1 = new ParBilinearForm(fespace);
   m1->AddDomainIntegrator(new VectorFEMassIntegrator());
   m1->Assemble();
   m1->Finalize();
   HypreParMatrix *M1 = m1->ParallelAssemble();
   delete m1;

   Operator * IrrProj = new IrrotationalProjector(*fespace,*h1_fespace);
   Operator * DirProj = new DirectionalProjector(*fespace,*M1,zeta);

   IrrProj->Mult(*X,*X_Irr);
   DirProj->Mult(*X,*X_Dir);

   x_irr = *X_Irr;
   x_dir = *X_Dir;

   delete IrrProj;
   delete DirProj;

   /*
   // 9. Set up the parallel bilinear form corresponding to the EM diffusion
   //    operator curl muinv curl + sigma I, by adding the curl-curl and the
   //    mass domain integrators and finally imposing non-homogeneous Dirichlet
   //    boundary conditions. The boundary conditions are implemented by
   //    marking all the boundary attributes from the mesh as essential
   //    (Dirichlet). After serial and parallel assembly we extract the
   //    parallel matrix A.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));
   a->Assemble();
   a->Finalize();

   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   HypreParVector *X = x.ParallelProject();

   // 11. Eliminate essential BC from the parallel system
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   a->ParallelEliminateEssentialBC(ess_bdr, *A, *X, *B);

   *X = 0.0;

   delete a;
   delete sigma;
   delete muinv;
   delete b;
   */

   /*
   // 12. Define and apply a parallel PCG solver for AX=B with the AMS
   //     preconditioner from hypre.
   HypreSolver *ams = new HypreAMS(*A, fespace);
   HyprePCG *pcg = new HyprePCG(*A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*ams);
   pcg->Mult(*B, *X);
   */
   // 13. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor.
   // x = *X;
   /*
   // 14. Compute and print the L^2 norm of the error.
   {
      double err = x.ComputeL2Error(E);
      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
      }
   }
   */
   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name, sol_p_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;
      sol_p_name << "sol_p." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);

      ofstream sol_p_ofs(sol_p_name.str().c_str());
      sol_p_ofs.precision(8);
      x_dir.Save(sol_p_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
      sol_sock << "window_title 'Original Field'\n" << flush;

      socketstream sol_sock_i(vishost, visport);
      sol_sock_i << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_i.precision(8);
      sol_sock_i << "solution\n" << *pmesh << x_irr << flush;
      sol_sock_i << "window_title 'Irrotational Field'\n" << flush;

      socketstream sol_sock_p(vishost, visport);
      sol_sock_p << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_p.precision(8);
      sol_sock_p << "solution\n" << *pmesh << x_dir << flush;
      sol_sock_p << "window_title 'Projected Field'\n" << flush;
   }

   // 17. Free the used memory.
   // delete pcg;
   // delete ams;
   delete X;
   delete X_Irr;
   delete X_Dir;
   // delete B;
   // delete A;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

IrrotationalProjector::IrrotationalProjector(
   ParFiniteElementSpace & HCurlFESpace,
   ParFiniteElementSpace & H1FESpace)
{
   ParBilinearForm s0(&H1FESpace);
   s0.AddDomainIntegrator(new DiffusionIntegrator());
   s0.Assemble();
   s0.Finalize();
   S0_ = s0.ParallelAssemble();

   ParBilinearForm m1(&HCurlFESpace);
   m1.AddDomainIntegrator(new VectorFEMassIntegrator());
   m1.Assemble();
   m1.Finalize();
   M1_ = m1.ParallelAssemble();

   Grad_ = new ParDiscreteGradOperator(&H1FESpace,&HCurlFESpace);

   amg_  = new HypreBoomerAMG(*S0_);

   xDiv_ = new HypreParVector(&H1FESpace);
   yPot_ = new HypreParVector(&H1FESpace);
}

IrrotationalProjector::~IrrotationalProjector()
{
   if ( amg_  != NULL ) { delete amg_; }
   if ( S0_   != NULL ) { delete S0_; }
   if ( M1_   != NULL ) { delete M1_; }
   if ( Grad_ != NULL ) { delete Grad_; }
   if ( xDiv_ != NULL ) { delete xDiv_; }
   if ( yPot_ != NULL ) { delete yPot_; }
}

void
IrrotationalProjector::Mult(const Vector &x, Vector &y) const
{
   M1_->Mult(x,y);
   Grad_->MultTranspose(y,*xDiv_);
   amg_->Mult(*xDiv_,*yPot_);
   Grad_->Mult(*yPot_,y);
}

DirectionalProjector::DirectionalProjector(
   ParFiniteElementSpace & HCurlFESpace,
   HypreParMatrix & M1,
   const Vector & zeta,
   Coefficient * c)
   : Operator(M1.Width()),
     M1_(&M1)
{
   xDual_ = new HypreParVector(M1);
   pcg_   = new HyprePCG(M1);

   pcg_->SetTol(1.0e-8);
   pcg_->SetMaxIter(100);

   MatrixCoefficient * zozCoef = NULL;

   DenseMatrix zetaOuter(zeta.Size());
   for (int i=0; i<zeta.Size(); i++)
   {
      for (int j=0; j<zeta.Size(); j++)
      {
         zetaOuter(i,j) = zeta[i] * zeta[j];
      }
   }

   if ( c != NULL )
   {
      zozCoef = new MatrixFunctionCoefficient(zetaOuter,*c);
   }
   else
   {
      zozCoef = new MatrixConstantCoefficient(zetaOuter);
   }

   ParBilinearForm m1zoz(&HCurlFESpace);
   m1zoz.AddDomainIntegrator(new VectorFEMassIntegrator(*zozCoef));
   m1zoz.Assemble();
   m1zoz.Finalize();
   M1zoz_ = m1zoz.ParallelAssemble();

   delete zozCoef;
}

DirectionalProjector::~DirectionalProjector()
{
   if ( pcg_   != NULL ) { delete pcg_; }
   if ( M1zoz_ != NULL ) { delete M1zoz_; }
   if ( xDual_ != NULL ) { delete xDual_; }
}

void
DirectionalProjector::Mult(const Vector &x, Vector &y) const
{
   M1zoz_->Mult(x,*xDual_);
   pcg_->Mult(*xDual_,y);
}

// A parameter for the exact solution.
const double kappa = M_PI;

void E_exact(const Vector &x, Vector &E)
{
   if (x.Size() == 3)
   {
      E(0) = sin(kappa*x(1)) + sin(kappa*x(0))*cos(kappa*x(1))*cos(kappa*x(2));
      E(1) = sin(kappa*x(2)) + cos(kappa*x(0))*sin(kappa*x(1))*cos(kappa*x(2));
      E(2) = sin(kappa*x(0)) + cos(kappa*x(0))*cos(kappa*x(1))*sin(kappa*x(2));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (x.Size() == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
}
