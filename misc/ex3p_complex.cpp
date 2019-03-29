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

class VectorBlochWaveProjector : public Operator
{
public:
   /*
   VectorBlochWaveProjector(HypreParMatrix & A,
          ParFiniteElementSpace & HCurlFESpace,
          Operator & irrProj, Operator & dirProj)
     : Operator(2*A.Width()), irrProj_(&irrProj), dirProj_(&dirProj)
   */
   VectorBlochWaveProjector(ParFiniteElementSpace & HCurlFESpace,
                            ParFiniteElementSpace & H1FESpace,
                            double beta, const Vector & zeta);

   ~VectorBlochWaveProjector()
   {
      delete urDummy_; delete uiDummy_; delete vrDummy_; delete viDummy_;
      delete u0_; delete v0_;
      delete amg_cos_; delete amg_sin_;
      delete S0_cos_; delete S0_sin_; delete M1_cos_; delete M1_sin_;
      delete Grad_;
   }

   virtual void Mult(const Vector &x, Vector &y) const;

private:
   int locSize_;

   // ParFiniteElementSpace * HCurlFESpace_;
   // ParFiniteElementSpace * H1FESpace_;

   HypreParMatrix * S0_cos_;
   HypreParMatrix * S0_sin_;
   HypreParMatrix * M1_cos_;
   HypreParMatrix * M1_sin_;

   HypreBoomerAMG * amg_cos_;
   HypreBoomerAMG * amg_sin_;

   ParDiscreteInterpolationOperator * Grad_;

   // Operator * irrProj_;
   // Operator * dirProj_;
   mutable HypreParVector * urDummy_;
   mutable HypreParVector * uiDummy_;
   mutable HypreParVector * vrDummy_;
   mutable HypreParVector * viDummy_;
   mutable HypreParVector * u0_;
   mutable HypreParVector * v0_;

protected:

   class CosCoefficient : public FunctionCoefficient
   {
   public:
      CosCoefficient(double beta, const Vector & zeta);

      double Eval(ElementTransformation &T,
                  const IntegrationPoint &ip);
   private:
      double beta_;
      const Vector & zeta_;
   };

   class SinCoefficient : public FunctionCoefficient
   {
   public:
      SinCoefficient(double beta, const Vector & zeta);

      double Eval(ElementTransformation &T,
                  const IntegrationPoint &ip);
   private:
      double beta_;
      const Vector & zeta_;
   };
   /*
   class BetaCoefficient {
   public:
     BetaCoefficient(double beta, const Vector & zeta);
     ~BetaCoefficient();

     FunctionCoefficient & cosCoef() { return *cosCoef_; }
     FunctionCoefficient & sinCoef() { return *sinCoef_; }

   private:
     double beta_;
     const Vector zeta_;

     FunctionCoefficient * cosCoef_;
     FunctionCoefficient * sinCoef_;

     protected:

     // static double cosFunc_(const Vector & x);
     // static double sinFunc_(const Vector & x);
   };
   */
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

VectorBlochWaveProjector::CosCoefficient::CosCoefficient(double beta,
                                                         const Vector & zeta)
   : FunctionCoefficient((double(*)(const Vector &))NULL),
     beta_(beta),
     zeta_(zeta)
{}

double
VectorBlochWaveProjector::CosCoefficient::Eval(ElementTransformation & T,
                                               const IntegrationPoint & ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   return ( cos(beta_ * (transip * zeta_) ) );
}

VectorBlochWaveProjector::SinCoefficient::SinCoefficient(double beta,
                                                         const Vector & zeta)
   : FunctionCoefficient((double(*)(const Vector &))NULL),
     beta_(beta),
     zeta_(zeta)
{}

double
VectorBlochWaveProjector::SinCoefficient::Eval(ElementTransformation & T,
                                               const IntegrationPoint & ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   return ( sin(beta_ * (transip * zeta_) ) );
}

VectorBlochWaveProjector::VectorBlochWaveProjector(
   ParFiniteElementSpace & HCurlFESpace,
   ParFiniteElementSpace & H1FESpace,
   double beta, const Vector & zeta)
   : Operator(2*HCurlFESpace.GlobalTrueVSize())/*,
    HCurlFESpace_(&HCurlFESpace),
    H1FESpace_(&H1FESpace)*/
{
   cout << "Constructing VectorBlochWaveProjector" << endl;
   locSize_ = HCurlFESpace.TrueVSize();

   urDummy_ = new HypreParVector(HCurlFESpace.GetComm(),
                                 HCurlFESpace.GlobalTrueVSize(),
                                 NULL,
                                 HCurlFESpace.GetTrueDofOffsets());
   uiDummy_ = new HypreParVector(HCurlFESpace.GetComm(),
                                 HCurlFESpace.GlobalTrueVSize(),
                                 NULL,
                                 HCurlFESpace.GetTrueDofOffsets());

   vrDummy_ = new HypreParVector(HCurlFESpace.GetComm(),
                                 HCurlFESpace.GlobalTrueVSize(),
                                 NULL,
                                 HCurlFESpace.GetTrueDofOffsets());
   viDummy_ = new HypreParVector(HCurlFESpace.GetComm(),
                                 HCurlFESpace.GlobalTrueVSize(),
                                 NULL,
                                 HCurlFESpace.GetTrueDofOffsets());

   u0_ = new HypreParVector(&H1FESpace);
   v0_ = new HypreParVector(&H1FESpace);

   CosCoefficient cosCoef(beta,zeta);
   SinCoefficient sinCoef(beta,zeta);

   cout << "Building M1(cos)" << endl;
   ParBilinearForm m1_cos(&HCurlFESpace);
   m1_cos.AddDomainIntegrator(new VectorFEMassIntegrator(cosCoef));
   m1_cos.Assemble();
   m1_cos.Finalize();
   M1_cos_ = m1_cos.ParallelAssemble();

   cout << "Building M1(sin)" << endl;
   ParBilinearForm m1_sin(&HCurlFESpace);
   m1_sin.AddDomainIntegrator(new VectorFEMassIntegrator(sinCoef));
   m1_sin.Assemble();
   m1_sin.Finalize();
   M1_sin_ = m1_sin.ParallelAssemble();

   cout << "Building S0(cos)" << endl;
   ParBilinearForm s0_cos(&H1FESpace);
   s0_cos.AddDomainIntegrator(new DiffusionIntegrator(cosCoef));
   s0_cos.Assemble();
   s0_cos.Finalize();
   S0_cos_ = s0_cos.ParallelAssemble();

   cout << "Building S0(sin)" << endl;
   ParBilinearForm s0_sin(&H1FESpace);
   s0_sin.AddDomainIntegrator(new DiffusionIntegrator(sinCoef));
   s0_sin.Assemble();
   s0_sin.Finalize();
   S0_sin_ = s0_sin.ParallelAssemble();

   amg_cos_  = new HypreBoomerAMG(*S0_cos_);
   amg_sin_  = new HypreBoomerAMG(*S0_sin_);

   Grad_ = new ParDiscreteGradOperator(&H1FESpace,&HCurlFESpace);
}

void
VectorBlochWaveProjector::Mult(const Vector &x, Vector &y) const
{
   cout << "VectorBlochWaveProjector::Mult" << endl;
   double * data_X = (double*)x.GetData();
   double * data_Y = (double*)y;

   urDummy_->SetData(&data_X[0]);
   uiDummy_->SetData(&data_X[locSize_]);

   vrDummy_->SetData(&data_Y[0]);
   viDummy_->SetData(&data_Y[locSize_]);

   M1_cos_->Mult(*urDummy_,*vrDummy_);
   M1_sin_->Mult(*uiDummy_,*vrDummy_,1.0,1.0);
   Grad_->MultTranspose(*vrDummy_,*u0_);
   amg_cos_->Mult(*u0_,*v0_);

   *vrDummy_ = *urDummy_;

   Grad_->Mult(*v0_,*vrDummy_,-1.0,1.0);

   M1_cos_->Mult(*uiDummy_,*viDummy_);
   M1_sin_->Mult(*urDummy_,*viDummy_,-1.0,1.0);
   Grad_->MultTranspose(*viDummy_,*u0_);
   amg_cos_->Mult(*u0_,*v0_);

   *viDummy_ = *uiDummy_;

   Grad_->Mult(*v0_,*viDummy_,-1.0,1.0);

   /*
   irrProj_->Mult(*uDummy_,*vDummy_);
   dirProj_->Mult(*vDummy_,*u_);
   *vDummy_ = *uDummy_;
   *vDummy_ -= *u_;


   irrProj_->Mult(*uDummy_,*vDummy_);
   dirProj_->Mult(*vDummy_,*u_);
   *vDummy_ = *uDummy_;
   *vDummy_ -= *u_;
   */
   /*
   irrProj_->Mult(x,y);
   dirProj_->Mult(y,*u_);
   y -= *u_;
   */
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
