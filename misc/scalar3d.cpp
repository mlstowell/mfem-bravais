#include "mfem.hpp"
#include "../common/pfem_extras.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <set>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;

class WeightedBlockJacobiPrecond : public Solver
{
public:
   WeightedBlockJacobiPrecond(MPI_Comm comm,
                              BlockDiagonalPreconditioner & BDP,
                              BlockOperator & LU,
                              double w)
      : comm_(comm), BDP_(&BDP), A_(NULL), LU_(&LU), r_(NULL), v_(NULL), w_(w)
   {
      MPI_Comm_size(comm,&numProcs_);
   }

   ~WeightedBlockJacobiPrecond()
   {
      delete r_;
      delete v_;
   }

   void SetOperator(const Operator & A)
   {
      A_ = &A;

      int locSize = A.Width();
      int glbSize;
      HYPRE_Int * part;

      if (HYPRE_AssumedPartitionCheck())
      {
         part = new HYPRE_Int[2];

         MPI_Scan(&locSize, &part[1], 1, HYPRE_MPI_INT, MPI_SUM, comm_);

         part[0] = part[1] - locSize;
         // part[1]++;

         MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm_);
      }
      else
      {
         part = new HYPRE_Int[numProcs_+1];

         MPI_Allgather(&locSize, 1, MPI_INT,
                       &part[1], 1, HYPRE_MPI_INT, comm_);

         part[0] = 0;
         for (int i=0; i<numProcs_; i++)
         {
            part[i+1] += part[i];
         }

         glbSize = part[numProcs_];
      }

      r_ = new HypreParVector(comm_,glbSize,NULL,part);
      v_ = new HypreParVector(comm_,glbSize,NULL,part);

   }

   void Mult(const Vector & x, Vector &y) const
   {
      y   = 0.0;

      for (int i=0; i<5; i++)
      {
         A_->Mult(y,*r_);
         *r_ *= -1.0;
         *r_ += x;

         BDP_->Mult(*r_,*v_);
         *v_ *= w_;
         y += *v_;
      }

   }

private:

   MPI_Comm comm_;
   int numProcs_;

   BlockDiagonalPreconditioner * BDP_;
   const Operator * A_;
   BlockOperator * LU_;
   mutable HypreParVector * r_,*v_;
   double w_;
};

class ScalarFloquetWaveEquation
{
public:
   ScalarFloquetWaveEquation(ParMesh & pmesh, int order);

   ~ScalarFloquetWaveEquation();

   HYPRE_Int *GetTrueDofOffsets() { return tdof_offsets_; }

   void SetBeta(double beta) { beta_ = beta; }
   void SetAzimuth(double alpha_a) { alpha_a_ = alpha_a; }
   void SetInclination(double alpha_i) { alpha_i_ = alpha_i; }

   void SetMassCoef(Coefficient & m) { mCoef_ = &m; }
   void SetStiffnessCoef(Coefficient & k) { kCoef_ = &k; }
   void Setup();

   BlockOperator * GetAOperator() { return A_; }
   BlockOperator * GetMOperator() { return M_; }

   BlockDiagonalPreconditioner * GetBDP() { return BDP_; }

   Solver * GetPrecond() { return WBJP_; }

   ParFiniteElementSpace * GetFESpace() { return H1FESpace_; }

private:

   H1_ParFESpace  * H1FESpace_;
   ND_ParFESpace  * HCurlFESpace_;

   double           alpha_a_;
   double           alpha_i_;
   double           beta_;
   Vector           zeta_;

   Coefficient    * mCoef_;
   Coefficient    * kCoef_;

   Array<int>       block_offsets_;
   Array<int>       block_trueOffsets_;
   Array<HYPRE_Int> tdof_offsets_;

   BlockOperator  * A_;
   BlockOperator  * M_;
   BlockOperator * LU_;

   HypreParMatrix * A0_;
   HypreParMatrix * M0_;
   HypreParMatrix * M1_;
   HypreParMatrix * S0_;
   // HypreParMatrix * T0_;

   HypreParMatrix * DKZ_;
   // HypreParMatrix * DKZT_;

   HypreBoomerAMG * S0Inv_;

   ParDiscreteInterpolationOperator * Grad_;
   ParDiscreteInterpolationOperator * Zeta_;

   BlockDiagonalPreconditioner * BDP_;
   WeightedBlockJacobiPrecond  * WBJP_;
};

class LinearCombinationIntegrator : public BilinearFormIntegrator
{
private:
   int own_integrators;
   DenseMatrix elem_mat;
   Array<BilinearFormIntegrator*> integrators;
   Array<double> c;

public:
   LinearCombinationIntegrator(int own_integs = 1)
   { own_integrators = own_integs; }

   void AddIntegrator(double scalar, BilinearFormIntegrator *integ)
   { integrators.Append(integ); c.Append(scalar); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual ~LinearCombinationIntegrator();
};
/*
class ShiftedGradientIntegrator: public BilinearFormIntegrator
{
public:
   ShiftedGradientIntegrator(VectorCoefficient &vq,
                             const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), VQ(&vq) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

private:
   Vector      shape;
   DenseMatrix dshape;
   DenseMatrix dshapedxt;
   DenseMatrix invdfdx;
   Vector      D;
   VectorCoefficient *VQ;
};
*/
void * mfem_BlockOperatorMatvecCreate( void *A,
                                       void *x )
{
   void *matvec_data;

   matvec_data = NULL;

   return ( matvec_data );
}

HYPRE_Int mfem_BlockOperatorMatvec( void *matvec_data,
                                    HYPRE_Complex alpha,
                                    void *A,
                                    void *x,
                                    HYPRE_Complex beta,
                                    void *y )
{
   BlockOperator *Aop = (BlockOperator*)A;

   int width = Aop->Width();

   hypre_ParVector * xPar = (hypre_ParVector *)x;
   hypre_ParVector * yPar = (hypre_ParVector *)y;

   Vector xVec(xPar->local_vector->data, width);
   Vector yVec(yPar->local_vector->data, width);

   Aop->Mult( xVec, yVec );

   return 0;
}

HYPRE_Int mfem_BlockOperatorMatvecDestroy( void *matvec_data )
{
   return 0;
}

HYPRE_Int
mfem_BlockOperatorAMGSolve(void *solver,
                           void *A,
                           void *b,
                           void *x)
{
   BlockOperator *Aop = (BlockOperator*)A;

   int width = Aop->Width();

   hypre_ParVector * bPar = (hypre_ParVector *)b;
   hypre_ParVector * xPar = (hypre_ParVector *)x;

   Vector bVec(bPar->local_vector->data, width);
   Vector xVec(xPar->local_vector->data, width);

   Aop->Mult( bVec, xVec );

   return 0;
}

HYPRE_Int
mfem_BlockOperatorAMGSetup(void *solver,
                           void *A,
                           void *b,
                           void *x)
{
   return 0;
}

// Material Coefficients
double mass_coef(const Vector &);
double stiffness_coef(const Vector &);

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
   int sr = 0, pr = 2;
   bool visualization = 1;
   bool visit = true;
   int nev = 5;
   double beta = 1.0;
   double alpha_a = 0.0, alpha_i = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sr, "-sr", "--serial-refinement",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-pr", "--parallel-refinement",
                  "Number of parallel refinement levels.");
   args.AddOption(&nev, "-nev", "--num_eigs",
                  "Number of eigenvalues requested.");
   args.AddOption(&beta, "-b", "--phase-shift",
                  "Phase Shift Magnitude in degrees");
   args.AddOption(&alpha_a, "-az", "--azimuth",
                  "Azimuth in degrees");
   args.AddOption(&alpha_i, "-inc", "--inclination",
                  "Inclination in degrees");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visualization",
                  "Enable or disable VisIt visualization.");
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

   L2_ParFESpace * L2FESpace = new L2_ParFESpace(pmesh,0,pmesh->Dimension());

   int nElems = L2FESpace->GetVSize();
   cout << myid << ": nElems = " << nElems << endl;

   ParGridFunction * m = new ParGridFunction(L2FESpace);
   ParGridFunction * k = new ParGridFunction(L2FESpace);

   FunctionCoefficient mFunc(mass_coef);
   FunctionCoefficient kFunc(stiffness_coef);

   m->ProjectCoefficient(mFunc);
   k->ProjectCoefficient(kFunc);

   if (visualization)
   {
      socketstream m_sock, k_sock;
      char vishost[] = "localhost";
      int  visport   = 19916;

      m_sock.open(vishost, visport);
      k_sock.open(vishost, visport);
      m_sock.precision(8);
      k_sock.precision(8);

      m_sock << "parallel " << num_procs << " " << myid << "\n"
             << "solution\n" << *pmesh << *m << flush
             << "window_title 'Mass Coefficient'\n" << flush;

      k_sock << "parallel " << num_procs << " " << myid << "\n"
             << "solution\n" << *pmesh << *k << flush
             << "window_title 'Stiffness Coefficient'\n" << flush;
   }

   GridFunctionCoefficient mCoef(m);
   GridFunctionCoefficient kCoef(k);

   ScalarFloquetWaveEquation * eq =
      new ScalarFloquetWaveEquation(*pmesh, order);

   eq->SetBeta(beta);
   eq->SetAzimuth(alpha_a);
   eq->SetInclination(alpha_i);

   eq->SetMassCoef(mCoef);
   eq->SetStiffnessCoef(kCoef);

   // eq->SetSigma(sigma);

   eq->Setup();

   // 9. Define and configure the LOBPCG eigensolver and a BoomerAMG
   //    preconditioner to be used within the solver.
   HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);

   lobpcg->SetNumModes(nev);
   lobpcg->SetPreconditioner(*eq->GetPrecond());
   lobpcg->SetMaxIter(1000);
   lobpcg->SetTol(1e-6);
   lobpcg->SetPrecondUsageMode(1);
   lobpcg->SetPrintLevel(1);

   // Set the matrices which define the linear system
   lobpcg->SetMassMatrix(*eq->GetMOperator());
   lobpcg->SetOperator(*eq->GetAOperator());

   // Obtain the eigenvalues and eigenvectors
   Array<double> eigenvalues;

   lobpcg->Solve();
   lobpcg->GetEigenvalues(eigenvalues);

   if ( visit )
   {
      ParGridFunction ur(eq->GetFESpace());
      ParGridFunction ui(eq->GetFESpace());

      int h1_loc_size = eq->GetFESpace()->TrueVSize();

      HypreParVector urVec(eq->GetFESpace()->GetComm(),
                           eq->GetFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetFESpace()->GetTrueDofOffsets());
      HypreParVector uiVec(eq->GetFESpace()->GetComm(),
                           eq->GetFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetFESpace()->GetTrueDofOffsets());

      VisItDataCollection visit_dc("Scalar3D-Parallel", pmesh);
      visit_dc.RegisterField("m", m);
      visit_dc.RegisterField("k", k);
      visit_dc.RegisterField("mode_r", &ur);
      visit_dc.RegisterField("mode_i", &ui);

      for (int i=0; i<nev; i++)
      {
         //eq->TestVector(lobpcg->GetEigenvector(i));
         double * data = (double*)lobpcg->GetEigenvector(i);
         urVec.SetData(&data[0]);
         uiVec.SetData(&data[h1_loc_size]);

         ur = urVec;
         ui = uiVec;

         visit_dc.SetCycle(i+1);
         visit_dc.SetTime(eigenvalues[i]);
         visit_dc.Save();
      }
   }

   if (visualization)
   {
      socketstream ur_sock, ui_sock;
      char vishost[] = "localhost";
      int  visport   = 19916;

      ur_sock.open(vishost, visport);
      ui_sock.open(vishost, visport);
      ur_sock.precision(8);
      ui_sock.precision(8);

      ParGridFunction ur(eq->GetFESpace());
      ParGridFunction ui(eq->GetFESpace());

      int h1_loc_size = eq->GetFESpace()->TrueVSize();

      HypreParVector urVec(eq->GetFESpace()->GetComm(),
                           eq->GetFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetFESpace()->GetTrueDofOffsets());
      HypreParVector uiVec(eq->GetFESpace()->GetComm(),
                           eq->GetFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetFESpace()->GetTrueDofOffsets());

      for (int e=0; e<nev; e++)
      {
         if ( myid == 0 )
         {
            cout << e << ":  Eigenvalue " << eigenvalues[e];
            double trans_eig = eigenvalues[e];
            if ( trans_eig > 0.0 )
            {
               cout << ", omega " << sqrt(trans_eig);
            }
            cout << endl;
         }

         double * data = (double*)lobpcg->GetEigenvector(e);
         urVec.SetData(&data[0]);
         uiVec.SetData(&data[h1_loc_size]);

         ur = urVec;
         ui = uiVec;

         ur_sock << "parallel " << num_procs << " " << myid << "\n"
                 << "solution\n" << *pmesh << ur << flush
                 << "window_title 'Re(u)'\n" << flush;
         ui_sock << "parallel " << num_procs << " " << myid << "\n"
                 << "solution\n" << *pmesh << ui << flush
                 << "window_title 'Im(u)'\n" << flush;

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
      ur_sock.close();
      ui_sock.close();
   }

   delete lobpcg;
   delete eq;
   delete pmesh;

   MPI_Finalize();

   cout << "Exiting Main" << endl;

   return 0;
}

double mass_coef(const Vector & x)
{
   if ( x.Norml2() <= 0.5 )
      //if ( fabs(x(0)) <= 0.5 )
   {
      return 10.0;
   }
   else
   {
      return 1.0;
   }
}

double stiffness_coef(const Vector &x)
{
   if ( x.Norml2() <= 0.5 )
      // if ( fabs(x(0)) <= 0.5 )
   {
      return 5.0;
   }
   else
   {
      return 0.1;
   }
}

ScalarFloquetWaveEquation::ScalarFloquetWaveEquation(ParMesh & pmesh,
                                                     int order)
   : H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     alpha_a_(0.0),
     alpha_i_(90.0),
     beta_(0.0),
     mCoef_(NULL),
     kCoef_(NULL),
     A_(NULL),
     M_(NULL),
     A0_(NULL),
     M0_(NULL),
     S0_(NULL),
     // T0_(NULL),
     DKZ_(NULL),
     // DKZT_(NULL),
     S0Inv_(NULL),
     Grad_(NULL),
     Zeta_(NULL),
     BDP_(NULL),
     WBJP_(NULL)
{
   int dim = pmesh.Dimension();

   zeta_.SetSize(dim);

   H1FESpace_    = new H1_ParFESpace(&pmesh,order,dim);
   HCurlFESpace_ = new ND_ParFESpace(&pmesh,order,dim);

   block_offsets_.SetSize(3);
   block_offsets_[0] = 0;
   block_offsets_[1] = H1FESpace_->GetVSize();
   block_offsets_[2] = H1FESpace_->GetVSize();
   block_offsets_.PartialSum();

   block_trueOffsets_.SetSize(3);
   block_trueOffsets_[0] = 0;
   block_trueOffsets_[1] = H1FESpace_->TrueVSize();
   block_trueOffsets_[2] = H1FESpace_->TrueVSize();
   block_trueOffsets_.PartialSum();

   tdof_offsets_.SetSize(H1FESpace_->GetNRanks()+1);
   HYPRE_Int *    h1_tdof_offsets = H1FESpace_->GetTrueDofOffsets();
   for (int i=0; i<tdof_offsets_.Size(); i++)
   {
      tdof_offsets_[i] = 2 * h1_tdof_offsets[i];
   }

}

ScalarFloquetWaveEquation::~ScalarFloquetWaveEquation()
{
   if ( A_         != NULL ) { delete A_; }
   if ( M_         != NULL ) { delete M_; }
   delete LU_;
   delete WBJP_;
   if ( BDP_       != NULL ) { delete BDP_; }
   if ( S0Inv_     != NULL ) { delete S0Inv_; }
   if ( A0_        != NULL ) { delete A0_; }
   if ( M0_        != NULL ) { delete M0_; }
   if ( M1_        != NULL ) { delete M1_; }
   if ( S0_        != NULL ) { delete S0_; }
   // if ( T0_        != NULL ) { delete T0_; }
   if ( DKZ_       != NULL ) { delete DKZ_; }
   // if ( DKZT_      != NULL ) { delete DKZT_; }
   delete Grad_;
   delete Zeta_;

   if ( H1FESpace_ != NULL ) { delete H1FESpace_; }
}

void
ScalarFloquetWaveEquation::Setup()
{
   if ( zeta_.Size() == 3 )
   {
      zeta_[0] = cos(alpha_i_*M_PI/180.0)*cos(alpha_a_*M_PI/180.0);
      zeta_[1] = cos(alpha_i_*M_PI/180.0)*sin(alpha_a_*M_PI/180.0);
      zeta_[2] = sin(alpha_i_*M_PI/180.0);
   }
   else
   {
      zeta_[0] = cos(alpha_a_*M_PI/180.0);
      zeta_[1] = sin(alpha_a_*M_PI/180.0);
   }
   cout << "Phase Shift: " << beta_ << " (deg)"<< endl;
   cout << "Zeta:  ";
   zeta_.Print(cout);
   /*
   VectorCoefficient * zCoef = NULL;

   if ( kCoef_ )
   {
      zCoef = new VectorFunctionCoefficient(zeta_,*kCoef_);
   }
   else
   {
      zCoef = new VectorConstantCoefficient(zeta_);
   }

   ParBilinearForm dkz(H1FESpace_);
   dkz.AddDomainIntegrator(new ShiftedGradientIntegrator(*zCoef));
   dkz.Assemble();
   dkz.Finalize();

   DKZ_  = dkz.ParallelAssemble();
   DKZT_ = DKZ_->Transpose();

   delete zCoef;
   */
   cout << "Building M0" << endl;
   ParBilinearForm m0(H1FESpace_);
   m0.AddDomainIntegrator(new MassIntegrator(*mCoef_));
   m0.Assemble();
   m0.Finalize();
   M0_ = m0.ParallelAssemble();

   cout << "Building M1" << endl;
   ParBilinearForm m1(HCurlFESpace_);
   m1.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoef_));
   m1.Assemble();
   m1.Finalize();
   M1_ = m1.ParallelAssemble();

   cout << "Building Grad and Zeta" << endl;
   Grad_ = new ParDiscreteGradOperator(H1FESpace_,HCurlFESpace_);
   Zeta_ = new ParDiscreteVectorProductOperator(H1FESpace_,
                                                HCurlFESpace_,zeta_);

   HypreParMatrix * GMG = RAP(M1_,Grad_->ParallelAssemble());

   // T0_ = ParAdd(GMG,M0_);

   if ( fabs(beta_) > 0.0 )
   {
      HypreParMatrix * ZMZ = RAP(M1_,Zeta_->ParallelAssemble());

      HypreParMatrix * GMZ = RAP(Grad_->ParallelAssemble(),M1_,
                                 Zeta_->ParallelAssemble());
      HypreParMatrix * ZMG = RAP(Zeta_->ParallelAssemble(),M1_,
                                 Grad_->ParallelAssemble());

      *ZMZ *= beta_*beta_*M_PI*M_PI/32400.0;
      S0_ = ParAdd(GMG,ZMZ);

      *ZMG *= -1.0;
      DKZ_ = ParAdd(GMZ,ZMG);

      delete GMG;
      delete ZMZ;
      delete GMZ;
      delete ZMG;
   }
   else
   {
      S0_ = GMG;
   }
   /*

   cout << "Building A0" << endl;
   LinearCombinationIntegrator * bfi = new LinearCombinationIntegrator();
   if ( fabs(beta_) > 0.0 )
   {
      bfi->AddIntegrator(beta_*beta_*M_PI*M_PI/32400.0,
                         new MassIntegrator(*kCoef_));
   }
   bfi->AddIntegrator(1.0, new DiffusionIntegrator(*kCoef_));
   ParBilinearForm a0(H1FESpace_);
   a0.AddDomainIntegrator(bfi);
   a0.Assemble();
   a0.Finalize();
   A0_ = a0.ParallelAssemble();

   ParBilinearForm s0(H1FESpace_);
   s0.AddDomainIntegrator(new DiffusionIntegrator(*kCoef_));
   s0.AddDomainIntegrator(new MassIntegrator(*mCoef_));
   s0.Assemble();
   s0.Finalize();
   S0_ = s0.ParallelAssemble();
   */
   /*
   A_ = new BlockOperator(block_trueOffsets_);
   A_->SetDiagonalBlock(0,A0_);
   A_->SetDiagonalBlock(1,A0_);
   A_->SetBlock(0,1,DKZ_,beta_*M_PI/180.0);
   A_->SetBlock(1,0,DKZT_,beta_*M_PI/180.0);
   A_->owns_blocks = 0;
   */
   A_ = new BlockOperator(block_trueOffsets_);
   A_->SetDiagonalBlock(0,S0_);
   A_->SetDiagonalBlock(1,S0_);
   if ( fabs(beta_) > 0.0 )
   {
      A_->SetBlock(0,1,DKZ_,beta_*M_PI/180.0);
      A_->SetBlock(1,0,DKZ_,-beta_*M_PI/180.0);
   }
   A_->owns_blocks = 0;

   M_ = new BlockOperator(block_trueOffsets_);
   M_->SetDiagonalBlock(0,M0_);
   M_->SetDiagonalBlock(1,M0_);
   M_->owns_blocks = 0;

   LU_ = new BlockOperator(block_trueOffsets_);
   if ( fabs(beta_) > 0.0 )
   {
      LU_->SetBlock(0,1,DKZ_,beta_*M_PI/180.0);
      LU_->SetBlock(1,0,DKZ_,-beta_*M_PI/180.0);
   }
   LU_->owns_blocks = 0;

   S0Inv_ = new HypreBoomerAMG(*S0_);
   BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets_);
   BDP_->SetDiagonalBlock(0,S0Inv_);
   BDP_->SetDiagonalBlock(1,S0Inv_);
   BDP_->owns_blocks = 0;

   WBJP_ = new WeightedBlockJacobiPrecond(MPI_COMM_WORLD,*BDP_, *LU_, 0.5);
   WBJP_->SetOperator(*A_);
   /*
   T0Inv_ = new HypreBoomerAMG(*T0_);
   BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets_);
   BDP_->SetDiagonalBlock(0,T0Inv_);
   BDP_->SetDiagonalBlock(1,T0Inv_);
   BDP_->owns_blocks = 0;
   */
   cout << "Leaving Setup" << endl;
}

void LinearCombinationIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   MFEM_ASSERT(integrators.Size() > 0, "empty LinearCombinationIntegrator.");

   integrators[0]->AssembleElementMatrix(el, Trans, elmat);
   elmat *= c[0];
   for (int i = 1; i < integrators.Size(); i++)
   {
      integrators[i]->AssembleElementMatrix(el, Trans, elem_mat);
      elem_mat *= c[i];
      elmat += elem_mat;
   }
}

LinearCombinationIntegrator::~LinearCombinationIntegrator()
{
   if (own_integrators)
   {
      for (int i = 0; i < integrators.Size(); i++)
      {
         delete integrators[i];
      }
   }
}
/*
void ShiftedGradientIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   double w;

   elmat.SetSize(nd);
   shape.SetSize(nd);
   dshape.SetSize(nd,dim);
   dshapedxt.SetSize(nd,dim);
   invdfdx.SetSize(dim,dim);
   D.SetSize(dim);

   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // int order = 2 * el.GetOrder();
      int order = 2 * el.GetOrder() + Trans.OrderW();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint (&ip);
      CalcAdjugate(Trans.Jacobian(), invdfdx);
      w = Trans.Weight() * ip.weight;
      Mult(dshape, invdfdx, dshapedxt);

      VQ -> Eval(D, Trans, ip);
      D *= w;

      for (int d = 0; d < dim; d++)
      {
         for (int j = 0; j < nd; j++)
         {
            for (int k = 0; k < nd; k++)
            {
               elmat(j, k) += dshapedxt(j,d) * D[d] * shape(k)
                              - shape(j) * D[d] * dshapedxt(k,d);
            }
         }
      }
   }
}
*/
