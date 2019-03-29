#include "mfem.hpp"
#include "maxwell_bloch.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <set>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;
using namespace mfem::bloch;

/// This class computes the irrotational portion of a vector field.
/// This vector field must be discretized using Nedelec basis
/// functions.
/*
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
*/
class VectorBlochWaveProjector : public Operator
{
public:
   VectorBlochWaveProjector(ParFiniteElementSpace & HDivFESpace,
                            ParFiniteElementSpace & HCurlFESpace,
                            ParFiniteElementSpace & H1FESpace,
                            double beta, const Vector & zeta,
                            Coefficient * mCoef = NULL,
                            Coefficient * kCoef = NULL);

   ~VectorBlochWaveProjector()
   {
      delete urDummy_; delete uiDummy_; delete vrDummy_; delete viDummy_;
      delete u0_; delete v0_;
      delete Grad_;
   }

   virtual void Mult(const Vector &x, Vector &y) const;

private:
   int locSize_;

   HypreParMatrix * Z_;
   HypreParMatrix * M1_;
   HypreParMatrix * A0_;
   HypreParMatrix * DKZ_;
   HypreParMatrix * DKZT_;

   HypreBoomerAMG * amg_cos_;
   MINRESSolver * minres_;

   ParDiscreteInterpolationOperator * Grad_;
   ParDiscreteInterpolationOperator * Zeta_;

   Array<int>       block_offsets0_;
   Array<int>       block_offsets1_;
   Array<int>       block_trueOffsets0_;
   Array<int>       block_trueOffsets1_;

   BlockOperator * S0_;
   BlockOperator * M_;
   BlockOperator * G_;

   mutable HypreParVector * urDummy_;
   mutable HypreParVector * uiDummy_;
   mutable HypreParVector * vrDummy_;
   mutable HypreParVector * viDummy_;

   mutable BlockVector * u0_;
   mutable BlockVector * v0_;
   mutable BlockVector * u1_;
   mutable BlockVector * v1_;
};
/*
class ParDiscreteVectorProductOperator
   : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteVectorProductOperator(ParFiniteElementSpace *dfes,
                                    ParFiniteElementSpace *rfes,
                                    const Vector & v);
};

class ParDiscreteVectorCrossProductOperator
   : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteVectorCrossProductOperator(ParFiniteElementSpace *dfes,
                                         ParFiniteElementSpace *rfes,
                                         const Vector & v);
};
*/
/** Class for constructing the vector product as a DiscreteLinearOperator
    from an H1-conforming space to an H(curl)-conforming space. The range
    space can be vector L2 space as well. */
/*
class VectorProductInterpolator : public DiscreteInterpolator
{
public:
VectorProductInterpolator(const Vector & v)
 : v_(v), sp_(v) {}

virtual void AssembleElementMatrix2(const FiniteElement &h1_fe,
                                  const FiniteElement &nd_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix &elmat)
{
 Vector nd_proj(nd_fe.GetDof());

 sp_.SetBasis(h1_fe);

 elmat.SetSize(nd_fe.GetDof(),h1_fe.GetDof());
 for (int k = 0; k < h1_fe.GetDof(); k++)
 {
    sp_.SetIndex(k);

    nd_fe.Project(sp_,Trans,nd_proj);

    for (int j = 0; j < nd_fe.GetDof(); j++)
    {
       elmat(j,k) = nd_proj(j);
    }
 }
}
private:

class ScalarProduct_ : public VectorCoefficient
{
public:
 ScalarProduct_(const Vector & v)
    : VectorCoefficient(v.Size()), v_(v), h1_(NULL), ind_(0)
 {}

 void SetBasis(const FiniteElement & h1)
 {
    h1_ = &h1;
    shape_.SetSize(h1.GetDof());
 }
 void SetIndex(int ind) { ind_ = ind; }
 void Eval(Vector & vs, ElementTransformation &T,
           const IntegrationPoint &ip)
 {
    vs.SetSize(v_.Size());

    h1_->CalcShape(ip, shape_);

    for (int i=0; i<v_.Size(); i++)
    {
       vs(i) = v_(i) * shape_(ind_);
    }
 }

private:
 const Vector & v_;
 const FiniteElement * h1_;
 Vector shape_;
 int ind_;
};

const Vector & v_;
ScalarProduct_ sp_;
};
*/
/** Class for constructing the vector cross product as a DiscreteLinearOperator
    from an H(curl)-conforming space to an H(div)-conforming space. The range
    space can be vector L2 space as well. */
/*
class VectorCrossProductInterpolator : public DiscreteInterpolator
{
public:
VectorCrossProductInterpolator(const Vector & v)
: v_(v), cp_(v) {}

virtual void AssembleElementMatrix2(const FiniteElement &nd_fe,
                         const FiniteElement &rt_fe,
                         ElementTransformation &Trans,
                         DenseMatrix &elmat)
{
Vector rt_proj(rt_fe.GetDof());

cp_.SetBasis(nd_fe);

elmat.SetSize(rt_fe.GetDof(),nd_fe.GetDof());
for (int k = 0; k < nd_fe.GetDof(); k++)
{
cp_.SetIndex(k);

rt_fe.Project(cp_,Trans,rt_proj);

for (int j = 0; j < rt_fe.GetDof(); j++)
{
elmat(j,k) = rt_proj(j);
}
}
}
private:

class CrossProduct_ : public VectorCoefficient
{
public:
CrossProduct_(const Vector & v)
: VectorCoefficient(3), v_(v), nd_(NULL), ind_(0)
{
MFEM_ASSERT( v.Size() == 3,
        "Vector Cross products are only defined in three dimensions");
}
void SetBasis(const FiniteElement & nd)
{
nd_ = &nd;
vshape_.SetSize(nd.GetDof(),v_.Size());
}
void SetIndex(int ind) { ind_ = ind; }
void Eval(Vector & vxw, ElementTransformation &T,
  const IntegrationPoint &ip)
{
vxw.SetSize(3);

nd_->CalcVShape(T, vshape_);

vxw(0) = v_(1) * vshape_(ind_,2) - v_(2) * vshape_(ind_,1);
vxw(1) = v_(2) * vshape_(ind_,0) - v_(0) * vshape_(ind_,2);
vxw(2) = v_(0) * vshape_(ind_,1) - v_(1) * vshape_(ind_,0);
}

private:
const Vector & v_;
const FiniteElement * nd_;
DenseMatrix vshape_;
int ind_;
};

const Vector & v_;
CrossProduct_ cp_;
};
*/
class AMSProj : public Operator //HypreSolver
{
public:
   // AMSProj(HypreAMS & ams, Operator & irrProj, Operator & dirProj)
   AMSProj(HypreParMatrix & A, ParFiniteElementSpace & HCurlFESpace,
           Operator & irrProj, Operator & dirProj)
      : Operator(A.Width()),
        ams_(NULL), irrProj_(&irrProj), dirProj_(&dirProj), u_(NULL), v_(NULL)
   {
      u_ = new HypreParVector(A);
      v_ = new HypreParVector(A);

      ams_ = new HypreAMS(A,&HCurlFESpace);
      ams_->SetSingularProblem();
   }

   virtual ~AMSProj() { delete u_; delete v_; }

   virtual void Mult(const Vector &x, Vector &y) const;

private:
   HypreAMS * ams_;
   Operator * irrProj_;
   Operator * dirProj_;

   mutable HypreParVector * u_;
   mutable HypreParVector * v_;
};

class VectorFloquetWaveEquation
{
public:
   VectorFloquetWaveEquation(ParMesh & pmesh, int order);

   ~VectorFloquetWaveEquation();

   HYPRE_Int *GetTrueDofOffsets() { return tdof_offsets_; }

   void SetBeta(double beta) { beta_ = beta; }
   void SetAzimuth(double alpha_a) { alpha_a_ = alpha_a; }
   void SetInclination(double alpha_i) { alpha_i_ = alpha_i; }
   // void SetSigma(double sigma) { sigma_ = sigma; }

   void SetMassCoef(Coefficient & m) { mCoef_ = &m; }
   void SetStiffnessCoef(Coefficient & k) { kCoef_ = &k; }
   void Setup();

   BlockOperator * GetAOperator() { return A_; }
   BlockOperator * GetMOperator() { return M_; }

   // BlockDiagonalPreconditioner * GetBDP() { return BDP_; }
   Solver * GetPreconditioner() { return Precond_; }
   Operator * GetSubSpaceProjector() { return SubSpaceProj_; }
   // IterativeSolver * GetSolver() { return solver_; }

   ParFiniteElementSpace * GetFESpace() { return HCurlFESpace_; }

   void TestVector(HypreParVector & v);

private:

   H1_ParFESpace  * H1FESpace_;
   ND_ParFESpace  * HCurlFESpace_;
   RT_ParFESpace  * HDivFESpace_;

   double           alpha_a_;
   double           alpha_i_;
   double           beta_;
   // double           sigma_;
   Vector           zeta_;
   // DenseMatrix      zetaCross_;

   Coefficient    * mCoef_;
   Coefficient    * kCoef_;

   Array<int>       block_offsets_;
   Array<int>       block_trueOffsets_;
   Array<int>       block_trueOffsets2_;
   Array<HYPRE_Int> tdof_offsets_;

   BlockOperator  * A_;
   BlockOperator  * M_;

   HypreParMatrix * M1_;
   HypreParMatrix * M2_;
   HypreParMatrix * S1_;
   HypreParMatrix * T1_;

   HypreParMatrix * DKZ_;
   HypreParMatrix * DKZT_;

   HypreAMS                    * T1Inv_;
   // HypreSolver                    * T1Inv_;
   // HypreSolver                 * T1InvProj_;
   // Operator                 * T1InvProj_;

   // Operator * IrrProj_;
   // Operator * DirProj_;
   ParDiscreteInterpolationOperator * Curl_;
   ParDiscreteInterpolationOperator * Zeta_;

   BlockDiagonalPreconditioner * BDP_;

   Solver   * Precond_;
   Operator * SubSpaceProj_;

   HypreParVector * tmpVecA_;
   HypreParVector * tmpVecB_;

   // IterativeSolver * solver_;

   class VectorBlochWavePrecond : public Solver
   {
   public:
      VectorBlochWavePrecond(ParFiniteElementSpace & HCurlFESpace,
                             BlockDiagonalPreconditioner & BDP,
                             Operator & subSpaceProj,
                             //BlockOperator & LU,
                             double w)
         : Solver(2*HCurlFESpace.GlobalTrueVSize()),
           BDP_(&BDP), subSpaceProj_(&subSpaceProj), u_(NULL)
      {
         cout << "VectorBlochWavePrecond" << endl;

         MPI_Comm comm = HCurlFESpace.GetComm();
         int numProcs = HCurlFESpace.GetNRanks();

         int locSize = 2*HCurlFESpace.TrueVSize();
         int glbSize = 0;

         HYPRE_Int * part = NULL;

         if (HYPRE_AssumedPartitionCheck())
         {
            part = new HYPRE_Int[2];

            MPI_Scan(&locSize, &part[1], 1, HYPRE_MPI_INT, MPI_SUM, comm);

            part[0] = part[1] - locSize;

            MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm);
         }
         else
         {
            part = new HYPRE_Int[numProcs+1];

            MPI_Allgather(&locSize, 1, MPI_INT,
                          &part[1], 1, HYPRE_MPI_INT, comm);

            part[0] = 0;
            for (int i=0; i<numProcs; i++)
            {
               part[i+1] += part[i];
            }

            glbSize = part[numProcs];
         }

         r_ = new HypreParVector(comm,glbSize,part);
         u_ = new HypreParVector(comm,glbSize,part);
         v_ = new HypreParVector(comm,glbSize,part);
      }

      ~VectorBlochWavePrecond() { delete u_; }

      void Mult(const Vector & x, Vector & y) const
      {
         BDP_->Mult(x,*u_);
         /*
         *u_ = 0.0;

         for (int i=0; i<5; i++)
         {
           A_->Mult(y,*r_);
           *r_ *= -1.0;
           *r_ += x;

           BDP_->Mult(*r_,*v_);
           *v_ *= w_;
           *u_ += *v_;
         }
         */
         {
            // cout << "VectorBlochWavePrecond::Mult" << endl;
            // BDP_->Mult(x,*u_);
            // cout << "foo" << endl;
            subSpaceProj_->Mult(*u_,y);
            // cout << "Leaving VectorBlochWavePrecond::Mult" << endl;
         }
         //subSpaceProj_->Mult(x,y);
      }

      void SetOperator(const Operator & A) { A_ = &A; }

   private:
      BlockDiagonalPreconditioner * BDP_;
      const Operator * A_;
      Operator * subSpaceProj_;
      // Operator * LU_;
      mutable HypreParVector *r_, *u_, *v_;
      double w_;
   };
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

class ShiftedCurlIntegrator: public BilinearFormIntegrator
{
public:
   ShiftedCurlIntegrator(VectorCoefficient &vq,
                         const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), VQ(&vq) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

private:
   DenseMatrix vshape;
   DenseMatrix dshape;
   DenseMatrix dshapedxt;
   DenseMatrix invdfdx;
   Vector      D;
   VectorCoefficient *VQ;
};
*/
class VectorProjectionIntegrator: public BilinearFormIntegrator
{
public:
   VectorProjectionIntegrator(VectorCoefficient &vq,
                              const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), VQ(&vq) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

private:
   DenseMatrix vshape;
   Vector      D;
   VectorCoefficient *VQ;
};

// Material Coefficients
static int prob_ = -1;
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
   Vector zeta(3);
   // double sigma = 0.0;

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
   args.AddOption(&prob_, "-p", "--problem-type",
                  "Problem Geometry.");
   args.AddOption(&nev, "-nev", "--num_eigs",
                  "Number of eigenvalues requested.");
   args.AddOption(&beta, "-b", "--phase-shift",
                  "Phase Shift Magnitude in degrees");
   args.AddOption(&alpha_a, "-az", "--azimuth",
                  "Azimuth in degrees");
   args.AddOption(&alpha_i, "-inc", "--inclination",
                  "Inclination in degrees");
   // args.AddOption(&sigma, "-s", "--shift",
   //   "Shift parameter for eigenvalue solve");
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

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      //int offx = Ww+10, offy = Wh+45; // window offsets
      int offy = Wh+45; // window offsets

      VisualizeField(m_sock, vishost, visport, *m,
                     "Mass Coefficient", Wx, Wy, Ww, Wh);
      VisualizeField(k_sock, vishost, visport, *k,
                     "Stiffness Coefficient", Wx, Wy+offy, Ww, Wh);
   }

   GridFunctionCoefficient mCoef(m);
   GridFunctionCoefficient kCoef(k);
   /*
   VectorFloquetWaveEquation * eq =
      new VectorFloquetWaveEquation(*pmesh, order);
   */
   MaxwellBlochWaveEquation * eq =
      new MaxwellBlochWaveEquation(*pmesh, order);

   HYPRE_Int size = eq->GetHCurlFESpace()->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of complex unknowns: " << size << endl;
   }

   zeta[0] = cos(alpha_i*M_PI/180.0)*cos(alpha_a*M_PI/180.0);
   zeta[1] = cos(alpha_i*M_PI/180.0)*sin(alpha_a*M_PI/180.0);
   zeta[2] = sin(alpha_i*M_PI/180.0);

   eq->SetNumEigs(nev);
   eq->SetBeta(beta);
   eq->SetZeta(zeta);
   // eq->SetAzimuth(alpha_a);
   // eq->SetInclination(alpha_i);

   eq->SetMassCoef(mCoef);
   eq->SetStiffnessCoef(kCoef);

   // eq->SetSigma(sigma);

   eq->Setup();

   // 9. Define and configure the LOBPCG eigensolver and a BoomerAMG
   //    preconditioner to be used within the solver.
   /*
   HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);

   lobpcg->SetNumModes(nev);
   lobpcg->SetPreconditioner(*eq->GetPreconditioner());
   lobpcg->SetMaxIter(2000);
   lobpcg->SetTol(1e-7);
   lobpcg->SetPrecondUsageMode(1);
   lobpcg->SetPrintLevel(1);

   // Set the matrices which define the linear system
   lobpcg->SetMassMatrix(*eq->GetMOperator());
   lobpcg->SetOperator(*eq->GetAOperator());
   lobpcg->SetSubSpaceProjector(*eq->GetSubSpaceProjector());
   */
   // Obtain the eigenvalues and eigenvectors
   vector<double> eigenvalues;
   /*
   lobpcg->Solve();
   lobpcg->GetEigenvalues(eigenvalues);
   */
   eq->Solve();
   eq->GetEigenvalues(eigenvalues);

   if ( visit )
   {
      ParGridFunction Er(eq->GetHCurlFESpace());
      ParGridFunction Ei(eq->GetHCurlFESpace());

      ParGridFunction Br(eq->GetHDivFESpace());
      ParGridFunction Bi(eq->GetHDivFESpace());

      // int hcurl_loc_size = eq->GetFESpace()->TrueVSize();

      HypreParVector ErVec(eq->GetHCurlFESpace()->GetComm(),
                           eq->GetHCurlFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetHCurlFESpace()->GetTrueDofOffsets());
      HypreParVector EiVec(eq->GetHCurlFESpace()->GetComm(),
                           eq->GetHCurlFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetHCurlFESpace()->GetTrueDofOffsets());

      HypreParVector BrVec(eq->GetHDivFESpace()->GetComm(),
                           eq->GetHDivFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetHDivFESpace()->GetTrueDofOffsets());
      HypreParVector BiVec(eq->GetHDivFESpace()->GetComm(),
                           eq->GetHDivFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetHDivFESpace()->GetTrueDofOffsets());

      VisItDataCollection visit_dc("Vector3D-Parallel", pmesh);
      visit_dc.RegisterField("epsilon", m);
      visit_dc.RegisterField("muInv", k);
      visit_dc.RegisterField("E_r", &Er);
      visit_dc.RegisterField("E_i", &Ei);
      visit_dc.RegisterField("B_r", &Br);
      visit_dc.RegisterField("B_i", &Bi);

      for (int i=0; i<nev; i++)
      {
         //eq->TestVector(lobpcg->GetEigenvector(i));
         /*
               double * data = (double*)lobpcg->GetEigenvector(i);
               urVec.SetData(&data[0]);
               uiVec.SetData(&data[hcurl_loc_size]);
         */
         eq->GetEigenvector(i, ErVec, EiVec, BrVec, BiVec);

         Er = ErVec;
         Ei = EiVec;

         Br = BrVec;
         Bi = BiVec;

         visit_dc.SetCycle(i+1);
         visit_dc.SetTime(eigenvalues[i]);
         visit_dc.Save();
      }
   }

   if (visualization)
   {
      socketstream Er_sock, Ei_sock, Br_sock, Bi_sock, En_sock;
      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh+45; // window offsets

      /*
      ur_sock.open(vishost, visport);
      ui_sock.open(vishost, visport);
      ur_sock.precision(8);
      ui_sock.precision(8);
      */
      ParGridFunction Er(eq->GetHCurlFESpace());
      ParGridFunction Ei(eq->GetHCurlFESpace());

      ParGridFunction Br(eq->GetHDivFESpace());
      ParGridFunction Bi(eq->GetHDivFESpace());

      // int hcurl_loc_size = eq->GetFESpace()->TrueVSize();

      HypreParVector ErVec(eq->GetHCurlFESpace()->GetComm(),
                           eq->GetHCurlFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetHCurlFESpace()->GetTrueDofOffsets());
      HypreParVector EiVec(eq->GetHCurlFESpace()->GetComm(),
                           eq->GetHCurlFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetHCurlFESpace()->GetTrueDofOffsets());

      HypreParVector BrVec(eq->GetHDivFESpace()->GetComm(),
                           eq->GetHDivFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetHDivFESpace()->GetTrueDofOffsets());
      HypreParVector BiVec(eq->GetHDivFESpace()->GetComm(),
                           eq->GetHDivFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetHDivFESpace()->GetTrueDofOffsets());

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
         /*
              double * data = (double*)lobpcg->GetEigenvector(e);
              urVec.SetData(&data[0]);
              uiVec.SetData(&data[hcurl_loc_size]);
         */
         // eq->GetEigenvector(e, ErVec, EiVec, BrVec, BiVec);
         eq->GetEigenvectorE(e, ErVec, EiVec);

         if ( e == 0 )
         {
            ErVec.Print("Er.vec");
            EiVec.Print("Ei.vec");
         }

         Er = ErVec;
         Ei = EiVec;

         Br = BrVec;
         Bi = BiVec;

         VisualizeField(Er_sock, vishost, visport, Er,
                        "Re(E)", Wx+offx, Wy, Ww, Wh);
         VisualizeField(Ei_sock, vishost, visport, Ei,
                        "Im(E)", Wx+offx, Wy+offy, Ww, Wh);

         VisualizeField(Br_sock, vishost, visport, Br,
                        "Re(B)", Wx+2*offx, Wy, Ww, Wh);
         VisualizeField(Bi_sock, vishost, visport, Bi,
                        "Im(B)", Wx+2*offx, Wy+offy, Ww, Wh);
         VisualizeField(En_sock, vishost, visport,
                        *eq->GetEigenvectorEnergy(e),
                        "Energy", Wx+3*offx, Wy, Ww, Wh);

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
      Er_sock.close();
      Ei_sock.close();
      Br_sock.close();
      Bi_sock.close();
      En_sock.close();
   }
   /*
   // 9. Define and configure the LOBPCG eigensolver and a BoomerAMG
   //    preconditioner to be used within the solver.
   HypreAME    * ame = new HypreAME(MPI_COMM_WORLD);
   HypreSolver * ams = new HypreAMS(*eq->GetAOperator(),eq->GetFESpace(),1);

   ame->SetNumModes(nev);
   ame->SetPreconditioner(*ams);
   ame->SetMaxIter(100);
   ame->SetTol(1e-8);
   ame->SetPrintLevel(1);

   // Set the matrices which define the linear system
   ame->SetMassMatrix(*eq->GetMOperator());
   ame->SetOperator(*eq->GetAOperator());

   // Perform the eigensolve
   ame->Solve();

   // Obtain the eigenvalues and eigenvectors
   Array<double> eigenvalues;

   ame->GetEigenvalues(eigenvalues);
   */
   /*
   cout << "Creating v" << endl;
   int * col_part = eq->GetTrueDofOffsets();
   HypreParVector v(MPI_COMM_WORLD,col_part[num_procs],NULL,col_part);

   HypreMultiVector * eigenvectors = new HypreMultiVector(nev, v);

   int seed = 123;
   eigenvectors->Randomize(seed);

   // 10. Define and configure the ARPACK eigensolver and a GMRES
   //     solver to be used within the eigensolver.
   cout << "Creating arpack" << endl;
   ParArPackSym  * arpack = new ParArPackSym(MPI_COMM_WORLD);

   arpack->SetPrintLevel(0);
   arpack->SetMaxIter(400);
   arpack->SetTol(1e-4);
   arpack->SetShift(sigma);
   arpack->SetMode(3);

   cout << "getting operators" << endl;
   // arpack->SetA(*eq->GetAOperator());
   arpack->SetB(*eq->GetMOperator());
   arpack->SetSolver(*eq->GetSolver());

   // Obtain the eigenvalues and eigenvectors
   Vector eigenvalues(nev);
   eigenvalues = -1.0;

   cout << "getting eigenvalues" << endl;
   // arpack->Solve(eigenvalues);
   arpack->Solve(eigenvalues, *eigenvectors);

   eigenvalues.Print(cout);
   */
   /*
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

       double * data = hypre_VectorData(hypre_ParVectorLocalVector(
                           (hypre_ParVector*)eigenvectors->GetVector(e)));
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
   */
   delete eq;
   delete pmesh;

   MPI_Finalize();

   cout << "Exiting Main" << endl;

   return 0;
}

double mass_coef(const Vector & x)
{
   double eps1 = 10.0;
   double eps2 = 100.0;

   switch ( prob_ )
   {
      case 0:
         // Slab
         if ( fabs(x(0)) <= 0.5 ) { return eps1; }
         break;
      case 1:
         // Cylinder
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= 0.5 ) { return eps1; }
         break;
      case 2:
         // Sphere
         if ( x.Norml2() <= 0.5 ) { return eps1; }
         break;
      case 3:
         // Sphere and 3 Rods
         if ( x.Norml2() <= 0.5 ) { return eps1; }
         if ( sqrt(x(1)*x(1)+x(2)*x(2)) <= 0.2 ) { return eps1; }
         if ( sqrt(x(2)*x(2)+x(0)*x(0)) <= 0.2 ) { return eps1; }
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= 0.2 ) { return eps1; }
         break;
      case 4:
         // Sphere and 4 Rods
         if ( x.Norml2() <= 0.5 ) { return eps1; }
         break;
      case 5:
         // Two spheres in a BCC configuration
         if ( x.Norml2() <= 0.99 ) { return eps2; }
         {
            for (int i=0; i<8; i++)
            {
               int i1 = i%2;
               int i2 = (i/2)%2;
               int i4 = i/4;

               Vector u = x;
               u(0) -= i1?-1.0:1.0;
               u(1) -= i2?-1.0:1.0;
               u(2) -= i4?-1.0:1.0;

               if ( u.Norml2() <= 0.74 ) { return eps2; }
            }
         }
      default:
         return 1.0;
   }
   return 1.0;
}

double stiffness_coef(const Vector &x)
{
   return 1.0;
}

VectorFloquetWaveEquation::VectorFloquetWaveEquation(ParMesh & pmesh,
                                                     int order)
   : H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     alpha_a_(0.0),
     alpha_i_(90.0),
     beta_(0.0),
     // sigma_(0.0),
     mCoef_(NULL),
     kCoef_(NULL),
     A_(NULL),
     M_(NULL),
     M1_(NULL),
     M2_(NULL),
     S1_(NULL),
     T1_(NULL),
     DKZ_(NULL),
     DKZT_(NULL),
     T1Inv_(NULL),
     Curl_(NULL),
     Zeta_(NULL),
     BDP_(NULL)
     // solver_(NULL)
{
   int dim = pmesh.Dimension();

   zeta_.SetSize(dim);
   // zetaCross_.SetSize(dim);

   H1FESpace_    = new H1_ParFESpace(&pmesh,order,dim);
   HCurlFESpace_ = new ND_ParFESpace(&pmesh,order,dim);
   HDivFESpace_  = new RT_ParFESpace(&pmesh,order,dim);

   block_offsets_.SetSize(3);
   block_offsets_[0] = 0;
   block_offsets_[1] = HCurlFESpace_->GetVSize();
   block_offsets_[2] = HCurlFESpace_->GetVSize();
   block_offsets_.PartialSum();

   block_trueOffsets_.SetSize(3);
   block_trueOffsets_[0] = 0;
   block_trueOffsets_[1] = HCurlFESpace_->TrueVSize();
   block_trueOffsets_[2] = HCurlFESpace_->TrueVSize();
   block_trueOffsets_.PartialSum();

   block_trueOffsets2_.SetSize(3);
   block_trueOffsets2_[0] = 0;
   block_trueOffsets2_[1] = HDivFESpace_->TrueVSize();
   block_trueOffsets2_[2] = HDivFESpace_->TrueVSize();
   block_trueOffsets2_.PartialSum();

   tdof_offsets_.SetSize(HCurlFESpace_->GetNRanks()+1);
   HYPRE_Int * hcurl_tdof_offsets = HCurlFESpace_->GetTrueDofOffsets();
   for (int i=0; i<tdof_offsets_.Size(); i++)
   {
      tdof_offsets_[i] = 2 * hcurl_tdof_offsets[i];
   }

}

VectorFloquetWaveEquation::~VectorFloquetWaveEquation()
{
   delete A_;
   delete M_;
   delete BDP_;
   delete T1Inv_;

   // delete solver_;
   delete M1_;
   delete M2_;
   delete S1_;
   delete T1_;
   delete DKZ_;
   delete DKZT_;
   delete Curl_;
   delete Zeta_;
   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;
}

void
VectorFloquetWaveEquation::Setup()
{
   if ( zeta_.Size() == 3 )
   {
      zeta_[0] = cos(alpha_i_*M_PI/180.0)*cos(alpha_a_*M_PI/180.0);
      zeta_[1] = cos(alpha_i_*M_PI/180.0)*sin(alpha_a_*M_PI/180.0);
      zeta_[2] = sin(alpha_i_*M_PI/180.0);
      /*
      for (int i=0; i<3; i++)
      {
         for (int j=0; j<3; j++)
         {
            zetaCross_(i,j) = zeta_[i] * zeta_[j];
         }
         for (int j=0; j<3; j++)
         {
            zetaCross_(i,i) -= zeta_[j] * zeta_[j];
         }
      }
      */
   }
   else
   {
      zeta_[0] = cos(alpha_a_*M_PI/180.0);
      zeta_[1] = sin(alpha_a_*M_PI/180.0);
      /*
      for (int i=0; i<2; i++)
      {
         for (int j=0; j<2; j++)
         {
            zetaCross_(i,j) = zeta_[i] * zeta_[j];
         }
         for (int j=0; j<2; j++)
         {
            zetaCross_(i,i) -= zeta_[j] * zeta_[j];
         }
      }
      */
   }
   cout << "Phase Shift: " << beta_ << " (deg)"<< endl;
   cout << "Zeta:  ";
   zeta_.Print(cout);
   /*
   // VectorCoefficient *   zCoef = NULL;
   MatrixCoefficient * zxzCoef = NULL;
   // MatrixCoefficient * zozCoef = NULL;

   if ( kCoef_ )
   {
      // zCoef   = new VectorFunctionCoefficient(zeta_,*kCoef_);
      zxzCoef = new MatrixFunctionCoefficient(zetaCross_,*kCoef_);
   }
   else
   {
      // zCoef   = new VectorConstantCoefficient(zeta_);
      zxzCoef = new MatrixConstantCoefficient(zetaCross_);
   }
   */
   cout << "Building M2" << endl;
   ParBilinearForm m2(HDivFESpace_);
   m2.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoef_));
   m2.Assemble();
   m2.Finalize();
   M2_ = m2.ParallelAssemble();

   Zeta_ = new ParDiscreteVectorCrossProductOperator(HCurlFESpace_,
                                                     HDivFESpace_,zeta_);
   Curl_ = new ParDiscreteCurlOperator(HCurlFESpace_,HDivFESpace_);

   HypreParMatrix * CMC = RAP(M2_,Curl_->ParallelAssemble());
   HypreParMatrix * ZMZ = RAP(M2_,Zeta_->ParallelAssemble());

   HypreParMatrix * CMZ = RAP(Curl_->ParallelAssemble(),M2_,
                              Zeta_->ParallelAssemble());
   HypreParMatrix * ZMC = RAP(Zeta_->ParallelAssemble(),M2_,
                              Curl_->ParallelAssemble());

   *ZMC *= -1.0;
   DKZ_ = ParAdd(CMZ,ZMC);
   delete CMZ;
   delete ZMC;

   if ( fabs(beta_) > 0.0 )
   {
      *ZMZ *= beta_*beta_*M_PI*M_PI/32400.0;
      S1_ = ParAdd(CMC,ZMZ);
      delete CMC;
   }
   else
   {
      S1_ = CMC;
   }
   delete ZMZ;
   // S1_->Print("S1.mat");
   //delete zCoef;
   /*
   cout << "Building S1" << endl;
   LinearCombinationIntegrator * bfi = new LinearCombinationIntegrator();
   */
   /*
   if ( fabs(sigma_) > 0.0 )
   {
     bfi->AddIntegrator(-sigma_,
           new VectorFEMassIntegrator(*mCoef_));
   }
   */
   /*
   if ( fabs(beta_) > 0.0 )
   {
     bfi->AddIntegrator(-beta_*beta_*M_PI*M_PI/32400.0,
           new VectorFEMassIntegrator(*zxzCoef));
   }
   bfi->AddIntegrator(1.0, new CurlCurlIntegrator(*kCoef_));
   ParBilinearForm s1(HCurlFESpace_);
   s1.AddDomainIntegrator(bfi);
   s1.Assemble();
   s1.Finalize();
   S1_ = s1.ParallelAssemble();
   */
   /*
   cout << "Building T1" << endl;
   LinearCombinationIntegrator * bfiT = new LinearCombinationIntegrator();
   */
   /*
   if ( fabs(sigma_) > 0.0 )
   {
     bfiT->AddIntegrator(sigma_,
       new VectorFEMassIntegrator(*mCoef_));
   }
   */
   /*
   if ( fabs(beta_) > 0.0 )
   {
      bfiT->AddIntegrator(beta_*beta_*M_PI*M_PI/32400.0,
                          new VectorFEMassIntegrator(*zxzCoef));
   }
   bfiT->AddIntegrator(1.0, new CurlCurlIntegrator(*kCoef_));
   ParBilinearForm t1(HCurlFESpace_);
   t1.AddDomainIntegrator(bfiT);
   t1.Assemble();
   t1.Finalize();
   T1_ = t1.ParallelAssemble();
   */
   // delete zxzCoef;

   cout << "Building M1" << endl;
   ParBilinearForm m1(HCurlFESpace_);
   m1.AddDomainIntegrator(new VectorFEMassIntegrator(*mCoef_));
   m1.Assemble();
   m1.Finalize();
   M1_ = m1.ParallelAssemble();

   cout << "Building A" << endl;
   A_ = new BlockOperator(block_trueOffsets_);
   A_->SetDiagonalBlock(0,S1_);
   A_->SetDiagonalBlock(1,S1_);
   if ( fabs(beta_) > 0.0 )
   {
      A_->SetBlock(0,1,DKZ_,beta_*M_PI/180.0);
      A_->SetBlock(1,0,DKZ_,-beta_*M_PI/180.0);
   }
   A_->owns_blocks = 0;

   cout << "Building M" << endl;
   M_ = new BlockOperator(block_trueOffsets_);
   M_->SetDiagonalBlock(0,M1_);
   M_->SetDiagonalBlock(1,M1_);
   M_->owns_blocks = 0;

   // M1_->Print("M1.mat");
   // S1_->Print("S1.mat");
   // DKZ_->Print("DKZ.mat");

   cout << "Building T1Inv" << endl;
   // T1Inv_ = new HypreDiagScale(*S1_);
   if ( fabs(beta_) < 1.0 )
   {
      // T1_ = ParAdd(S1_,M1_);
      // T1Inv_ = new HypreAMS(*T1_,HCurlFESpace_);
      T1Inv_ = new HypreAMS(*S1_,HCurlFESpace_);
      // T1Inv_->SetSingularProblem();
   }
   else
   {
      T1Inv_ = new HypreAMS(*S1_,HCurlFESpace_);
      T1Inv_->SetSingularProblem();
   }

   SubSpaceProj_ = new VectorBlochWaveProjector(*HDivFESpace_,
                                                *HCurlFESpace_,
                                                *H1FESpace_,
                                                beta_,zeta_,
                                                mCoef_,kCoef_);
   /*
   {
      cout << "Building C" << endl;
      BlockOperator * C = new BlockOperator(block_trueOffsets2_,
                                            block_trueOffsets_);
      C->SetBlock(0,0,Curl_->ParallelAssemble());
      C->SetBlock(1,1,Curl_->ParallelAssemble());
      if ( fabs(beta_) > 0.0 )
      {
         C->SetBlock(0,1,Zeta_->ParallelAssemble(),beta_*M_PI/180.0);
         C->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/180.0);
      }
      C->owns_blocks = 0;

      cout << "Testing VectorBlochWaveProjector" << endl;
      MPI_Comm comm = HCurlFESpace_->GetComm();
      int numProcs = HCurlFESpace_->GetNRanks();

      int locSize = 2*HCurlFESpace_->TrueVSize();
      int locSize2 = 2*HDivFESpace_->TrueVSize();
      int glbSize = 0;
      int glbSize2 = 0;

      HYPRE_Int * part = NULL;
      HYPRE_Int * part2 = NULL;

      if (HYPRE_AssumedPartitionCheck())
      {
   cout << "HYPRE_AssumedPartitionCheck() is true" << endl;
   cout << "Local Sizes: " << locSize << " " << locSize2 << endl;
         part = new HYPRE_Int[2];
         part2 = new HYPRE_Int[2];

         MPI_Scan(&locSize, &part[1], 1, HYPRE_MPI_INT, MPI_SUM, comm);
         MPI_Scan(&locSize2, &part2[1], 1, HYPRE_MPI_INT, MPI_SUM, comm);

         part[0] = part[1] - locSize;
         part2[0] = part2[1] - locSize2;
    // part[1]++;
    // part2[1]++;

         MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm);
         MPI_Allreduce(&locSize2, &glbSize2, 1, HYPRE_MPI_INT, MPI_SUM, comm);
   cout << "Global Sizes: " << glbSize << " " << glbSize2 << endl;
      }
      else
      {
   cout << "HYPRE_AssumedPartitionCheck() is false" << endl;
         part = new HYPRE_Int[numProcs+1];
         part2 = new HYPRE_Int[numProcs+1];

         MPI_Allgather(&locSize, 1, MPI_INT,
                       &part[1], 1, HYPRE_MPI_INT, comm);
         MPI_Allgather(&locSize2, 1, MPI_INT,
                       &part2[1], 1, HYPRE_MPI_INT, comm);

         part[0] = 0;
         part2[0] = 0;
         for (int i=0; i<numProcs; i++)
         {
            part[i+1] += part[i];
            part2[i+1] += part2[i];
         }

         glbSize = part[numProcs];
         glbSize2 = part2[numProcs];
      }

      tmpVecA_ = new HypreParVector(comm,glbSize,part);
      tmpVecB_ = new HypreParVector(comm,glbSize,part);

      HypreParVector x(comm,glbSize,part);
      HypreParVector Cx(comm,glbSize2,part2);
      HypreParVector Ax(comm,glbSize,part);
      HypreParVector ATx(comm,glbSize,part);
      HypreParVector Px(comm,glbSize,part);
      HypreParVector CPx(comm,glbSize2,part2);
      HypreParVector APx(comm,glbSize,part);
      HypreParVector PAPx(comm,glbSize,part);

      x.Randomize(123);
      SubSpaceProj_->Mult(x,Px);
      A_->Mult(x,Ax);
      A_->MultTranspose(x,ATx);
      A_->Mult(Px,APx);
      SubSpaceProj_->Mult(APx,PAPx);
      C->Mult(x,Cx);
      C->Mult(Px,CPx);

      cout << "Norm x:    " << x.Norml2() << endl;
      cout << "Norm Px:   " << Px.Norml2() << endl;
      cout << "Norm Ax:   " << Ax.Norml2() << endl;
      cout << "Norm ATx:  " << ATx.Norml2() << endl;
      cout << "Norm APx:  " << APx.Norml2() << endl;
      cout << "Norm PAPx: " << PAPx.Norml2() << endl;
      cout << "Norm Cx:   " << Cx.Norml2() << endl;
      cout << "Norm CPx:  " << CPx.Norml2() << endl;
      ATx -= Ax;
      cout << "Norm ATx-Ax:  " << ATx.Norml2() << endl;

      delete C;
   }
   */

   BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets_);
   BDP_->SetDiagonalBlock(0,T1Inv_);
   BDP_->SetDiagonalBlock(1,T1Inv_);
   BDP_->owns_blocks = 0;

   Precond_ = new VectorBlochWavePrecond(*HCurlFESpace_,*BDP_,*SubSpaceProj_,0.5);
   Precond_->SetOperator(*A_);
   /*
   solver_ = new GMRESSolver(MPI_COMM_WORLD);
   solver_->SetPrintLevel(0);
   solver_->SetRelTol(1.0e-6);
   solver_->SetMaxIter(500);
   solver_->SetPreconditioner(*BDP_);
   solver_->SetOperator(*A_shift_);
   */

   cout << "Leaving Setup" << endl;
}

void VectorFloquetWaveEquation::TestVector(HypreParVector & v)
{
   SubSpaceProj_->Mult(v,*tmpVecA_);
   A_->Mult(*tmpVecA_,*tmpVecB_);

   cout << "========================" << endl;
   cout << "Norm v:    " << v.Norml2() << endl;
   cout << "Norm Pv:   " << tmpVecA_->Norml2() << endl;

   A_->Mult(v,*tmpVecA_);

   cout << "Norm Av:   " << tmpVecA_->Norml2() << endl;
   cout << "Norm APv:  " << tmpVecB_->Norml2() << endl;
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

void ShiftedCurlIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   double w;

   elmat.SetSize(nd);
   vshape.SetSize(nd,dim);
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
      el.CalcVShape(ip, vshape);
      el.CalcCurlShape(ip, dshape);

      Trans.SetIntPoint (&ip);

      w = ip.weight;

      MultABt(dshape, Trans.Jacobian(), dshapedxt);

      VQ -> Eval(D, Trans, ip);
      D *= w;

      for (int d = 0; d < dim; d++)
      {
         for (int j = 0; j < nd; j++)
         {
            for (int k = 0; k < nd; k++)
            {
               elmat(j, k) += dshapedxt(j,d) *
                              (D[(d+1)%3] * vshape(k,(d+2)%3) - D[(d+2)%3] * vshape(k,(d+1)%3))
                              - dshapedxt(k,d) *
                              (D[(d+1)%3] * vshape(j,(d+2)%3) - D[(d+2)%3] * vshape(j,(d+1)%3));
            }
         }
      }
   }
}
*/
/*
void VectorProjectionIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   double w;

   elmat.SetSize(nd);
   vshape.SetSize(nd,dim);
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
      el.CalcVShape(ip, vshape);

      Trans.SetIntPoint (&ip);

      w = ip.weight;

      VQ -> Eval(D, Trans, ip);
      D *= w;

      for (int d = 0; d < dim; d++)
      {
   for (int j = 0; j < nd; j++)
   {
     for (int k = 0; k < nd; k++)
     {
       elmat(j, k) += dshapedxt(j,d) *
         (D[(d+1)%3] * vshape(k,(d+2)%3) - D[(d+2)%3] * vshape(k,(d+1)%3))
         - dshapedxt(k,d) *
         (D[(d+1)%3] * vshape(j,(d+2)%3) - D[(d+2)%3] * vshape(j,(d+1)%3));
     }
   }
      }
   }
}
*/
/*
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
  if ( amg_  != NULL ) delete amg_;
  if ( S0_   != NULL ) delete S0_;
  if ( M1_   != NULL ) delete M1_;
  if ( Grad_ != NULL ) delete Grad_;
  if ( xDiv_ != NULL ) delete xDiv_;
  if ( yPot_ != NULL ) delete yPot_;
}

void
IrrotationalProjector::Mult(const Vector &x, Vector &y) const
{
  M1_->Mult(x,y);
  Grad_->MultTranspose(y,*xDiv_);
  amg_->Mult(*xDiv_,*yPot_);
  Grad_->Mult(*yPot_,y);
}
*/
/*
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
  delete pcg_;
  delete M1zoz_;
  delete xDual_;
}

void
DirectionalProjector::Mult(const Vector &x, Vector &y) const
{
  M1zoz_->Mult(x,*xDual_);
  pcg_->Mult(*xDual_,y);
}
*/

double phi(const Vector & x)
{
   return cos(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(2));
}

void Phi(const Vector & x, Vector & v)
{
   double a = 10.0*M_PI/180.0;
   double i = 15.0*M_PI/180.0;
   v(0) = cos(i)*cos(a); v(1) = cos(i)*sin(a); v(2) = sin(i);
   v *= phi(x);
}

VectorBlochWaveProjector::VectorBlochWaveProjector(
   ParFiniteElementSpace & HDivFESpace,
   ParFiniteElementSpace & HCurlFESpace,
   ParFiniteElementSpace & H1FESpace,
   double beta, const Vector & zeta,
   Coefficient * mCoef,
   Coefficient * kCoef)
   : Operator(2*HCurlFESpace.GlobalTrueVSize())/*,
    HCurlFESpace_(&HCurlFESpace),
    H1FESpace_(&H1FESpace)*/
{
   cout << "Constructing VectorBlochWaveProjector" << endl;

   block_offsets0_.SetSize(3);
   block_offsets0_[0] = 0;
   block_offsets0_[1] = H1FESpace.GetVSize();
   block_offsets0_[2] = H1FESpace.GetVSize();
   block_offsets0_.PartialSum();

   block_offsets1_.SetSize(3);
   block_offsets1_[0] = 0;
   block_offsets1_[1] = HCurlFESpace.GetVSize();
   block_offsets1_[2] = HCurlFESpace.GetVSize();
   block_offsets1_.PartialSum();

   block_trueOffsets0_.SetSize(3);
   block_trueOffsets0_[0] = 0;
   block_trueOffsets0_[1] = H1FESpace.TrueVSize();
   block_trueOffsets0_[2] = H1FESpace.TrueVSize();
   block_trueOffsets0_.PartialSum();

   block_trueOffsets1_.SetSize(3);
   block_trueOffsets1_[0] = 0;
   block_trueOffsets1_[1] = HCurlFESpace.TrueVSize();
   block_trueOffsets1_[2] = HCurlFESpace.TrueVSize();
   block_trueOffsets1_.PartialSum();


   locSize_ = HCurlFESpace.TrueVSize();

   u0_ = new BlockVector(block_trueOffsets0_);
   v0_ = new BlockVector(block_trueOffsets0_);
   u1_ = new BlockVector(block_trueOffsets1_);
   v1_ = new BlockVector(block_trueOffsets1_);

   cout << "Building M1" << endl;
   ParBilinearForm m1(&HCurlFESpace);
   m1.AddDomainIntegrator(new VectorFEMassIntegrator(mCoef));
   m1.Assemble();
   m1.Finalize();
   M1_ = m1.ParallelAssemble();

   cout << "Building M0" << endl;
   ParBilinearForm m0(&H1FESpace);
   m0.AddDomainIntegrator(new MassIntegrator(*mCoef));
   m0.Assemble();
   m0.Finalize();
   HypreParMatrix * M0 = m0.ParallelAssemble();

   cout << "Building M" << endl;
   M_ = new BlockOperator(block_trueOffsets1_);
   M_->SetDiagonalBlock(0,M1_);
   M_->SetDiagonalBlock(1,M1_,1.0);
   M_->owns_blocks = 0;

   Grad_ = new ParDiscreteGradOperator(&H1FESpace,&HCurlFESpace);
   Zeta_ = new ParDiscreteVectorProductOperator(&H1FESpace,&HCurlFESpace,zeta);
   ParDiscreteVectorCrossProductOperator Z12(&HCurlFESpace,&HDivFESpace,zeta);
   ParDiscreteCurlOperator T12(&HCurlFESpace,&HDivFESpace);

   cout << "Building G" << endl;
   G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
   G_->SetBlock(0,0,Grad_->ParallelAssemble());
   G_->SetBlock(1,1,Grad_->ParallelAssemble());
   if ( fabs(beta) > 0.0 )
   {
      G_->SetBlock(0,1,Zeta_->ParallelAssemble(),beta*M_PI/180.0);
      G_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta*M_PI/180.0);
   }
   G_->owns_blocks = 0;

   HypreParMatrix * GMG = RAP(M1_,Grad_->ParallelAssemble());
   HypreParMatrix * ZMZ = RAP(M1_,Zeta_->ParallelAssemble());

   HypreParMatrix * GMZ = RAP(Grad_->ParallelAssemble(),M1_,
                              Zeta_->ParallelAssemble());
   HypreParMatrix * ZMG = RAP(Zeta_->ParallelAssemble(),M1_,
                              Grad_->ParallelAssemble());
   *GMZ *= -1.0;
   DKZ_ = ParAdd(GMZ,ZMG);
   // DKZ_->Print("DKZ.mat");

   // delete GMG;
   delete GMZ;
   delete ZMG;
   /*
   Zeta_->ParallelAssemble()->Print("Z01.mat");
   Grad_->ParallelAssemble()->Print("T01.mat");
   Z12.ParallelAssemble()->Print("Z12.mat");
   T12.ParallelAssemble()->Print("T12.mat");
   */
   delete M0;
   // delete M2;

   if ( fabs(beta) > 0.0 )
   {
      *ZMZ *= beta*beta*M_PI*M_PI/32400.0;
      // ZMZ->Print("ZMZ_scaled.mat");
      //A0_ = ParAdd(S0,ZMZ);
      A0_ = ParAdd(GMG,ZMZ);
      // delete S0;
      delete GMG;
   }
   else
   {
      A0_ = GMG;
   }
   // A0_->Print("A0.mat");

   delete ZMZ;

   cout << "Building S0" << endl;
   S0_ = new BlockOperator(block_trueOffsets0_);
   cout << "Setting diag blocks" << endl;
   S0_->SetDiagonalBlock(0,A0_,1.0);
   S0_->SetDiagonalBlock(1,A0_,1.0);
   if ( fabs(beta) > 0.0 )
   {
      cout << "Setting offd blocks" << endl;
      S0_->SetBlock(0,1,DKZ_,-beta*M_PI/180.0);
      S0_->SetBlock(1,0,DKZ_,beta*M_PI/180.0);
   }
   S0_->owns_blocks = 0;

   cout << "Creating MINRES Solver" << endl;
   minres_ = new MINRESSolver(H1FESpace.GetComm());
   minres_->SetOperator(*S0_);
   minres_->SetRelTol(1e-13);
   minres_->SetMaxIter(3000);
   minres_->SetPrintLevel(0);
   cout << "done" << endl;

   cout << "Leaving VectorBlochWaveProjector c'tor" << endl;
}

void
VectorBlochWaveProjector::Mult(const Vector &x, Vector &y) const
{
   M_->Mult(x,y);
   G_->MultTranspose(y,*u0_);
   *v0_ = 0.0;
   minres_->Mult(*u0_,*v0_);
   G_->Mult(*v0_,y);
   y *= -1.0;
   y += x;
}
/*
ParDiscreteVectorProductOperator::ParDiscreteVectorProductOperator(
   ParFiniteElementSpace *dfes,
   ParFiniteElementSpace *rfes,
   const Vector & v)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new VectorProductInterpolator(v));
   pdlo_->Assemble();
   pdlo_->Finalize();
   mat_ = pdlo_->ParallelAssemble();
}

ParDiscreteVectorCrossProductOperator::ParDiscreteVectorCrossProductOperator(
   ParFiniteElementSpace *dfes,
   ParFiniteElementSpace *rfes,
   const Vector & v)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new VectorCrossProductInterpolator(v));
   pdlo_->Assemble();
   pdlo_->Finalize();
   mat_ = pdlo_->ParallelAssemble();
}
*/
void
AMSProj::Mult(const Vector &x, Vector &y) const
{
   cout << "AMSProj::Mult" << endl;
   /*
   ams_->Mult(x,y);
   irrProj_->Mult(y,*u_);
   dirProj_->Mult(*u_,*v_);
   y -= *v_;
   */
   /*
   ams_->Mult(x,y);
   irrProj_->Mult(y,*u_);
   y -= *u_;
   */
   y = x;
   irrProj_->Mult(y,*u_);
   y -= *u_;

}
/*
AMSProj::operator HYPRE_Solver() const
{}

HYPRE_PtrToParSolverFcn
AMSProj::SetupFcn() const
{}

HYPRE_PtrToParSolverFcn
AMSProj::SolveFcn() const
{}
*/
