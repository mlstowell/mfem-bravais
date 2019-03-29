#include "mfem.hpp"
#include "pfem_extras.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <set>

//#include "temp_multivector.h"

using namespace std;
using namespace mfem;

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
   /*
   VectorBlochWaveProjector(HypreParMatrix & A,
          ParFiniteElementSpace & HCurlFESpace,
          Operator & irrProj, Operator & dirProj)
     : Operator(2*A.Width()), irrProj_(&irrProj), dirProj_(&dirProj)
   */
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
      // delete amg_cos_; //delete pcg_cos_;// delete amg_sin_;
      // delete S0_cos_; delete S0_sin_; delete M1_cos_; delete M1_sin_;
      delete Grad_;
   }

   virtual void Mult(const Vector &x, Vector &y) const;

private:
   int locSize_;

   // ParFiniteElementSpace * HCurlFESpace_;
   // ParFiniteElementSpace * H1FESpace_;
   /*
   HypreParMatrix * S0_cos_;
   HypreParMatrix * S0_sin_;
   HypreParMatrix * M1_cos_;
   HypreParMatrix * M1_sin_;
   */
   HypreParMatrix * Z_;
   HypreParMatrix * M1_;
   HypreParMatrix * A0_;
   HypreParMatrix * DKZ_;
   HypreParMatrix * DKZT_;

   HypreBoomerAMG * amg_cos_;
   // HyprePCG       * pcg_cos_;
   // HypreBoomerAMG * amg_sin_;
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

   // Operator * irrProj_;
   // Operator * dirProj_;
   mutable HypreParVector * urDummy_;
   mutable HypreParVector * uiDummy_;
   mutable HypreParVector * vrDummy_;
   mutable HypreParVector * viDummy_;
   // mutable HypreParVector * u0_;
   // mutable HypreParVector * v0_;
   mutable BlockVector * u0_;
   mutable BlockVector * v0_;
   mutable BlockVector * u1_;
   mutable BlockVector * v1_;

protected:
   /*
   class CosCoefficient : public FunctionCoefficient {
   public:
     CosCoefficient(double beta, const Vector & zeta);

     double Eval(ElementTransformation &T,
    const IntegrationPoint &ip);
   private:
     double beta_;
     const Vector & zeta_;
   };

   class SinCoefficient : public FunctionCoefficient {
   public:
     SinCoefficient(double beta, const Vector & zeta);

     double Eval(ElementTransformation &T,
    const IntegrationPoint &ip);
   private:
     double beta_;
     const Vector & zeta_;
   };
   */
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

class ParDiscreteVectorProductOperator
   : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteVectorProductOperator(ParFiniteElementSpace *dfes,
                                    ParFiniteElementSpace *rfes,
                                    const Vector & v);
   /*
    ParDiscreteVectorProductOperator(ParFiniteElementSpace *dfes,
              ParFiniteElementSpace *rfes,
              VectorCoefficient & v);
   */
};

class ParDiscreteVectorCrossProductOperator
   : public ParDiscreteInterpolationOperator
{
public:
   ParDiscreteVectorCrossProductOperator(ParFiniteElementSpace *dfes,
                                         ParFiniteElementSpace *rfes,
                                         const Vector & v);
};

/** Class for constructing the vector product as a DiscreteLinearOperator
    from an H1-conforming space to an H(curl)-conforming space. The range
    space can be vector L2 space as well. */
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
/*
class VectorProductInterpolator : public DiscreteInterpolator
{
public:
  VectorProductInterpolator(VectorCoefficient & v)
    : v_(v) {}
  // ~VectorProductInterpolator() { delete v_;}

   virtual void AssembleElementMatrix2(const FiniteElement &h1_fe,
                                       const FiniteElement &nd_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   {
     Vector nd_proj(nd_fe.GetDof());
     Vector h1_shp(h1_fe.GetDof());

     nd_fe.Project(v_,Trans,nd_proj);

     elmat.SetSize(nd_fe.GetDof(),h1_fe.GetDof());
     for (int k = 0; k < nd_fe.GetDof(); k++)
     {
       h1_fe.CalcShape(nd_fe.GetNodes().IntPoint(k),h1_shp);

       for (int j = 0; j < h1_fe.GetDof(); j++)
       {
    elmat(k,j) = nd_proj(k) * h1_shp(j);
       }
     }
   }
private:
  VectorCoefficient & v_;
};
*/

/** Class for constructing the vector cross product as a DiscreteLinearOperator
    from an H(curl)-conforming space to an H(div)-conforming space. The range
    space can be vector L2 space as well. */
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
   /*
   virtual operator HYPRE_Solver() const;
   virtual HYPRE_PtrToParSolverFcn SetupFcn() const;
   virtual HYPRE_PtrToParSolverFcn SolveFcn() const;
   */
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
   void SetSigma(double sigma) { sigma_ = sigma; }

   void SetMassCoef(Coefficient & m) { mCoef_ = &m; }
   void SetStiffnessCoef(Coefficient & k) { kCoef_ = &k; }
   void Setup();

   BlockOperator * GetAOperator() { return A_; }
   BlockOperator * GetMOperator() { return M_; }
   BlockOperator * GetShiftedOperator() { return A_shift_; }

   // BlockDiagonalPreconditioner * GetBDP() { return BDP_; }
   Solver * GetPreconditioner() { return Precond_; }
   Operator * GetSubSpaceProjector() { return SubSpaceProj_; }
   // IterativeSolver * GetSolver() { return solver_; }

   ParFiniteElementSpace * GetFESpace() { return HCurlFESpace_; }

private:

   H1_ParFESpace  * H1FESpace_;
   ND_ParFESpace  * HCurlFESpace_;
   RT_ParFESpace  * HDivFESpace_;

   double           alpha_a_;
   double           alpha_i_;
   double           beta_;
   double           sigma_;
   Vector           zeta_;
   DenseMatrix      zetaCross_;

   Coefficient    * mCoef_;
   Coefficient    * kCoef_;

   Array<int>       block_offsets_;
   Array<int>       block_trueOffsets_;
   Array<int>       block_trueOffsets2_;
   Array<HYPRE_Int> tdof_offsets_;

   BlockOperator  * A_;
   BlockOperator  * M_;
   BlockOperator  * A_shift_;

   HypreParMatrix * M1_;
   HypreParMatrix * M2_;
   HypreParMatrix * S1_;
   HypreParMatrix * T1_;

   HypreParMatrix * M1_scaled_;
   HypreParMatrix * S1_shift_;

   HypreParMatrix * DKZ_;
   HypreParMatrix * DKZT_;

   // HypreAMS                    * T1Inv_;
   HypreSolver                    * T1Inv_;
   // HypreSolver                 * T1InvProj_;
   // Operator                 * T1InvProj_;

   // Operator * IrrProj_;
   // Operator * DirProj_;
   ParDiscreteInterpolationOperator * Curl_;
   ParDiscreteInterpolationOperator * Zeta_;

   BlockDiagonalPreconditioner * BDP_;

   Solver   * Precond_;
   Operator * SubSpaceProj_;

   // IterativeSolver * solver_;

   class VectorBlochWavePrecond : public Solver
   {
   public:
      VectorBlochWavePrecond(ParFiniteElementSpace & HCurlFESpace,
                             BlockDiagonalPreconditioner & BDP,
                             Operator & subSpaceProj)
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
            part[1]++;

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

         u_ = new HypreParVector(comm,glbSize,part);
      }

      ~VectorBlochWavePrecond() { delete u_; }

      void Mult(const Vector & x, Vector & y) const
      {
         {
            // cout << "VectorBlochWavePrecond::Mult" << endl;
            BDP_->Mult(x,*u_);
            // cout << "foo" << endl;
            subSpaceProj_->Mult(*u_,y);
            // cout << "Leaving VectorBlochWavePrecond::Mult" << endl;
         }
         // subSpaceProj_->Mult(x,y);
      }

      void SetOperator(const Operator & op) {}

   private:
      BlockDiagonalPreconditioner * BDP_;
      Operator * subSpaceProj_;
      mutable HypreParVector * u_;
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
   double sigma = 0.0;

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
   args.AddOption(&sigma, "-s", "--shift",
                  "Shift parameter for eigenvalue solve");
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

   VectorFloquetWaveEquation * eq =
      new VectorFloquetWaveEquation(*pmesh, order);

   HYPRE_Int size = eq->GetFESpace()->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of complex unknowns: " << size << endl;
   }

   eq->SetBeta(beta);
   eq->SetAzimuth(alpha_a);
   eq->SetInclination(alpha_i);

   eq->SetMassCoef(mCoef);
   eq->SetStiffnessCoef(kCoef);

   eq->SetSigma(sigma);
   cout << "calling eq->setup" << endl;
   eq->Setup();
   cout << "done calling eq->setup" << endl;
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
   cout << "creating ARPACK solver" << endl;
   ParArPackSym * arpack = new ParArPackSym(MPI_COMM_WORLD);
   /*
   GMRESSolver * gmres = new GMRESSolver(MPI_COMM_WORLD);

   gmres->SetOperator(*eq->GetShiftedOperator());
   gmres->SetKDim(25);
   gmres->SetRelTol(1e-12);
   gmres->SetMaxIter(1000);
   gmres->SetPrintLevel(0);
   */
   MINRESSolver * minres = new MINRESSolver(MPI_COMM_WORLD);
   // minres->SetOperator(*eq->GetShiftedOperator());
   minres->SetOperator(*eq->GetMOperator());
   minres->SetRelTol(1e-13);
   minres->SetMaxIter(3000);
   minres->SetPrintLevel(0);

   arpack->SetPrintLevel(3);
   // arpack->SetMode(3);
   arpack->SetMode(2);
   arpack->SetNumModes(nev);
   // arpack->SetShift(sigma);
   arpack->SetMaxIter(500);
   arpack->SetTol(1e-6);

   arpack->SetOperator(*eq->GetAOperator());
   arpack->SetMassMatrix(*eq->GetMOperator());
   // arpack->SetSolver(*gmres);
   arpack->SetSolver(*minres);

   // Obtain the eigenvalues and eigenvectors
   Array<double> eigenvalues;

   cout << "calling arpack->Solve()" << endl;
   arpack->Solve();
   cout << "calling arpack->GetEigenvalues()" << endl;
   arpack->GetEigenvalues(eigenvalues);
   cout << "called arpack->GetEigenvalues()" << endl;

   if ( visit )
   {
      ParGridFunction ur(eq->GetFESpace());
      ParGridFunction ui(eq->GetFESpace());

      int hcurl_loc_size = eq->GetFESpace()->TrueVSize();

      HypreParVector urVec(eq->GetFESpace()->GetComm(),
                           eq->GetFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetFESpace()->GetTrueDofOffsets());
      HypreParVector uiVec(eq->GetFESpace()->GetComm(),
                           eq->GetFESpace()->GlobalTrueVSize(),
                           NULL,
                           eq->GetFESpace()->GetTrueDofOffsets());

      VisItDataCollection visit_dc("Vector3D-ArPack-Parallel", pmesh);
      visit_dc.RegisterField("epsilon", m);
      visit_dc.RegisterField("muInv", k);
      visit_dc.RegisterField("mode_r", &ur);
      visit_dc.RegisterField("mode_i", &ui);

      for (int i=0; i<nev; i++)
      {
         double * data = (double*)arpack->GetEigenvector(i);
         urVec.SetData(&data[0]);
         uiVec.SetData(&data[hcurl_loc_size]);

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

      int hcurl_loc_size = eq->GetFESpace()->TrueVSize();

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

         double * data = (double*)arpack->GetEigenvector(e);
         urVec.SetData(&data[0]);
         uiVec.SetData(&data[hcurl_loc_size]);

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
   delete arpack;
   // delete gmres;
   delete minres;
   delete eq;
   delete pmesh;

   MPI_Finalize();

   cout << "Exiting Main" << endl;

   return 0;
}

double mass_coef(const Vector & x)
{
   // if ( x.Norml2() <= 0.5 )
   /*
   if ( fabs(x(0)) <= 0.5 )
   {
     return 10.0;
   }
   else
   {
     return 1.0;
   }
   */
   return 1.0;
}

double stiffness_coef(const Vector &x)
{
   // if ( x.Norml2() <= 0.5 )
   /*
   if ( fabs(x(0)) <= 0.5 )
   {
     return 5.0;
   }
   else
   {
     return 0.1;
   }
   */
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
     // T1InvProj_(NULL),
     // IrrProj_(NULL),
     // DirProj_(NULL),
     BDP_(NULL)//,
     // solver_(NULL)
{
   int dim = pmesh.Dimension();

   zeta_.SetSize(dim);
   zetaCross_.SetSize(dim);

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
   // delete T1InvProj_;
   // delete IrrProj_;
   // delete DirProj_;

   // delete solver_;
   delete M1_;
   delete M2_;
   delete S1_;
   delete T1_;
   delete DKZ_;
   delete DKZT_;
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
   }
   else
   {
      zeta_[0] = cos(alpha_a_*M_PI/180.0);
      zeta_[1] = sin(alpha_a_*M_PI/180.0);

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
   }
   cout << "Phase Shift: " << beta_ << " (deg)"<< endl;
   cout << "Zeta:  ";
   zeta_.Print(cout);

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

   /*
   ParBilinearForm dkz(HCurlFESpace_);
   dkz.AddDomainIntegrator(new ShiftedCurlIntegrator(*zCoef));
   dkz.Assemble();
   dkz.Finalize();

   DKZ_  = dkz.ParallelAssemble();
   DKZT_ = DKZ_->Transpose();
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
   //DKZT_ = DKZ_->Transpose();
   delete CMZ;
   delete ZMC;
   //DKZ_->Print("DKZ.mat");
   //DKZT_->Print("DKZT.mat");

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
   cout << "Building T1" << endl;
   LinearCombinationIntegrator * bfiT = new LinearCombinationIntegrator();
   /*
   if ( fabs(sigma_) > 0.0 )
   {
     bfiT->AddIntegrator(sigma_,
       new VectorFEMassIntegrator(*mCoef_));
   }
   */
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

   delete zxzCoef;

   cout << "Building M1" << endl;
   ParBilinearForm m1(HCurlFESpace_);
   m1.AddDomainIntegrator(new VectorFEMassIntegrator(*mCoef_));
   m1.Assemble();
   m1.Finalize();
   M1_ = m1.ParallelAssemble();
   M1_scaled_ = m1.ParallelAssemble();
   if ( M1_ == M1_scaled_ ) { cout << "The matricies are the same!!!!!!" << endl; }
   *M1_scaled_ *= -sigma_;

   S1_shift_ = ParAdd(S1_,M1_scaled_);

   cout << "Building A" << endl;
   A_ = new BlockOperator(block_trueOffsets_);
   A_->SetDiagonalBlock(0,S1_);
   A_->SetDiagonalBlock(1,S1_);
   A_->SetBlock(0,1,DKZ_,beta_*M_PI/180.0);
   A_->SetBlock(1,0,DKZ_,-beta_*M_PI/180.0);
   A_->owns_blocks = 0;

   cout << "Building M" << endl;
   M_ = new BlockOperator(block_trueOffsets_);
   M_->SetDiagonalBlock(0,M1_);
   M_->SetDiagonalBlock(1,M1_);
   M_->owns_blocks = 0;

   cout << "Building Shifted A" << endl;
   A_shift_ = new BlockOperator(block_trueOffsets_);
   A_shift_->SetDiagonalBlock(0,S1_shift_);
   A_shift_->SetDiagonalBlock(1,S1_shift_);
   A_shift_->SetBlock(0,1,DKZ_,beta_*M_PI/180.0);
   A_shift_->SetBlock(1,0,DKZ_,-beta_*M_PI/180.0);
   A_shift_->owns_blocks = 0;

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

   M1_->Print("M1.mat");
   S1_->Print("S1.mat");
   S1_shift_->Print("S1_shift.mat");
   DKZ_->Print("DKZ.mat");

   cout << "Building T1Inv" << endl;
   T1Inv_ = new HypreDiagScale(*S1_);
   // T1Inv_ = new HypreAMS(*T1_,HCurlFESpace_);
   // T1Inv_->SetSingularProblem();
   /*
   cout << "Building irProj" << endl;
   IrrProj_ = new IrrotationalProjector(*HCurlFESpace_,*H1FESpace_);

   cout << "Building dProj" << endl;
   DirProj_ = new DirectionalProjector(*HCurlFESpace_,*M1_,zeta_,mCoef_);

   cout << "Building T1InvProj" << endl;
   // T1InvProj_ = new AMSProj(*T1Inv_,*IrrProj_,*DirProj_);
   // T1InvProj_ = new AMSProj(*T1_,*HCurlFESpace_,*IrrProj_,*DirProj_);
   */
   /*
   SubSpaceProj_ = new VectorBlochWaveProjector(*T1_,*HCurlFESpace_,
                    *IrrProj_,*DirProj_);
   */
   /*
   cout << "Building Subspace projector" << endl;
   SubSpaceProj_ = new VectorBlochWaveProjector(*HDivFESpace_,
                                                *HCurlFESpace_,
                                                *H1FESpace_,
                                                beta_,zeta_,
                                                mCoef_,kCoef_);

   {
     cout << "Testing Subspace projector" << endl;
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
         part = new HYPRE_Int[2];
         part2 = new HYPRE_Int[2];

         MPI_Scan(&locSize, &part[1], 1, HYPRE_MPI_INT, MPI_SUM, comm);
         MPI_Scan(&locSize2, &part2[1], 1, HYPRE_MPI_INT, MPI_SUM, comm);

         part[0] = part[1] - locSize;
         part2[0] = part2[1] - locSize2;
         part[1]++;
         part2[1]++;

         MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm);
         MPI_Allreduce(&locSize2, &glbSize2, 1, HYPRE_MPI_INT, MPI_SUM, comm);
      }
      else
      {
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

      HypreParVector x(comm,glbSize,part);
      HypreParVector Cx(comm,glbSize2,part2);
      HypreParVector Ax(comm,glbSize,part);
      HypreParVector ATx(comm,glbSize,part);
      HypreParVector Px(comm,glbSize,part);
      HypreParVector CPx(comm,glbSize2,part2);
      HypreParVector APx(comm,glbSize,part);
      HypreParVector PAPx(comm,glbSize,part);

      x.Randomize(123);
      cout << "SubSpaceProj->Mult" << endl;
      SubSpaceProj_->Mult(x,Px);
      cout << "A->Mult" << endl;
      A_->Mult(x,Ax);
      cout << "A->MultTranspose" << endl;
      A_->MultTranspose(x,ATx);
      cout << "A->Mult" << endl;
      A_->Mult(Px,APx);
      SubSpaceProj_->Mult(APx,PAPx);
      cout << "C->Mult" << endl;
      C->Mult(x,Cx);
      cout << "C->Mult" << endl;
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
   }
   */
   delete C;

   BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets_);
   BDP_->SetDiagonalBlock(0,T1Inv_);
   BDP_->SetDiagonalBlock(1,T1Inv_);
   BDP_->owns_blocks = 0;

   Precond_ = new VectorBlochWavePrecond(*HCurlFESpace_,*BDP_,*SubSpaceProj_);
   /*
   solver_ = new GMRESSolver(MPI_COMM_WORLD);
   solver_->SetPrintLevel(0);
   solver_->SetRelTol(1.0e-6);
   solver_->SetMaxIter(500);
   solver_->SetPreconditioner(*BDP_);
   solver_->SetOperator(*A_shift_);
   */
   /*
   ParGridFunction x1(HCurlFESpace_);
   x1.Randomize();

   HypreParVector * X1 = x1.ParallelAverage();
   HypreParVector * X1_ir = new HypreParVector(*T1_);
   HypreParVector * Y1_ir = new HypreParVector(*T1_);
   HypreParVector * X1_d = new HypreParVector(*T1_);
   HypreParVector * X1_dd = new HypreParVector(*T1_);
   HypreParVector * Y1_d = new HypreParVector(*T1_);

   cout << "Building irProj" << endl;
   Operator * irProj = new IrrotationalProjector(*HCurlFESpace_,*H1FESpace_);

   cout << "Building dProj" << endl;
   Operator *  dProj = new DirectionalProjector(*HCurlFESpace_,*M1_,zeta_,mCoef_);

   cout << "Calling irProj->Mult" << endl;
   irProj->Mult(*X1,*X1_ir);

   T1_->Mult(*X1_ir,*Y1_ir);

   cout << "Calling dProj->Mult" << endl;
   dProj->Mult(*X1,*X1_d);
   dProj->Mult(*X1_d,*X1_dd);

   DKZ_->Mult(*X1_d,*Y1_d);

   cout << "Norm of X1:             " << X1->Norml2() << endl;
   cout << "Norm of X1_ir:          " << X1_ir->Norml2() << endl;
   cout << "Norm of T1 IrProj X1:   " << Y1_ir->Norml2() << endl;
   cout << "Norm of X1_d:           " << X1_d->Norml2() << endl;
   cout << "Norm of X1_dd:          " << X1_dd->Norml2() << endl;
   cout << "Norm of S1 DirProj X1:  " << Y1_d->Norml2() << endl;

   delete irProj;
   delete dProj;

   delete X1;
   delete X1_ir;
   delete X1_d;
   delete Y1_ir;
   delete Y1_d;
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
/*
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

  return( cos(beta_ * (transip * zeta_) ) );
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

  return( sin(beta_ * (transip * zeta_) ) );
}
*/
/*
VectorBlochWaveProjector::BetaCoefficient::BetaCoefficient(double beta,
                        const Vector & zeta)
  : beta_(beta),
    zeta_(zeta)
{
  // cosCoef_ = new FunctionCoefficient(this->cosFunc_);
  // sinCoef_ = new FunctionCoefficient(sinFunc_);
}

VectorBlochWaveProjector::BetaCoefficient::~BetaCoefficient()
{
  delete cosCoef_;
  delete sinCoef_;
}
*/
/*
double
VectorBlochWaveProjector::BetaCoefficient::cosFunc_(const Vector & x)
{
  return cos(beta_ * ( x * zeta_ ) );
}

double
VectorBlochWaveProjector::BetaCoefficient::sinFunc_(const Vector & x)
{
  return sin(beta_ * ( x * zeta_ ) );
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
   /*
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
   */
   // u0_ = new HypreParVector(&H1FESpace);
   // v0_ = new HypreParVector(&H1FESpace);
   u0_ = new BlockVector(block_trueOffsets0_);
   v0_ = new BlockVector(block_trueOffsets0_);
   u1_ = new BlockVector(block_trueOffsets1_);
   v1_ = new BlockVector(block_trueOffsets1_);
   /*
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

   amg_cos_ = new HypreBoomerAMG(*S0_cos_);

   Grad_ = new ParDiscreteGradOperator(&H1FESpace,&HCurlFESpace);

   cout << "Building S0" << endl;
   S0_ = new BlockOperator(block_trueOffsets0_);
   S0_->SetDiagonalBlock(0,S0_cos_);
   S0_->SetDiagonalBlock(1,S0_cos_,-1.0);
   S0_->SetBlock(0,1,S0_sin_);
   S0_->SetBlock(1,0,S0_sin_);
   S0_->owns_blocks = 0;

   cout << "Building M1" << endl;
   M1_ = new BlockOperator(block_trueOffsets1_);
   M1_->SetDiagonalBlock(0,M1_cos_);
   M1_->SetDiagonalBlock(1,M1_cos_,-1.0);
   M1_->SetBlock(0,1,M1_sin_);
   M1_->SetBlock(1,0,M1_sin_);
   M1_->owns_blocks = 0;

   cout << "Building G" << endl;
   G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
   G_->SetBlock(0,0,Grad_->ParallelAssemble());
   G_->SetBlock(1,1,Grad_->ParallelAssemble());
   G_->owns_blocks = 0;
   */
   /*
   VectorCoefficient * zmCoef = NULL;
   VectorConstantCoefficient zCoef(zeta);

   if ( mCoef )
   {
     zmCoef = new VectorFunctionCoefficient(zeta,*mCoef);
   }
   else
   {
     zmCoef = new VectorConstantCoefficient(zeta);
   }
   */
   /*
   cout << "Building DKZ" << endl;
   ParBilinearForm dkz(&H1FESpace);
   dkz.AddDomainIntegrator(new ShiftedGradientIntegrator(*zmCoef));
   dkz.Assemble();
   dkz.Finalize();

   DKZ_  = dkz.ParallelAssemble();
   // DKZT_ = DKZ_->Transpose();

   delete zmCoef;
   */
   /*
   cout << "Building A0" << endl;
   LinearCombinationIntegrator * bfi = new LinearCombinationIntegrator();
   if ( fabs(beta) > 0.0 )
   {
     bfi->AddIntegrator(beta*beta*M_PI*M_PI/32400.0,
           new MassIntegrator(*mCoef));
   }
   bfi->AddIntegrator(1.0, new DiffusionIntegrator(*mCoef));
   ParBilinearForm a0(&H1FESpace);
   a0.AddDomainIntegrator(bfi);
   a0.Assemble();
   a0.Finalize();
   A0_ = a0.ParallelAssemble();
   */
   /*
   ParBilinearForm a0(&H1FESpace);
   a0.AddDomainIntegrator(new DiffusionIntegrator(*mCoef));
   a0.Assemble();
   a0.Finalize();
   A0_ = a0.ParallelAssemble();
   */
   /*
   cout << "Building S0" << endl;
   S0_ = new BlockOperator(block_trueOffsets0_);
   S0_->SetDiagonalBlock(0,A0_,-1.0);
   S0_->SetDiagonalBlock(1,A0_,1.0);
   S0_->SetBlock(0,1,DKZ_,-beta*M_PI/180.0);
   S0_->SetBlock(1,0,DKZ_,-beta*M_PI/180.0);
   S0_->owns_blocks = 0;
   */
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

   cout << "Building M2" << endl;
   ParBilinearForm m2(&HDivFESpace);
   m2.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoef));
   m2.Assemble();
   m2.Finalize();
   HypreParMatrix * M2 = m2.ParallelAssemble();

   cout << "Building S0" << endl;
   ParBilinearForm s0(&H1FESpace);
   s0.AddDomainIntegrator(new DiffusionIntegrator(*mCoef));
   s0.Assemble();
   s0.Finalize();
   HypreParMatrix * S0 = s0.ParallelAssemble();

   cout << "Building M" << endl;
   M_ = new BlockOperator(block_trueOffsets1_);
   M_->SetDiagonalBlock(0,M1_);
   M_->SetDiagonalBlock(1,M1_,-1.0);
   M_->owns_blocks = 0;

   Grad_ = new ParDiscreteGradOperator(&H1FESpace,&HCurlFESpace);
   // Zeta_ = new ParDiscreteVectorProductOperator(&H1FESpace,&HCurlFESpace,zCoef);
   Zeta_ = new ParDiscreteVectorProductOperator(&H1FESpace,&HCurlFESpace,zeta);
   ParDiscreteVectorCrossProductOperator Z12(&HCurlFESpace,&HDivFESpace,zeta);
   ParDiscreteCurlOperator T12(&HCurlFESpace,&HDivFESpace);
   /*
    {
      ParGridFunction x0(&H1FESpace);
      ParGridFunction x1(&HCurlFESpace);
      ParGridFunction Zx0(&HCurlFESpace);

      FunctionCoefficient phiCoef(phi);
      VectorFunctionCoefficient PhiCoef(3,Phi);

      x0.ProjectCoefficient(phiCoef);
      x1.ProjectCoefficient(PhiCoef);
      Zx0 = 0.0;

      HypreParVector * X0 = x0.ParallelAverage();
      HypreParVector * X1 = x1.ParallelAverage();
      HypreParVector * ZX0 = Zx0.ParallelAverage();

      Zeta_->Mult(*X0,*ZX0);
      Zx0 = *Zx0;

      double nx1 = X1->Norml2();
      cout << "Norm  X0:  " << X0->Norml2() << endl;
      cout << "Norm ZX0:  " << ZX0->Norml2() << endl;
      cout << "Norm  X1:  " << nx1 << endl;

      *X1 -= *ZX0;
      cout << "Norm (X1-ZX0)/X1:  " << X1->Norml2()/nx1 << endl;
    }
   */
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

   // HypreParMatrix * GT = Grad_->ParallelAssemble()->Transpose();
   // HypreParMatrix * ZT = Zeta_->ParallelAssemble()->Transpose();

   HypreParMatrix * GMG = RAP(M1_,Grad_->ParallelAssemble());
   HypreParMatrix * ZMZ = RAP(M1_,Zeta_->ParallelAssemble());

   HypreParMatrix * GMZ = RAP(Grad_->ParallelAssemble(),M1_,
                              Zeta_->ParallelAssemble());
   HypreParMatrix * ZMG = RAP(Zeta_->ParallelAssemble(),M1_,
                              Grad_->ParallelAssemble());
   //*GMZ *= -1.0;
   /*
   S0->Print("S0.mat");
   GMG->Print("GMG.mat");
   GMZ->Print("GMZ.mat");
   ZMG->Print("ZMG.mat");
   ZMZ->Print("ZMZ.mat");
   */
   DKZ_ = ParAdd(GMZ,ZMG);
   // DKZ_->Print("DKZ.mat");

   // delete GMG;
   delete GMZ;
   delete ZMG;
   // delete GT;
   // delete ZT;
   /*
   GMG->Print("GMG.mat");
   ZMZ->Print("ZMZ.mat");
   DKZ_->Print("DKZ.mat");
   S0->Print("S0.mat");
   M0->Print("M0.mat");
   M1_->Print("M1.mat");
   M2->Print("M2.mat");
   Zeta_->ParallelAssemble()->Print("Z01.mat");
   Grad_->ParallelAssemble()->Print("T01.mat");
   Z12.ParallelAssemble()->Print("Z12.mat");
   T12.ParallelAssemble()->Print("T12.mat");
   */
   delete M0;
   delete M2;

   if ( fabs(beta) > 0.0 )
   {
      *ZMZ *= -beta*beta*M_PI*M_PI/32400.0;
      // ZMZ->Print("ZMZ_scaled.mat");
      //A0_ = ParAdd(S0,ZMZ);
      A0_ = ParAdd(GMG,ZMZ);
      delete S0;
   }
   else
   {
      A0_ = S0;
   }
   A0_->Print("A0.mat");
   delete GMG;
   delete ZMZ;

   cout << "Building S0" << endl;
   S0_ = new BlockOperator(block_trueOffsets0_);
   cout << "Setting diag blocks" << endl;
   S0_->SetDiagonalBlock(0,A0_,1.0);
   S0_->SetDiagonalBlock(1,A0_,-1.0);
   if ( fabs(beta) > 0.0 )
   {
      cout << "Setting offd blocks" << endl;
      S0_->SetBlock(0,1,DKZ_,beta*M_PI/180.0);
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
   /*
   {
     u0_->Randomize(123);
     cout << "Norm Initial    u0 = " << u0_->Norml2() << endl;
     G_->Mult(*u0_,*u1_);
     S0_->Mult(*u0_,*v0_);
     cout << "Norm         S0*u0 = " << v0_->Norml2() << endl;
     M_->Mult(*u1_,*v1_);
     G_->MultTranspose(*v1_,*u0_);
     cout << "Norm    G^T*M*G*u0 = " << u0_->Norml2() << endl;
     ofstream ofsU("u0.vec");
     ofstream ofsV("v0.vec");
     u0_->Print(ofsU,1);
     v0_->Print(ofsV,1);
     ofsU.close();
     ofsV.close();
     *v0_ -= *u0_;
     cout << "Norm of difference:  " << v0_->Norml2() << endl;
   }
   */
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
   /*
   M_->Mult(y,*u1_);
   G_->MultTranspose(*u1_,*u0_);
   cout << "Norm     x = " << x.Norml2() << endl;
   cout << "Norm     y = " << y.Norml2() << endl;
   cout << "Norm Div y = " << u0_->Norml2() << endl;
   */
   /*
   *u0_ = 1.0;
   *v0_ = 0.0;
   minres_->Mult(*u0_,*v0_);
   */
   /*
   M1_->Mult(x,y);
   G_->MultTranspose(y,*u0_);
   *v0_ = 0.0;
   minres_->Mult(*u0_,*v0_);
   G_->Mult(*v0_,y);
   // y -= x;
   y *= -1.0;
   */
   /*
   // cout << "VectorBlochWaveProjector::Mult" << endl;
   double * data_X = (double*)x.GetData();
   double * data_Y = (double*)y;

   urDummy_->SetData(&data_X[0]);
   uiDummy_->SetData(&data_X[locSize_]);

   vrDummy_->SetData(&data_Y[0]);
   viDummy_->SetData(&data_Y[locSize_]);

   M1_cos_->Mult(*urDummy_,*vrDummy_);
   M1_sin_->Mult(*uiDummy_,*vrDummy_,1.0,1.0);
   Grad_->MultTranspose(*vrDummy_,*u0_);
   // amg_cos_->Mult(*u0_,*v0_);
   // pcg_cos_->Mult(*u0_,*v0_);

   *vrDummy_ = *urDummy_;

   Grad_->Mult(*v0_,*vrDummy_,-1.0,1.0);

   M1_cos_->Mult(*uiDummy_,*viDummy_);
   M1_sin_->Mult(*urDummy_,*viDummy_,-1.0,1.0);
   Grad_->MultTranspose(*viDummy_,*u0_);
   // amg_cos_->Mult(*u0_,*v0_);
   // pcg_cos_->Mult(*u0_,*v0_);

   *viDummy_ = *uiDummy_;

   Grad_->Mult(*v0_,*viDummy_,-1.0,1.0);
   */
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
/*
ParDiscreteVectorProductOperator::ParDiscreteVectorProductOperator(
                   ParFiniteElementSpace *dfes,
                                                 ParFiniteElementSpace *rfes,
                   VectorCoefficient & v)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new VectorProductInterpolator(v));
   pdlo_->Assemble();
   pdlo_->Finalize();
   mat_ = pdlo_->ParallelAssemble();
}
*/
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
