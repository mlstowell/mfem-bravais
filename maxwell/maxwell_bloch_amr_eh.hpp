// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_MAXWELL_BLOCH_AMR
#define MFEM_MAXWELL_BLOCH_AMR

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mfem.hpp"
#include "../common/pfem_extras.hpp"
#include "../common/bravais.hpp"

namespace mfem
{

using miniapps::H1_ParFESpace;
using miniapps::ND_ParFESpace;
using miniapps::RT_ParFESpace;
using miniapps::L2_ParFESpace;
using miniapps::ParDiscreteGradOperator;
using miniapps::ParDiscreteCurlOperator;
using miniapps::ParDiscreteVectorProductOperator;
using miniapps::ParDiscreteVectorCrossProductOperator;
using bravais::BravaisLattice;
using bravais::HCurlFourierSeries;
//using bravais::RealPhaseCoefficient;
//using bravais::ImagPhaseCoefficient;

namespace bloch
{

// Physical Constants
// Permittivity of Free Space (units F/m)
//static double epsilon0_ = 8.8541878176e-12;
//static double epsilon0_ = 1.0;
#define MAXWELL_EPS0 8.8541878176e-12

// Permeability of Free Space (units H/m)
//static double mu0_ = 4.0e-7*M_PI;
//static double mu0_ = 1.0;
#define MAXWELL_MU0 4.0e-7*M_PI

class MaxwellBlochWaveProjectorAMR : public Operator
{
public:
   MaxwellBlochWaveProjectorAMR(ParFiniteElementSpace & HCurlFESpace,
                                ParFiniteElementSpace & H1FESpace,
                                BlockOperator & M,
                                double beta, const Vector & zeta,
                                int logging = 0);

   ~MaxwellBlochWaveProjectorAMR();

   void SetM(BlockOperator & M);
   void SetBeta(double beta);
   void SetZeta(const Vector & zeta);

   // To be called after M, Beta, or Zeta is changed
   void Setup();

   // To be called after mesh refinement
   void Update();

   virtual void Mult(const Vector &x, Vector &y) const;

   BlockOperator * GetAOperator() { return A_; }
   BlockOperator * GetMOperator() { return M_; }
   BlockOperator * GetGOperator() { return G_; }
   BlockVector   * GetBlockVector() { return u_; }

private:
   void OldSetup();
   void OldUpdate();

   void SetupSizes();
   void SetupTmpVectors();
   void SetupA0();
   void SetupB0();
   void SetupT01();
   void SetupZ01();
   void SetupBlockOperatorG();
   void SetupBlockOperatorA();
   void SetupSolver();

   // Updates the size of objects after mesh refinement
   /*
    void UpdateSizes();
    void UpdateTmpVectors();
    void UpdateA0();
    void UpdateB0();
    void UpdateT01();
    void UpdateZ01();
    void UpdateBlockOperatorG();
    void UpdateBlockOperatorA();
    void UpdateSolver();
   */
   // Recomputes G_, T01_, and Z01_ when beta, zeta, or the mesh changes
   // void UpdateG();

   // Recomputes S_ when beta, zeta, M_, or the mesh changes
   // void UpdateS();

   int myid_;
   int locSize_;
   int logging_;

   bool newSizes_;
   bool newM_;
   bool newBeta_;
   bool newZeta_;

   bool currSizes_;
   bool currVecs_;
   bool currA0_;
   bool currB0_;
   bool currT01_;
   bool currZ01_;
   bool currBOpA_;
   bool currBOpG_;

   ParFiniteElementSpace * HCurlFESpace_;
   ParFiniteElementSpace * H1FESpace_;

   double beta_;
   Vector zeta_;

   HypreParMatrix * T01_;
   HypreParMatrix * Z01_;
   HypreParMatrix * A0_;
   HypreParMatrix * B0_;
   // HypreParMatrix * DKZT_;
   HypreParMatrix * GMG_;
   HypreParMatrix * ZMZ_;
   HypreParMatrix * GMZ_;
   HypreParMatrix * ZMG_;

   ParDiscreteGradOperator          * t01_;
   ParDiscreteVectorProductOperator * z01_;

   Array<int>       block_trueOffsets0_;
   Array<int>       block_trueOffsets1_;

   BlockOperator * A_;
   BlockOperator * M_;
   BlockOperator * G_;

   // HypreBoomerAMG * amg_cos_;
   MINRESSolver   * AInv_;
   /*
   mutable HypreParVector * urDummy_;
   mutable HypreParVector * uiDummy_;
   mutable HypreParVector * vrDummy_;
   mutable HypreParVector * viDummy_;
   */
   mutable BlockVector * u_;
   mutable BlockVector * v_;
};
/*
class LinearCombinationOperator : public Operator
{
public:
   LinearCombinationOperator();
   ~LinearCombinationOperator();

   void AddTerm(double coef, Operator & op);

   void Mult(const Vector &x, Vector &y) const;

   int owns_terms;

private:

   std::vector<double>    coefs_;
   std::vector<Operator*> ops_;
   mutable Vector    u_;
};
*/
/** The folded spectrum operator is intended to help solve the
    generalized eigenvalue problem A x = lambda M x.  Eigenvalue
    solvers typically find the largest or smallest eigenvlaues, the
    folded spectrum technique is used to setup a related problem whose
    smallest eigenvalues correspond to eigenvalues from the interior
    of the original problem's spectrum.

    Consider the generalized eigenvlaue problem:
       A x = lambda M x
    Given a fixed lambda0 somewhere in the interior of A's spectrum:
       (A - lambda0 M) x = mu M x where mu = lambda - lambda0
    Form a standard eigenvlue problem:
       M^{-1} (A - lambda0 M) x = mu x
    Next we "square" the operator:
       M^{-1} (A - lambda0 M) M^{-1} (A - lambda0 M) x = mu^2 x
    Finally recast back to a generalized eigenvalue problem:
       (A - lambda0 M) M^{-1} (A - lambda0 M) x = mu^2 M x

    If A=A^T and M=M^T then this folded operator will be symmetric and
    positive semi-definite.  Also its smallest eigenvalues will be of
    the form (lambda - lambda0)^2 and clustered about lambda0.

    The LOBPCG algorithm should be well suited to compute these
    eigenvlues if we can find a reasonable preconditioner.  This
    algorithm will require solving M each time the operator is applied
    but, being a mass matrix, this should be reasonably quick.

    Regarding the preconditioner:
       [(A - lambda0 M) M^{-1} (A - lambda0 M)]^{-1}
          = (A - lambda0 M)^{-1} M (A - lambda0 M)^{-1}
 */
class FoldedSpectrumOperator : public Operator
{
public:

   FoldedSpectrumOperator(Operator & A, Operator & M, IterativeSolver & MInv,
                          double lambda0);

   ~FoldedSpectrumOperator() {}

   virtual void Mult(const Vector &x, Vector &y) const;

private:

   double lambda0_;

   Operator * A_;
   Operator * M_;
   IterativeSolver * MInv_;

   mutable Vector z0_;
   mutable Vector z1_;
};

class MaxwellBlochWaveEquationAMR
{
public:
   MaxwellBlochWaveEquationAMR(MPI_Comm & comm, Mesh & mesh, int order,
                               int ar, int logging = 0);

   ~MaxwellBlochWaveEquationAMR();

   HYPRE_Int *GetTrueDofOffsets() { return tdof_offsets_; }

   // void SetLatticeSize(double a) { a_ = a; }

   // Where kappa is the phase shift vector
   void SetKappa(const Vector & kappa);

   // Where beta*zeta = kappa
   void SetBeta(double beta);
   void SetZeta(const Vector & zeta);

   void SetAbsoluteTolerance(double atol);
   void SetNumEigs(int nev);
   void SetEpsilonCoef(Coefficient & eps);
   void SetMuCoef(Coefficient & mu);
   // void SetStiffnessCoef(Coefficient & a);
   void SetMaximumLightSpeed(double c);

   void Setup();

   void SetInitialVectors(int num_vecs, HypreParVector ** vecs);

   void SetBravaisLattice(BravaisLattice & bravais) { bravais_ = &bravais; }

   void Update();

   /// Solve the eigenproblem
   void Solve();

   /// Collect the converged eigenvalues
   void GetEigenvalues(std::vector<double> & eigenvalues);

   /// A convenience method which combines six methods into one
   void GetEigenvalues(/*int nev,*/ const Vector & kappa,
                                    // std::vector<HypreParVector*> & init_vecs,
                                    std::vector<double> & eigenvalues);

   void GetEigenvalues(/*int nev,*/ const Vector & kappa,
                                    // std::vector<HypreParVector*> & init_vecs,
                                    const std::set<int> & modes, double tol,
                                    std::vector<double> & eigenvalues);

   /// Extract a single eigenvector
   void GetEigenvector(unsigned int i,
                       HypreParVector & Er,
                       HypreParVector & Ei,
                       HypreParVector & Hr,
                       HypreParVector & Hi);
   void GetEigenvectorE(unsigned int i,
                        HypreParVector & Er,
                        HypreParVector & Ei);
   void GetEigenvectorH(unsigned int i,
                        HypreParVector & Hr,
                        HypreParVector & Hi);

   BlockOperator * GetAOperator() { return A_; }
   BlockOperator * GetMOperator() { return M_; }
   BlockOperator * GetCOperator() { return C_; }

   // Solver   * GetPreconditioner() { return Precond_; }
   // Operator * GetSubSpaceProjector() { return SubSpaceProj_; }

   ParFiniteElementSpace * GetHCurlFESpace() { return HCurlFESpace_; }
   ParFiniteElementSpace * GetHDivFESpace()  { return HDivFESpace_; }

   // ParGridFunction * GetEigenvectorEnergy(unsigned int i) { return energy_[i]; }

   void GetInnerProducts(DenseMatrix & mat);

   void GetFourierCoefficients(HypreParVector & Vr,
                               HypreParVector & Vi,
                               Array2D<double> &f);

   void GetFieldAverages(unsigned int i,
                         Vector & Er, Vector & Ei,
                         Vector & Br, Vector & Bi,
                         Vector & Dr, Vector & Di,
                         Vector & Hr, Vector & Hi);

   void WriteVisitFields(const std::string & prefix,
                         const std::string & label);

   void DisplayToGLVis(socketstream & a_sock, socketstream & m_sock,
                       char vishost[], int visport,
                       int Wx, int Wy, int Ww, int Wh, int offx, int offy);

private:
   int numModes(int lattice_type);

   // void OldSetup();

   void SetupSizes();
   void SetupTmpVectors();
   // void SetupA1();
   // void SetupB1();
   void SetupM1();
   void SetupM2();
   void SetupT12();
   void SetupD12();
   void SetupZ12();
   void SetupAMS();
   // void SetupPrecond();
   void SetupSolver();

   void SetupBlockOperatorC();
   void SetupBlockOperatorA();
   void SetupBlockOperatorM();
   void SetupBlockOperatorA12();
   void SetupBlockOperatorM12();
   // void SetupSubSpaceProjector();
   void SetupBlockDiagPrecond();
   // void SetupBlockPrecond();
   void SetupBlockSolver();

   void UpdateFES();
   void UpdateTmpVectors();

   MPI_Comm comm_;
   int myid_;
   int num_procs_;
   int order_;
   int ar_;
   int logging_;
   int hcurl_loc_size_;
   int hdiv_loc_size_;
   int nev_;

   int hcurl_glb_size_;
   int hdiv_glb_size_;
   HYPRE_Int * part_;

   bool newSizes_;
   bool newBeta_;
   bool newZeta_;
   // bool newOmega_;
   bool newACoef_;
   bool newMCoef_;

   bool currSizes_;
   bool currVecs_;
   bool currA1_;
   bool currB1_;
   bool currM1_;
   bool currM2_;
   bool currT12_;
   bool currD12_;
   bool currZ12_;
   bool currAMS_;
   bool currBOpA_;
   bool currBOpC_;
   bool currBOpM_;
   bool currBOpA12_;
   bool currBOpM12_;
   bool currBPC_; // Block Preconditioner
   bool currBDP_; // Block Diagonal Precondtioner
   bool currSSP_; // Sub-Space Projector

   ParMesh        * pmesh_;
   H1_ParFESpace  * H1FESpace_;
   ND_ParFESpace  * HCurlFESpace_;
   RT_ParFESpace  * HDivFESpace_;
   L2_ParFESpace  * L2FESpace_;

   BravaisLattice     * bravais_;
   HCurlFourierSeries * fourierHCurl_;

   double           atol_;
   double           beta_;
   double           omega_max_;
   double           omega_shift_;

   Vector           zeta_;
   Vector           kappa_;

   double epsAvg_;
   double muAvg_;

   Coefficient    * epsCoef_;
   Coefficient    * muCoef_;
   Coefficient    * epsInvCoef_;
   Coefficient    * muInvCoef_;
   ConstantCoefficient oneCoef_;
   VectorCoefficient * zCoef_;

   ParGridFunction * epsCoefGF_;
   ParGridFunction * muCoefGF_;

   ParLinearForm * oneLF_;

   // Array<int>       block_offsets_;
   Array<int>       block_trueOffsets1_;
   Array<int>       block_trueOffsets2_;
   Array<int>       block_trueOffsets12_;
   Array<HYPRE_Int> tdof_offsets_;

   Vector      lambdaB_;
   DenseMatrix vecB_;

   BlockOperator  * A_;
   BlockOperator  * M_;
   BlockOperator  * C_;

   BlockOperator  * A12_;
   BlockOperator  * B12_;
   BlockOperator  * M12_;

   BlockVector    * blkHCurl_;
   BlockVector    * blkHDiv_;

   HypreParMatrix * M1_;
   HypreParMatrix * M2_;
   HypreParMatrix * A1_;
   HypreParMatrix * T1_;
   HypreParMatrix * T1i_;
   HypreParMatrix * T2_;
   HypreParMatrix * T2i_;
   HypreParMatrix * T12_;
   HypreParMatrix * D12_;
   HypreParMatrix * D12T_;
   HypreParMatrix * Z12_;
   HypreParMatrix * Z12T_;

   HypreParMatrix * B1_;
   // HypreParMatrix * B1T_;

   HypreParMatrix * CMC_;
   HypreParMatrix * ZMZ_;
   HypreParMatrix * CMZ_;
   HypreParMatrix * ZMC_;

   HypreAMS * T1Inv_ams_;
   HypreAMS * T2Inv_ams_;
   // Solver   * T1Inv_minres_;

   ParDiscreteCurlOperator * t12_;
   // ParDiscreteVectorCrossProductOperator * z12_;
   ParBilinearForm      * m1_;
   ParBilinearForm      * m2_;
   ParBilinearForm      * t1_;
   ParBilinearForm      * t1i_;
   ParBilinearForm      * t2_;
   ParBilinearForm      * t2i_;
   ParMixedBilinearForm * d12_;
   ParMixedBilinearForm * z12_;

   BlockDiagonalPreconditioner * BDP_;

   // Solver   * Precond_;
   // MaxwellBlochWaveProjectorAMR * SubSpaceProj_;

   // FoldedSpectrumOperator * FSO_;
   // IterativeSolver        * FSOInv_;
   // IterativeSolver        * MInv_;
   // BlockDiagonalPreconditioner * MPC_;

   // HypreDiagScale * M1PC_;
   //  HypreDiagScale * M2PC_;
   //  HypreDiagScale * T2PC_;

   HypreParVector ** vecs_;
   HypreParVector * vec0_;

   int                num_init_vecs_;
   HypreParVector  ** init_vecs_;
   ParGridFunction ** init_er_;
   ParGridFunction ** init_ei_;
   ParGridFunction ** init_hr_;
   ParGridFunction ** init_hi_;

   HypreLOBPCG * lobpcg_;
   HypreAME    * ame_;

   // ParGridFunction ** energy_;

   class MaxwellBlochWavePrecond : public Solver
   {
   public:
      MaxwellBlochWavePrecond(ParFiniteElementSpace & HCurlFESpace,
                              BlockDiagonalPreconditioner & BDP,
                              Operator * subSpaceProj,
                              double w);

      ~MaxwellBlochWavePrecond();

      void Mult(const Vector & x, Vector & y) const;

      void SetOperator(const Operator & A);

      void Update();

   private:
      MPI_Comm    comm_;
      int         myid_;
      int         numProcs_;
      HYPRE_Int * part_;

      ParFiniteElementSpace * HCurlFESpace_;
      BlockDiagonalPreconditioner * BDP_;
      const Operator * A_;
      Operator * subSpaceProj_;
      mutable HypreParVector *u_;
   };
};
/*
void
ElementwiseEnergyNorm(BilinearFormIntegrator & bli,
                      ParGridFunction & x,
                      ParGridFunction & e);
*/
} // namespace bloch
} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_MAXWELL_BLOCH_AMR
