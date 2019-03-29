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

#ifndef MFEM_MAXWELL_BLOCH
#define MFEM_MAXWELL_BLOCH

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mfem.hpp"
#include "../common/pfem_extras.hpp"
//#include "bravais.hpp"

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

class MaxwellBlochWaveProjector : public Operator
{
public:
   MaxwellBlochWaveProjector(ParFiniteElementSpace & HDivFESpace,
                             ParFiniteElementSpace & HCurlFESpace,
                             ParFiniteElementSpace & H1FESpace,
                             HypreParMatrix & M1,
                             // BlockOperator & M,
                             double beta, const Vector & zeta);

   ~MaxwellBlochWaveProjector();

   void SetBeta(double beta);
   void SetZeta(const Vector & zeta);

   void Setup();

   void Update();

   virtual void Mult(const Vector &x, Vector &y) const;

private:
   int myid_;
   int locSize_;

   bool newBeta_;
   bool newZeta_;

   ParFiniteElementSpace * HDivFESpace_;
   ParFiniteElementSpace * HCurlFESpace_;
   ParFiniteElementSpace * H1FESpace_;

   double beta_;
   Vector zeta_;

   HypreParMatrix * Z_;
   HypreParMatrix * M1_;
   HypreParMatrix * A0_;
   //HypreParMatrix * DKZ_;
   // HypreParMatrix * DKZT_;

   HypreBoomerAMG * amg_cos_;
   MINRESSolver   * minres_;

   ParDiscreteGradOperator * Grad_;
   ParDiscreteVectorProductOperator * Zeta_;
   /*
    Array<int>       block_offsets0_;
    Array<int>       block_offsets1_;
    Array<int>       block_trueOffsets0_;
    Array<int>       block_trueOffsets1_;

    BlockOperator * S0_;
    BlockOperator * M_;
    BlockOperator * G_;
   */
   mutable HypreParVector * urDummy_;
   mutable HypreParVector * uiDummy_;
   mutable HypreParVector * vrDummy_;
   mutable HypreParVector * viDummy_;
   /*
    mutable BlockVector * u0_;
    mutable BlockVector * v0_;
    mutable BlockVector * u1_;
    mutable BlockVector * v1_;
   */
};

class LinearCombinationOperator : public Operator
{
public:
   LinearCombinationOperator();
   ~LinearCombinationOperator();

   void AddTerm(double coef, Operator & op);

   void Mult(const Vector &x, Vector &y) const;

   int owns_terms;

private:

   vector<double>    coefs_;
   vector<Operator*> ops_;
   mutable Vector    u_;
};

class MaxwellBlochWaveEquation
{
public:
   MaxwellBlochWaveEquation(ParMesh & pmesh, int order);

   ~MaxwellBlochWaveEquation();

   HYPRE_Int *GetTrueDofOffsets() { return tdof_offsets_; }

   // void SetLatticeSize(double a) { a_ = a; }

   // Where kappa is the phase shift vector
   void SetKappa(const Vector & kappa);

   // Where beta*zeta = kappa
   void SetBeta(double beta);
   void SetZeta(const Vector & zeta);

   // void SetAzimuth(double alpha_a);
   // void SetInclination(double alpha_i);

   // void SetOmega(double omega);
   void SetAbsoluteTolerance(double atol);
   void SetNumEigs(int nev);
   void SetMassCoef(Coefficient & m);
   void SetStiffnessCoef(Coefficient & k);

   void Setup();

   void SetInitialVectors(int num_vecs, HypreParVector ** vecs);

   void Update();

   /// Solve the eigenproblem
   void Solve();

   /// Collect the converged eigenvalues
   void GetEigenvalues(vector<double> & eigenvalues);

   /// A convenience method which combines six methods into one
   void GetEigenvalues(int nev, const Vector & kappa,
                       vector<HypreParVector*> & init_vecs,
                       vector<double> & eigenvalues);

   /// Extract a single eigenvector
   void GetEigenvector(unsigned int i,
                       HypreParVector & Er,
                       HypreParVector & Ei,
                       HypreParVector & Br,
                       HypreParVector & Bi);
   void GetEigenvectorE(unsigned int i,
                        HypreParVector & Er,
                        HypreParVector & Ei);
   void GetEigenvectorB(unsigned int i,
                        HypreParVector & Br,
                        HypreParVector & Bi);

   // BlockOperator * GetAOperator() { return A_; }
   // BlockOperator * GetMOperator() { return M_; }

   Solver   * GetPreconditioner() { return Precond_; }
   Operator * GetSubSpaceProjector() { return SubSpaceProj_; }

   ParFiniteElementSpace * GetHCurlFESpace() { return HCurlFESpace_; }
   ParFiniteElementSpace * GetHDivFESpace()  { return HDivFESpace_; }

   // void TestVector(const HypreParVector & v);

   ParGridFunction * GetEigenvectorEnergy(unsigned int i) { return energy_[i]; }

   void GetFourierCoefficients(HypreParVector & Vr,
                               HypreParVector & Vi,
                               Array2D<double> &f);

   void WriteVisitFields(const string & prefix, const string & label);

private:

   MPI_Comm comm_;
   int myid_;
   int hcurl_loc_size_;
   int hdiv_loc_size_;
   int nev_;

   // bool newAlpha_;
   bool newBeta_;
   bool newZeta_;
   bool newOmega_;
   bool newMCoef_;
   bool newKCoef_;

   ParMesh        * pmesh_;
   H1_ParFESpace  * H1FESpace_;
   ND_ParFESpace  * HCurlFESpace_;
   RT_ParFESpace  * HDivFESpace_;
   L2_ParFESpace  * L2FESpace_;

   // double           a_;
   // double           alpha_a_;
   // double           alpha_i_;
   double           atol_;
   double           beta_;
   double           omega_;
   Vector           zeta_;
   Vector           kappa_;

   Coefficient    * mCoef_;
   Coefficient    * kCoef_;
   // Coefficient   ** cCoef_;
   // Coefficient   ** sCoef_;
   /*
    RealPhaseCoefficient * cosCoef_;
    ImagPhaseCoefficient * sinCoef_;
   */
   /*
    VectorCoefficient  * xHatCoef0_;
    VectorCoefficient  * yHatCoef0_;
    VectorCoefficient  * zHatCoef0_;

    VectorCoefficient ** xHatCoefC_;
    VectorCoefficient ** yHatCoefC_;
    VectorCoefficient ** zHatCoefC_;

    VectorCoefficient ** xHatCoefS_;
    VectorCoefficient ** yHatCoefS_;
    VectorCoefficient ** zHatCoefS_;

    VectorCoefficient * xHatEpsCoef_;
    VectorCoefficient * yHatEpsCoef_;
    VectorCoefficient * zHatEpsCoef_;
   */
   /*
    Array<int>       block_offsets_;
    Array<int>       block_trueOffsets_;
    Array<int>       block_trueOffsets2_;
    Array<HYPRE_Int> tdof_offsets_;

    BlockOperator  * A_;
    BlockOperator  * M_;
    BlockOperator  * C_;

    BlockVector    * blkHCurl_;
    BlockVector    * blkHDiv_;
   */
   HypreParMatrix * M1_;
   HypreParMatrix * M2_;
   HypreParMatrix * S1_;
   HypreParMatrix * T1_;

   // HypreParMatrix * DKZ_;
   // HypreParMatrix * DKZT_;

   HypreAMS                    * T1Inv_;

   ParDiscreteCurlOperator * Curl_;
   ParDiscreteVectorCrossProductOperator * Zeta_;

   BlockDiagonalPreconditioner * BDP_;

   Solver   * Precond_;
   MaxwellBlochWaveProjector * SubSpaceProj_;

   // HypreParVector * tmpVecA_;
   // HypreParVector * tmpVecB_;
   HypreParVector ** vecs_;
   HypreParVector * vec0_;

   HypreLOBPCG * lobpcg_;
   HypreAME    * ame_;

   ParGridFunction ** energy_;

   LinearCombinationOperator * B_;
   MINRESSolver  * minres_;
   GMRESSolver   * gmres_;

   // ParGridFunction * cosKappaX_;
   // ParGridFunction * sinKappaX_;
   /*
    ParLinearForm  * jDualX0_;
    ParLinearForm  * jDualY0_;
    ParLinearForm  * jDualZ0_;

    ParLinearForm ** jDualXC_;
    ParLinearForm ** jDualYC_;
    ParLinearForm ** jDualZC_;

    ParLinearForm ** jDualXS_;
    ParLinearForm ** jDualYS_;
    ParLinearForm ** jDualZS_;

    ParLinearForm * jEpsDualX_;
    ParLinearForm * jEpsDualY_;
    ParLinearForm * jEpsDualZ_;
   */
   /*
    HypreParVector  * X0_;
    HypreParVector  * Y0_;
    HypreParVector  * Z0_;
    HypreParVector ** XC_;
    HypreParVector ** YC_;
    HypreParVector ** ZC_;
    HypreParVector ** XS_;
    HypreParVector ** YS_;
    HypreParVector ** ZS_;
   */
   class MaxwellBlochWavePrecond : public Solver
   {
   public:
      MaxwellBlochWavePrecond(ParFiniteElementSpace & HCurlFESpace,
                              // BlockDiagonalPreconditioner & BDP,
                              Operator & subSpaceProj,
                              double w);

      ~MaxwellBlochWavePrecond();

      void Mult(const Vector & x, Vector & y) const;

      void SetOperator(const Operator & A);

   private:
      int myid_;

      ParFiniteElementSpace * HCurlFESpace_;
      // BlockDiagonalPreconditioner * BDP_;
      const Operator * A_;
      Operator * subSpaceProj_;
      mutable HypreParVector *r_, *u_, *v_;
      double w_;
   };
};

void
ElementwiseEnergyNorm(BilinearFormIntegrator & bli,
                      ParGridFunction & x,
                      ParGridFunction & e);

} // namespace bloch
} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_MAXWELL_BLOCH
