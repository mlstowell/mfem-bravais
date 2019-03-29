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

#ifndef MFEM_MAXWELL_BLOCH_SHIFT
#define MFEM_MAXWELL_BLOCH_SHIFT

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

class MaxwellBlochWaveProjectorShift : public Operator
{
public:
   MaxwellBlochWaveProjectorShift(ParFiniteElementSpace & HCurlFESpace,
                                  ParFiniteElementSpace & H1FESpace,
                                  BlockOperator & M,
                                  double beta, const Vector & zeta);

   ~MaxwellBlochWaveProjectorShift();

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

   ParFiniteElementSpace * HCurlFESpace_;
   ParFiniteElementSpace * H1FESpace_;

   double beta_;
   Vector zeta_;

   HypreParMatrix * T01_;
   HypreParMatrix * Z01_;
   HypreParMatrix * A0_;
   HypreParMatrix * DKZ_;
   HypreParMatrix * DKZT_;

   HypreBoomerAMG * amg_cos_;
   MINRESSolver   * minres_;

   ParDiscreteGradOperator * Grad_;
   ParDiscreteVectorProductOperator * Zeta_;

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
class MaxwellBlochWaveEquationShift
{
public:
   MaxwellBlochWaveEquationShift(ParMesh & pmesh, int order,
                                 bool sns = false);

   ~MaxwellBlochWaveEquationShift();

   HYPRE_Int *GetTrueDofOffsets() { return tdof_offsets_; }

   // void SetLatticeSize(double a) { a_ = a; }

   // Where kappa is the phase shift vector
   void SetKappa(const Vector & kappa);

   // Where beta*zeta = kappa
   void SetBeta(double beta);
   void SetZeta(const Vector & zeta);

   void SetAbsoluteTolerance(double atol);
   void SetNumEigs(int nev);
   void SetMassCoef(Coefficient & m);
   void SetStiffnessCoef(Coefficient & k);
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
   void GetEigenvalues(int nev, const Vector & kappa,
                       std::vector<HypreParVector*> & init_vecs,
                       std::vector<double> & eigenvalues);

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

   BlockOperator * GetAOperator() { return A_; }
   BlockOperator * GetMOperator() { return M_; }

   Solver   * GetPreconditioner() { return Precond_; }
   Operator * GetSubSpaceProjector() { return SubSpaceProj_; }

   ParFiniteElementSpace * GetHCurlFESpace() { return HCurlFESpace_; }
   ParFiniteElementSpace * GetHDivFESpace()  { return HDivFESpace_; }

   ParGridFunction * GetEigenvectorEnergy(unsigned int i) { return energy_[i]; }

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

private:

   MPI_Comm comm_;
   int myid_;
   int hcurl_loc_size_;
   int hdiv_loc_size_;
   int nev_;

   bool newBeta_;
   bool newZeta_;
   bool newOmega_;
   bool newMCoef_;
   bool newKCoef_;
   bool shiftNullSpace_;

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

   Vector           zeta_;
   Vector           kappa_;

   Coefficient    * mCoef_;
   Coefficient    * kCoef_;

   Array<int>       block_offsets_;
   Array<int>       block_trueOffsets_;
   Array<int>       block_trueOffsets2_;
   Array<HYPRE_Int> tdof_offsets_;

   BlockOperator  * A_;
   BlockOperator  * M_;
   BlockOperator  * C_;

   BlockVector    * blkHCurl_;
   BlockVector    * blkHDiv_;

   HypreParMatrix * M0_;
   HypreParMatrix * M1_;
   HypreParMatrix * M2_;
   HypreParMatrix * S1_;
   HypreParMatrix * T1_;
   HypreParMatrix * T01_;
   HypreParMatrix * T01T_;
   HypreParMatrix * T12_;
   HypreParMatrix * Z01_;
   HypreParMatrix * Z01T_;
   HypreParMatrix * Z12_;

   HypreParMatrix * DKZ_;
   HypreParMatrix * DKZT_;

   HypreAMS * T1Inv_ams_;
   Solver   * T1Inv_minres_;

   ParDiscreteGradOperator * t01_;
   ParDiscreteCurlOperator * t12_;
   ParDiscreteVectorProductOperator      * z01_;
   ParDiscreteVectorCrossProductOperator * z12_;

   BlockDiagonalPreconditioner * BDP_;

   Solver   * Precond_;
   MaxwellBlochWaveProjectorShift * SubSpaceProj_;

   HypreParVector ** vecs_;
   HypreParVector * vec0_;

   HypreLOBPCG * lobpcg_;
   HypreAME    * ame_;

   ParGridFunction ** energy_;

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

   private:
      int myid_;

      BlockDiagonalPreconditioner * BDP_;
      const Operator * A_;
      Operator * subSpaceProj_;
      mutable HypreParVector *r_, *u_, *v_;
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

#endif // MFEM_MAXWELL_BLOCH_SHIFT
