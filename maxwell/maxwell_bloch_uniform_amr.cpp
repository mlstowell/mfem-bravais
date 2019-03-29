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

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "maxwell_bloch_uniform_amr.hpp"
#include <fstream>
/*
extern "C" {
#include "evsl.h"
#include "evsl_direct.h"
};
*/
using namespace std;

namespace mfem
{

using namespace miniapps;

namespace bloch
{

MaxwellBlochWaveEquationUniAMR::MaxwellBlochWaveEquationUniAMR(ParMesh & pmesh,
                                                   int order)
   : myid_(0),
     nev_(-1),
     // newAlpha_(true),
     newBeta_(true),
     newZeta_(true),
     newOmega_(true),
     newMCoef_(true),
     newKCoef_(true),
     pmesh_(&pmesh),
     H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     L2FESpace_(NULL),
     bravais_(NULL),
     fourierHCurl_(NULL),
     // alpha_a_(0.0),
     // alpha_i_(90.0),
     atol_(1.0e-6),
     beta_(0.0),
     // omega_(-1.0),
     mCoef_(NULL),
     kCoef_(NULL),
     /*     cosCoef_(NULL),
       sinCoef_(NULL),*/
     A_(NULL),
     M_(NULL),
     C_(NULL),
     blkHCurl_(NULL),
     blkHDiv_(NULL),
     M1_(NULL),
     M2_(NULL),
     S1_(NULL),
     T1_(NULL),
     T12_(NULL),
     Z12_(NULL),
     DKZ_(NULL),
     DKZT_(NULL),
     T1Inv_(NULL),
     Curl_(NULL),
     Zeta_(NULL),
     BDP_(NULL),
     Precond_(NULL),
     SubSpaceProj_(NULL),
     // tmpVecA_(NULL),
     // tmpVecB_(NULL),
     vecs_(NULL),
     vec0_(NULL),
     lobpcg_(NULL),
     ame_(NULL),
     energy_(NULL)/*,
     B_(NULL),
     minres_(NULL),
     gmres_(NULL),
     cosKappaX_(NULL),
     sinKappaX_(NULL)*/
{
   // Initialize MPI variables
   comm_ = pmesh.GetComm();
   MPI_Comm_rank(comm_, &myid_);

   if ( myid_ == 0 )
   {
      cout << "Constructing MaxwellBlochWaveEquationUniAMR" << endl;
   }

   int dim = pmesh.Dimension();

   zeta_.SetSize(dim);

   H1FESpace_    = new H1_ParFESpace(&pmesh,order,dim);
   HCurlFESpace_ = new ND_ParFESpace(&pmesh,order,dim);
   HDivFESpace_  = new RT_ParFESpace(&pmesh,order,dim);
   L2FESpace_    = new L2_ParFESpace(&pmesh,0,dim);

   hcurl_loc_size_ = HCurlFESpace_->TrueVSize();
   hdiv_loc_size_  = HDivFESpace_->TrueVSize();

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

   blkHCurl_ = new BlockVector(block_trueOffsets_);
   blkHDiv_  = new BlockVector(block_trueOffsets2_);
}

MaxwellBlochWaveEquationUniAMR::~MaxwellBlochWaveEquationUniAMR()
{
   delete lobpcg_;
   // delete minres_;
   // delete gmres_;
   // delete B_;

   if ( vecs_ != NULL )
   {
      for (int i=0; i<nev_; i++) { delete vecs_[i]; }
      delete [] vecs_;
   }

   // delete tmpVecA_;
   // delete tmpVecB_;

   delete SubSpaceProj_;
   delete Precond_;

   delete blkHCurl_;
   delete blkHDiv_;
   delete A_;
   delete M_;
   delete C_;
   delete BDP_;
   delete T1Inv_;

   delete M1_;
   delete M2_;
   delete S1_;
   delete T1_;
   delete T12_;
   delete Z12_;
   delete DKZ_;
   delete DKZT_;
   delete Curl_;
   delete Zeta_;

   delete fourierHCurl_;

   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;
   delete L2FESpace_;

   for (int i=0; i<3; i++)
   {
      delete AvgHCurl_coskx_[i];
      delete AvgHCurl_sinkx_[i];

      delete AvgHDiv_coskx_[i];
      delete AvgHDiv_sinkx_[i];

      delete AvgHCurl_eps_coskx_[i];
      delete AvgHCurl_eps_sinkx_[i];
   }
}

void
MaxwellBlochWaveEquationUniAMR::SetKappa(const Vector & kappa)
{
   kappa_ = kappa;
   beta_  = kappa.Norml2();  newBeta_ = true;
   zeta_  = kappa;           newZeta_ = true;
   if ( fabs(beta_) > 0.0 )
   {
      zeta_ /= beta_;
   }

   RealPhaseCoefficient rpc; rpc.SetKappa(kappa_);
   ImagPhaseCoefficient ipc; ipc.SetKappa(kappa_);

   ProductCoefficient eps_rpc(*mCoef_, rpc);
   ProductCoefficient eps_ipc(*mCoef_, ipc);

   ProductCoefficient muInv_rpc(*kCoef_, rpc);
   ProductCoefficient muInv_ipc(*kCoef_, ipc);

   for (int i=0; i<3; i++)
   {
      Vector v(3); v = 0.0; v[i] = 1.0;
      VectorConstantCoefficient vc(v);

      VectorFunctionCoefficient rpvc(v, rpc);
      VectorFunctionCoefficient ipvc(v, ipc);

      VectorFunctionCoefficient eps_rpvc(v, eps_rpc);
      VectorFunctionCoefficient eps_ipvc(v, eps_ipc);

      VectorFunctionCoefficient muInv_rpvc(v, muInv_rpc);
      VectorFunctionCoefficient muInv_ipvc(v, muInv_ipc);

      ParLinearForm avgHCurl_coskx(HCurlFESpace_);
      avgHCurl_coskx.AddDomainIntegrator(
         new VectorFEDomainLFIntegrator(rpvc));
      avgHCurl_coskx.Assemble();
      AvgHCurl_coskx_[i] = avgHCurl_coskx.ParallelAssemble();

      ParLinearForm avgHCurl_sinkx(HCurlFESpace_);
      avgHCurl_sinkx.AddDomainIntegrator(
         new VectorFEDomainLFIntegrator(ipvc));
      avgHCurl_sinkx.Assemble();
      AvgHCurl_sinkx_[i] = avgHCurl_sinkx.ParallelAssemble();

      ParLinearForm avgHCurl_eps_coskx(HCurlFESpace_);
      avgHCurl_eps_coskx.AddDomainIntegrator(
         new VectorFEDomainLFIntegrator(eps_rpvc));
      avgHCurl_eps_coskx.Assemble();
      AvgHCurl_eps_coskx_[i] = avgHCurl_eps_coskx.ParallelAssemble();

      ParLinearForm avgHCurl_eps_sinkx(HCurlFESpace_);
      avgHCurl_eps_sinkx.AddDomainIntegrator(
         new VectorFEDomainLFIntegrator(eps_ipvc));
      avgHCurl_eps_sinkx.Assemble();
      AvgHCurl_eps_sinkx_[i] = avgHCurl_eps_sinkx.ParallelAssemble();

      ParLinearForm avgHDiv_coskx(HDivFESpace_);
      avgHDiv_coskx.AddDomainIntegrator(new VectorFEDomainLFIntegrator(rpvc));
      avgHDiv_coskx.Assemble();
      AvgHDiv_coskx_[i] = avgHDiv_coskx.ParallelAssemble();

      ParLinearForm avgHDiv_sinkx(HDivFESpace_);
      avgHDiv_sinkx.AddDomainIntegrator(new VectorFEDomainLFIntegrator(ipvc));
      avgHDiv_sinkx.Assemble();
      AvgHDiv_sinkx_[i] = avgHDiv_sinkx.ParallelAssemble();

      ParLinearForm avgHDiv_muInv_coskx(HDivFESpace_);
      avgHDiv_muInv_coskx.AddDomainIntegrator(
         new VectorFEDomainLFIntegrator(muInv_rpvc));
      avgHDiv_muInv_coskx.Assemble();
      AvgHDiv_muInv_coskx_[i] = avgHDiv_muInv_coskx.ParallelAssemble();

      ParLinearForm avgHDiv_muInv_sinkx(HDivFESpace_);
      avgHDiv_muInv_sinkx.AddDomainIntegrator(
         new VectorFEDomainLFIntegrator(muInv_ipvc));
      avgHDiv_muInv_sinkx.Assemble();
      AvgHDiv_muInv_sinkx_[i] = avgHDiv_muInv_sinkx.ParallelAssemble();
   }
}

void
MaxwellBlochWaveEquationUniAMR::SetBeta(double beta)
{
   beta_ = beta; newBeta_ = true;
}

void
MaxwellBlochWaveEquationUniAMR::SetZeta(const Vector & zeta)
{
   zeta_ = zeta; newZeta_ = true;
}
/*
void
MaxwellBlochWaveEquationUniAMR::SetAzimuth(double alpha_a)
{
   alpha_a_ = alpha_a; newAlpha_ = true;
}

void
MaxwellBlochWaveEquationUniAMR::SetInclination(double alpha_i)
{
   alpha_i_ = alpha_i; newAlpha_ = true;
}
*/
/*
void
MaxwellBlochWaveEquationUniAMR::SetOmega(double omega)
{
   omega_ = omega; newOmega_ = true;
}
*/
void
MaxwellBlochWaveEquationUniAMR::SetAbsoluteTolerance(double atol)
{
   atol_ = atol;
}

void
MaxwellBlochWaveEquationUniAMR::SetNumEigs(int nev)
{
   nev_ = nev;
}

void
MaxwellBlochWaveEquationUniAMR::SetMassCoef(Coefficient & m)
{
   mCoef_ = &m; newMCoef_ = true;
}

void
MaxwellBlochWaveEquationUniAMR::SetStiffnessCoef(Coefficient & k)
{
   kCoef_ = &k; newKCoef_ = true;
}

void
MaxwellBlochWaveEquationUniAMR::Setup()
{
   /*
    if ( newAlpha_ )
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
    }
   */
   /*
    if ( myid_ == 0 )
    {
       cout << "Phase Shift: " << beta_*180.0/M_PI << " (deg)"<< endl;
       cout << "Zeta:  ";
       zeta_.Print(cout);
    }
   */
   if ( newKCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Building M2(k)" << endl; }
      ParBilinearForm m2(HDivFESpace_);
      m2.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoef_));
      m2.Assemble();
      m2.Finalize();
      delete M2_;
      M2_ = m2.ParallelAssemble();
   }

   if ( newZeta_ )
   {
      if ( myid_ == 0 ) { cout << "Building zeta cross operator" << endl; }
      delete Zeta_;
      Zeta_ = new ParDiscreteVectorCrossProductOperator(HCurlFESpace_,
                                                        HDivFESpace_,zeta_);
      Zeta_->Assemble();
      Zeta_->Finalize();
      Z12_ = Zeta_->ParallelAssemble();
   }

   if ( Curl_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Curl operator" << endl; }
      Curl_ = new ParDiscreteCurlOperator(HCurlFESpace_,HDivFESpace_);
      Curl_->Assemble();
      Curl_->Finalize();
      T12_ = Curl_->ParallelAssemble();
   }

   if ( newZeta_ || newBeta_ || newKCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Forming CMC" << endl; }
      HypreParMatrix * CMC = RAP(M2_,T12_);

      delete S1_;

      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
         HypreParMatrix * ZMZ = RAP(M2_, Z12_);
         HypreParMatrix * CMZ = RAP(T12_, M2_, Z12_);
         HypreParMatrix * ZMC = RAP(Z12_, M2_, T12_);

         *ZMC *= -1.0;
         delete DKZ_;
         DKZ_ = ParAdd(CMZ,ZMC);
         delete CMZ;
         delete ZMC;

         // *ZMZ *= beta_*beta_/(a_*a_);
         *ZMZ *= beta_*beta_;
         S1_ = ParAdd(CMC,ZMZ);
         delete CMC;
         delete ZMZ;
      }
      else
      {
         S1_ = CMC;
      }
   }

   if ( newMCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Building M1(m)" << endl; }
      ParBilinearForm m1(HCurlFESpace_);
      m1.AddDomainIntegrator(new VectorFEMassIntegrator(*mCoef_));
      m1.Assemble();
      m1.Finalize();
      delete M1_;
      M1_ = m1.ParallelAssemble();
   }

   if ( newZeta_ || newBeta_ || newKCoef_ )
   {
      if ( A_ == NULL )
      {
         if ( myid_ == 0 ) { cout << "Building Block A" << endl; }
         A_ = new BlockOperator(block_trueOffsets_);
      }
      A_->SetDiagonalBlock(0,S1_);
      A_->SetDiagonalBlock(1,S1_);
      if ( fabs(beta_) > 0.0 )
      {
         // A_->SetBlock(0,1,DKZ_, beta_*M_PI/(180.0*a_));
         // A_->SetBlock(1,0,DKZ_,-beta_*M_PI/(180.0*a_));
         // A_->SetBlock(0,1,DKZ_, beta_/a_);
         // A_->SetBlock(1,0,DKZ_,-beta_/a_);
         A_->SetBlock(0,1,DKZ_, beta_);
         A_->SetBlock(1,0,DKZ_,-beta_);
      }
      A_->owns_blocks = 0;
   }

   if ( newMCoef_ )
   {
      if ( M_ == NULL )
      {
         if ( myid_ == 0 ) { cout << "Building Block M" << endl; }
         M_ = new BlockOperator(block_trueOffsets_);
      }
      M_->SetDiagonalBlock(0,M1_);
      M_->SetDiagonalBlock(1,M1_);
      M_->owns_blocks = 0;
   }

   if ( newZeta_ || newBeta_ )
   {
      if ( C_ == NULL )
      {
         if ( myid_ == 0 ) { cout << "Building Block C" << endl; }
         C_ = new BlockOperator(block_trueOffsets2_, block_trueOffsets_);
      }
      C_->SetDiagonalBlock(0, T12_);
      C_->SetDiagonalBlock(1, T12_);
      if ( fabs(beta_) > 0.0 )
      {
         // C_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_*M_PI/(180.0*a_));
         // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/(180.0*a_));
         // C_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_/a_);
         // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_/a_);
         C_->SetBlock(0,1,Z12_, beta_);
         C_->SetBlock(1,0,Z12_,-beta_);
      }
      C_->owns_blocks = 0;
   }

   if ( newZeta_ || newBeta_ || newKCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Building T1Inv" << endl; }
      delete T1Inv_;
      if ( fabs(beta_*180.0) < M_PI )
      {
         cout << "HypreAMS::SetSingularProblem()" << endl;
         T1Inv_ = new HypreAMS(*S1_,HCurlFESpace_);
         T1Inv_->SetSingularProblem();
      }
      else
      {
         T1Inv_ = new HypreAMS(*S1_,HCurlFESpace_);
         // T1Inv_->SetSingularProblem();
      }

      if ( true || fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Building BDP" << endl; }
         delete BDP_;
         BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets_);
         BDP_->SetDiagonalBlock(0,T1Inv_);
         BDP_->SetDiagonalBlock(1,T1Inv_);
         BDP_->owns_blocks = 0;
      }
   }

   if ( ( newZeta_ || newBeta_ || newMCoef_ || newKCoef_ ) && nev_ > 0 )
   {
      if ( fabs(beta_) > 0.0 )
      {
         delete SubSpaceProj_;
         if ( myid_ == 0 ) { cout << "Building Subspace Projector" << endl; }
         SubSpaceProj_ = new MaxwellBlochWaveProjectorUniAMR(//*HDivFESpace_,
            *HCurlFESpace_,
            *H1FESpace_,
            *M_,beta_,zeta_);
         SubSpaceProj_->Setup();

         if ( myid_ == 0 ) { cout << "Building Preconditioner" << endl; }
         delete Precond_;
         Precond_ = new MaxwellBlochWavePrecond(*HCurlFESpace_,*BDP_,
                                                *SubSpaceProj_,0.5);
         Precond_->SetOperator(*A_);

         if ( myid_ == 0 ) { cout << "Building HypreLOBPCG solver" << endl; }
         delete lobpcg_;
         lobpcg_ = new HypreLOBPCG(comm_);

         lobpcg_->SetNumModes(nev_);
         lobpcg_->SetPreconditioner(*this->GetPreconditioner());
         lobpcg_->SetMaxIter(2000);
         lobpcg_->SetTol(atol_);
         lobpcg_->SetPrecondUsageMode(1);
         lobpcg_->SetPrintLevel(1);

         // Set the matrices which define the linear system
         lobpcg_->SetMassMatrix(*this->GetMOperator());
         lobpcg_->SetOperator(*this->GetAOperator());
         lobpcg_->SetSubSpaceProjector(*this->GetSubSpaceProjector());

         if ( false && vecs_ != NULL )
         {
            cout << "HypreLOBPCG::SetInitialVectors()" << endl;
            int n = 1 + (int)ceil(nev_/4);
            for (int i=nev_-n; i<nev_; i++) { vecs_[i]->Randomize(123); }
            lobpcg_->SetInitialVectors(nev_, vecs_);
         }
      }
      else
      {
         if ( myid_ == 0 ) { cout << "Building HypreAME solver" << endl; }
         delete ame_;
         ame_ = new HypreAME(comm_);
         ame_->SetNumModes(nev_/2);
         ame_->SetPreconditioner(*T1Inv_);
         ame_->SetMaxIter(2000);
         ame_->SetTol(atol_);
         ame_->SetRelTol(1e-8);
         ame_->SetPrintLevel(1);

         // Set the matrices which define the linear system
         ame_->SetMassMatrix(*M1_);
         ame_->SetOperator(*S1_);

         if ( vec0_ == NULL )
         {
            vec0_ = new HypreParVector(*M1_);
         }
         *vec0_ = 0.0;
      }
   }

   Vector xHat(3), yHat(3), zHat(3);
   xHat = yHat = zHat = 0.0;
   xHat(0) = 1.0; yHat(1) = 1.0; zHat(2) = 1.0;
   /*
   if ( omega_ >= 0.0 )
   {
      B_ = new LinearCombinationOperator();
      B_->AddTerm(omega_*omega_,*M_);
      B_->AddTerm(-1.0,*A_);
      B_->owns_terms = 0;

      if ( myid_ > 0 ) { cout << "Creating MINRES Solver" << endl; }
      delete minres_;
      minres_ = new MINRESSolver(comm_);
      minres_->SetOperator(*B_);
      minres_->SetRelTol(1e-6);
      minres_->SetMaxIter(3000);
      minres_->SetPrintLevel(2);

      if ( myid_ > 0 ) { cout << "Creating GMRES Solver" << endl; }
      delete gmres_;
      gmres_ = new GMRESSolver(comm_);
      gmres_->SetOperator(*B_);
      gmres_->SetRelTol(1e-6);
      gmres_->SetMaxIter(3000);
      gmres_->SetPrintLevel(1);
   }
   */
   newZeta_  = false;
   newBeta_  = false;
   newOmega_ = false;
   newMCoef_ = false;
   newKCoef_ = false;

   if ( myid_ == 0 ) { cout << "Leaving Setup" << endl; }
}

void
MaxwellBlochWaveEquationUniAMR::SetInitialVectors(int num_vecs,
                                            HypreParVector ** vecs)
{
   if ( lobpcg_ )
   {
      lobpcg_->SetInitialVectors(num_vecs, vecs);
   }
}

void MaxwellBlochWaveEquationUniAMR::Update()
{
   if ( myid_ == 0 ) { cout << "Building M2(k)" << endl; }
   ParBilinearForm m2(HDivFESpace_);
   m2.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoef_));
   m2.Assemble();
   m2.Finalize();
   delete M2_;
   M2_ = m2.ParallelAssemble();

   if ( Zeta_ )
   {
      Zeta_->Update();
      delete Z12_;
      Z12_ = Zeta_->ParallelAssemble();
   }
   if ( Curl_ )
   {
      Curl_->Update();
      delete T12_;
      T12_ = Curl_->ParallelAssemble();
   }

   if ( myid_ == 0 ) { cout << "Forming CMC" << endl; }
   HypreParMatrix * CMC = RAP(M2_, T12_);

   delete S1_;

   if ( fabs(beta_) > 0.0 )
   {
      if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
      HypreParMatrix * ZMZ = RAP(M2_,Z12_);

      HypreParMatrix * CMZ = RAP(T12_, M2_, Z12_);
      HypreParMatrix * ZMC = RAP(Z12_, M2_, T12_);

      *ZMC *= -1.0;
      delete DKZ_;
      DKZ_ = ParAdd(CMZ,ZMC);
      delete CMZ;
      delete ZMC;

      // *ZMZ *= beta_*beta_*M_PI*M_PI/(32400.0*a_*a_);
      // *ZMZ *= beta_*beta_/(a_*a_);
      *ZMZ *= beta_*beta_;
      S1_ = ParAdd(CMC,ZMZ);
      delete CMC;
      delete ZMZ;
   }
   else
   {
      S1_ = CMC;
   }

   if ( myid_ == 0 ) { cout << "Building M1(m)" << endl; }
   ParBilinearForm m1(HCurlFESpace_);
   m1.AddDomainIntegrator(new VectorFEMassIntegrator(*mCoef_));
   m1.Assemble();
   m1.Finalize();
   delete M1_;
   M1_ = m1.ParallelAssemble();

   if ( myid_ == 0 ) { cout << "Building Block A" << endl; }
   delete A_;
   A_ = new BlockOperator(block_trueOffsets_);
   A_->SetDiagonalBlock(0,S1_);
   A_->SetDiagonalBlock(1,S1_);
   if ( fabs(beta_) > 0.0 )
   {
      // A_->SetBlock(0,1,DKZ_,beta_*M_PI/(180.0*a_));
      // A_->SetBlock(1,0,DKZ_,-beta_*M_PI/(180.0*a_));
      // A_->SetBlock(0,1,DKZ_,beta_/a_);
      // A_->SetBlock(1,0,DKZ_,-beta_/a_);
      A_->SetBlock(0,1,DKZ_,beta_);
      A_->SetBlock(1,0,DKZ_,-beta_);
   }
   A_->owns_blocks = 0;

   if ( myid_ == 0 ) { cout << "Building Block M" << endl; }
   delete M_;
   M_ = new BlockOperator(block_trueOffsets_);
   M_->SetDiagonalBlock(0,M1_);
   M_->SetDiagonalBlock(1,M1_);
   M_->owns_blocks = 0;

   if ( myid_ == 0 ) { cout << "Building Block C" << endl; }
   delete C_;
   C_ = new BlockOperator(block_trueOffsets2_, block_trueOffsets_);
   C_->SetDiagonalBlock(0, T12_);
   C_->SetDiagonalBlock(1, T12_);
   if ( fabs(beta_) > 0.0 )
   {
      // C_->SetBlock(0,1,Zeta_->ParallelAssemble(),beta_*M_PI/(180.0*a_));
      // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/(180.0*a_));
      // C_->SetBlock(0,1,Zeta_->ParallelAssemble(),beta_/a_);
      // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_/a_);
      C_->SetBlock(0,1,Z12_,beta_);
      C_->SetBlock(1,0,Z12_,-beta_);
   }
   C_->owns_blocks = 0;

   if ( myid_ == 0 ) { cout << "Building T1Inv" << endl; }
   delete T1Inv_;
   if ( fabs(beta_) < 1.0 )
   {
      T1Inv_ = new HypreAMS(*S1_,HCurlFESpace_);
      T1Inv_->SetSingularProblem();
   }
   else
   {
      T1Inv_ = new HypreAMS(*S1_,HCurlFESpace_);
      T1Inv_->SetSingularProblem();
   }

   if ( myid_ == 0 ) { cout << "Building BDP" << endl; }
   delete BDP_;
   BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets_);
   BDP_->SetDiagonalBlock(0,T1Inv_);
   BDP_->SetDiagonalBlock(1,T1Inv_);
   BDP_->owns_blocks = 0;

   if ( SubSpaceProj_ ) { SubSpaceProj_->Update(); }

   if ( myid_ == 0 ) { cout << "Building Preconditioner" << endl; }
   delete Precond_;
   Precond_ = new MaxwellBlochWavePrecond(*HCurlFESpace_,*BDP_,*SubSpaceProj_,0.5);
   Precond_->SetOperator(*A_);

   if ( myid_ == 0 ) { cout << "Building HypreLOBPCG solver" << endl; }
   delete lobpcg_;
   lobpcg_ = new HypreLOBPCG(comm_);

   lobpcg_->SetNumModes(nev_);
   lobpcg_->SetPreconditioner(*this->GetPreconditioner());
   lobpcg_->SetMaxIter(2000);
   lobpcg_->SetTol(1e-6);
   lobpcg_->SetPrecondUsageMode(1);
   lobpcg_->SetPrintLevel(1);

   // Set the matrices which define the linear system
   lobpcg_->SetMassMatrix(*this->GetMOperator());
   lobpcg_->SetOperator(*this->GetAOperator());
   lobpcg_->SetSubSpaceProjector(*this->GetSubSpaceProjector());

   newZeta_  = false;
   newBeta_  = false;
   newMCoef_ = false;
   newKCoef_ = false;
}
/*
void MaxwellBlochWaveEquationUniAMR::TestVector(const HypreParVector & v)
{
   SubSpaceProj_->Mult(v,*tmpVecA_);
   A_->Mult(*tmpVecA_,*tmpVecB_);

   double normV  = v.Norml2();
   double normPV = tmpVecA_->Norml2();

   if ( myid_ == 0 )
   {
      cout << "========================" << endl;
      cout << "Norm v:    " << normV  << endl;
      cout << "Norm Pv:   " << normPV << endl;
   }

   A_->Mult(v,*tmpVecA_);

   double normAV  = tmpVecA_->Norml2();
   double normAPV = tmpVecB_->Norml2();

   if ( myid_ == 0 )
   {
      cout << "Norm Av:   " << normAV  << endl;
      cout << "Norm APv:  " << normAPV << endl;
   }
}
*/
void
MaxwellBlochWaveEquationUniAMR::Solve()
{
   if ( nev_ > 0 )
   {
      if ( fabs(beta_) > 0.0 )
      {
         lobpcg_->Solve();
         vecs_ = lobpcg_->StealEigenvectors();
         cout << "lobpcg done" << endl;
      }
      else
      {
         ame_->Solve();
         //vecs_ = ame_->StealEigenvectors();
         cout << "ame done" << endl;
      }
      /*
      CurlCurlIntegrator K(*kCoef_);

      energy_ = new ParGridFunction*[nev_];
      ParGridFunction er(HCurlFESpace_);
      ParGridFunction ei(HCurlFESpace_);
      ParGridFunction tmp(L2FESpace_);

      HypreParVector Er(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());
      HypreParVector Ei(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());
      */
      /*
      ofstream ofs("fourier.dat",ios::app);
      ofs << "Beta: " << beta_ << ", Zeta: ";
      zeta_.Print(ofs,3);
      for (int i=0; i<nev_; i++)
      {
         this->GetEigenvectorE(i,Er,Ei);
         er = Er; ei = Ei;
         energy_[i] = new ParGridFunction(L2FESpace_);
         ElementwiseEnergyNorm(K,er,*energy_[i]);
         ElementwiseEnergyNorm(K,ei,tmp);
         *energy_[i] += tmp;

         Array2D<double> fCoefs(27,6);
         this->GetFourierCoefficients(Er,Ei,fCoefs);
         fCoefs.Print(ofs,6);
      }
      ofs.close();
      */
   }
   /*
   if ( omega_ >= 0.0 )
   {
      BlockVector RHS(block_offsets_);
      RHS = 0.0;

      HypreParVector * EX = NULL;
      HypreParVector * EY = NULL;
      HypreParVector * EZ = NULL;
      if ( mCoef_ )
      {
         EX = jEpsDualX_->ParallelAssemble();
         EY = jEpsDualY_->ParallelAssemble();
         EZ = jEpsDualZ_->ParallelAssemble();
      }
      HypreParVector * T =
         new HypreParVector(comm_,
                            HCurlFESpace_->GlobalTrueVSize(),
                            NULL,
                            HCurlFESpace_->GetTrueDofOffsets());

      DenseMatrix Er(3), Ei(3), Dr(3), Di(3);
      DenseMatrix EMat(6);
      Vector DX(6), DY(6), DZ(6);
      cout << "blkHCurl.size = " << blkHCurl_->Size() << endl;

      ofstream ofsX("X.vec");
      ofstream ofsY("Y.vec");
      ofstream ofsZ("Z.vec");
      ofstream ofsRX("RX.vec");
      ofstream ofsRY("RY.vec");
      ofstream ofsRZ("RZ.vec");

      RHS.GetBlock(0) = *X0_;
      RHS.GetBlock(1) = 0.0;
      // RHS.Print(ofsRX,1);
      *blkHCurl_ = 0.0;
      //minres_->Mult(RHS,*blkHCurl_);
      gmres_->Mult(RHS,*blkHCurl_);
      cout << "GMRES Iterations: " << gmres_->GetNumIterations() << endl;
      //blkHCurl_->Print(ofsX,1);
      T->SetData(blkHCurl_->GetBlock(0));
      Er(0,0) = InnerProduct(*X0_,*T);
      Er(0,1) = InnerProduct(*Y0_,*T);
      Er(0,2) = InnerProduct(*Z0_,*T);
      if ( mCoef_ )
      {
         Dr(0,0) = InnerProduct(*EX,*T);
         Dr(0,1) = InnerProduct(*EY,*T);
         Dr(0,2) = InnerProduct(*EZ,*T);
      }
      T->SetData(blkHCurl_->GetBlock(1));
      Ei(0,0) = InnerProduct(*X0_,*T);
      Ei(0,1) = InnerProduct(*Y0_,*T);
      Ei(0,2) = InnerProduct(*Z0_,*T);
      if ( mCoef_ )
      {
         Di(0,0) = InnerProduct(*EX,*T);
         Di(0,1) = InnerProduct(*EY,*T);
         Di(0,2) = InnerProduct(*EZ,*T);
      }

      RHS.GetBlock(0) = *Y0_;
      RHS.GetBlock(1) = 0.0;
      // RHS.Print(ofsRY,1);
      *blkHCurl_ = 0.0;
      minres_->Mult(RHS,*blkHCurl_);
      // blkHCurl_->Print(ofsY,1);
      T->SetData(blkHCurl_->GetBlock(0));
      Er(1,0) = InnerProduct(*X0_,*T);
      Er(1,1) = InnerProduct(*Y0_,*T);
      Er(1,2) = InnerProduct(*Z0_,*T);
      if ( mCoef_ )
      {
         Dr(1,0) = InnerProduct(*EX,*T);
         Dr(1,1) = InnerProduct(*EY,*T);
         Dr(1,2) = InnerProduct(*EZ,*T);
      }
      T->SetData(blkHCurl_->GetBlock(1));
      Ei(1,0) = InnerProduct(*X0_,*T);
      Ei(1,1) = InnerProduct(*Y0_,*T);
      Ei(1,2) = InnerProduct(*Z0_,*T);
      if ( mCoef_ )
      {
         Di(1,0) = InnerProduct(*EX,*T);
         Di(1,1) = InnerProduct(*EY,*T);
         Di(1,2) = InnerProduct(*EZ,*T);
      }

      RHS.GetBlock(0) = *Z0_;
      RHS.GetBlock(1) = 0.0;
      // RHS.Print(ofsRZ,1);
      *blkHCurl_ = 0.0;
      minres_->Mult(RHS,*blkHCurl_);
      // blkHCurl_->Print(ofsZ,1);
      T->SetData(blkHCurl_->GetBlock(0));
      Er(2,0) = InnerProduct(*X0_,*T);
      Er(2,1) = InnerProduct(*Y0_,*T);
      Er(2,2) = InnerProduct(*Z0_,*T);
      if ( mCoef_ )
      {
         Dr(2,0) = InnerProduct(*EX,*T);
         Dr(2,1) = InnerProduct(*EY,*T);
         Dr(2,2) = InnerProduct(*EZ,*T);
      }
      T->SetData(blkHCurl_->GetBlock(1));
      Ei(2,0) = InnerProduct(*X0_,*T);
      Ei(2,1) = InnerProduct(*Y0_,*T);
      Ei(2,2) = InnerProduct(*Z0_,*T);
      if ( mCoef_ )
      {
         Di(2,0) = InnerProduct(*EX,*T);
         Di(2,1) = InnerProduct(*EY,*T);
         Di(2,2) = InnerProduct(*EZ,*T);
      }

      if ( ! mCoef_ )
      {
         for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
            {
              // Dr(i,j) = epsilon0_*Er(i,j);
              // Di(i,j) = epsilon0_*Ei(i,j);
               Dr(i,j) = MAXWELL_EPS0*Er(i,j);
               Di(i,j) = MAXWELL_EPS0*Ei(i,j);
            }
      }

      ofsX.close();
      ofsY.close();
      ofsZ.close();
      ofsRX.close();
      ofsRY.close();
      ofsRZ.close();

      cout << "Er:" << endl;
      Er.Print(cout,3);
      cout << "Ei:" << endl;
      Ei.Print(cout,3);
      cout << "Dr:" << endl;
      Dr.Print(cout,3);
      cout << "Di:" << endl;
      Di.Print(cout,3);

      cout << "|Er| = " << Er.Det() << endl;
      cout << "|Ei| = " << Ei.Det() << endl;

      for (int i=0; i<3; i++)
      {
         EMat(i,0) =  Er(i,0); EMat(i,1) =  Er(i,1); EMat(i,2) =  Er(i,2);
         EMat(i,3) = -Ei(i,0); EMat(i,4) = -Ei(i,1); EMat(i,5) = -Ei(i,2);

         EMat(i+3,0) =  Ei(i,0); EMat(i+3,1) =  Ei(i,1); EMat(i+3,2) =  Ei(i,2);
         EMat(i+3,3) =  Er(i,0); EMat(i+3,4) =  Er(i,1); EMat(i+3,5) =  Er(i,2);

         DX(i) = Dr(0,i);
         DX(i+3) = Di(0,i);

         DY(i) = Dr(1,i);
         DY(i+3) = Di(1,i);

         DZ(i) = Dr(2,i);
         DZ(i+3) = Di(2,i);
      }

      MatrixInverse * EMatInv = EMat.Inverse();
      Vector epsX(6), epsY(6), epsZ(6);
      EMatInv->Mult(DX,epsX);
      EMatInv->Mult(DY,epsY);
      EMatInv->Mult(DZ,epsZ);

      cout << "X column of eps: " << endl;
      epsX.Print(cout,1);
      cout << "Y column of eps: " << endl;
      epsY.Print(cout,1);
      cout << "Z column of eps: " << endl;
      epsZ.Print(cout,1);

      delete EMatInv;

      delete EX;
      delete EY;
      delete EZ;
      delete T;
   }
   */
   cout << "Solve done" << endl;
}

void
MaxwellBlochWaveEquationUniAMR::GetEigenvalues(vector<double> & eigenvalues)
{
   if ( lobpcg_ )
   {
      Array<double> eigs;
      lobpcg_->GetEigenvalues(eigs);
      eigenvalues.resize(eigs.Size());
      for (int i=0; i<eigs.Size(); i++)
      {
         eigenvalues[i] = eigs[i];
      }
   }
   else if ( ame_ )
   {
      Array<double> eigs0;
      ame_->GetEigenvalues(eigs0);
      eigenvalues.resize(2*eigs0.Size());
      for (int i=0; i<eigs0.Size(); i++)
      {
         eigenvalues[2*i+0] = eigs0[i];
         eigenvalues[2*i+1] = eigs0[i];
      }
   }
}

void
MaxwellBlochWaveEquationUniAMR::GetEigenvalues(int nev, const Vector & kappa,
                                         vector<HypreParVector*> & init_vecs,
                                         vector<double> & eigenvalues)
{
   this->SetNumEigs(nev);
   this->SetKappa(kappa);
   this->Setup();
   this->SetInitialVectors(nev, &init_vecs[0]);
   // double t0 = evsl_timer();

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   this->Solve();
   chrono.Stop();
   solve_times_.push_back(chrono.RealTime());
   // double t1 = evsl_timer();
   // cout << "Solve time: " << t1 - t0 << endl;
   this->GetEigenvalues(eigenvalues);
   // double t2 = evsl_timer();

   // cout << "GetEig time: " << t2 - t1 << endl;
   /*
   if ( lobpcg_)
      {
   SparseMatrix Ar, Ai, Mr;
   S1_->GetDiag(Ar);
   if (DKZ_) { DKZ_->GetDiag(Ai); Ai *= -beta_; }
   M1_->GetDiag(Mr);

   ComplexSparseMatrix Ac(&Ar, (DKZ_)?&Ai:NULL, false, false);
   ComplexSparseMatrix Mc(&Mr, NULL, false, false);

   SparseMatrix * Az = Ac.GetSystemMatrix();
   SparseMatrix * Mz = Mc.GetSystemMatrix();

   Ai *= -1.0/beta_;

   ofstream ofsA("A.mat");
   ofstream ofsM("M.mat");

   Az->PrintMM(ofsA);
   Mz->PrintMM(ofsM);
   csrMat Acsr;
   csrMat Bcsr;

   Acsr.owndata = 0;
   Acsr.nrows   = Az->Height();
   Acsr.ncols   = Az->Width();
   Acsr.ia      = Az->GetI();
   Acsr.ja      = Az->GetJ();
   Acsr.a       = Az->GetData();

   Bcsr.owndata = 0;
   Bcsr.nrows   = Mz->Height();
   Bcsr.ncols   = Mz->Width();
   Bcsr.ia      = Mz->GetI();
   Bcsr.ja      = Mz->GetJ();
   Bcsr.a       = Mz->GetData();
   */
   /*
   void * Bsol = NULL;

   SetupBSolDirect(&Mcsr, &Bsol);
   SetBSol(BSolDirect, Bsol);
   SetLTSol(LTSolDirect, Bsol);

   SetAMatrix(&Acsr);
   SetBMatrix(&Mcsr);
   SetGenEig();
   */
   /*
   {
   double t3 = evsl_timer();

   int n, i, j, npts, nslices, nvec, nev, mlan, ev_int, sl, ierr, totcnt;
   */
   /* find the eigenvalues of A in the interval [a,b] */
   /*
   double a, b, lmax, lmin, ecount, tol, *sli;
   double xintv[4];
   double *alleigs;
   int *counts; // #ev computed in each slice
   */
   /* initial vector: random */
   /*
   double *vinit;
   polparams pol;

   a = 0.01;
   b = 1.01 * eigenvalues[eigenvalues.size()-1];

   n = Acsr.nrows;
   nslices = 1;

   counts = (int*)malloc(nslices * sizeof(int));
   sli = (double*)malloc((nslices + 1) * sizeof(double));

   tol = 1e-6;
   int msteps = 40;
   nvec = 10;
   npts = 200;
   void * Bsol = NULL;

   FILE *fstats = stdout;

   alleigs = (double*)malloc(n * sizeof(double));
   */
   /*-------------------- use direct solver as the solver for B */
   // SetupBSolDirect(&Bcsr, &Bsol);
   /*-------------------- set the solver for B and LT */
   // SetBSol(BSolDirect, Bsol);
   // SetLTSol(LTSolDirect, Bsol);
   /*-------------------- set the left-hand side matrix A */
   // SetAMatrix(&Acsr);
   /*-------------------- set the right-hand side matrix B */
   // SetBMatrix(&Bcsr);
   /*-------------------- for generalized eigenvalue problem */
   // SetGenEig();
   /*-------------------- step 0: get eigenvalue bounds */
   //-------------------- initial vector
   /*
    vinit = (double *)malloc(n * sizeof(double));
    rand_double(n, vinit);
    // lmin = 0.0;
    // lmax = 1e6;

    ierr = LanTrbounds(50, 200, 1e-12, vinit, 1, &lmin, &lmax, fstats);
    printf("Step 0: Eigenvalue bound s for B^{-1}*A: [%.15e, %.15e]\n",
            lmin, lmax);
   */
   /*-------------------- interval and eig bounds */
   /*
    xintv[0] = a;
    xintv[1] = b;
    xintv[2] = lmin;
    xintv[3] = lmax;
   */
   /*-------------------- call LanczosDOS for spectrum slicing */
   /*-------------------- define landos parameters */
   /*
    double t = evsl_timer();
    double *xdos = (double *)calloc(npts, sizeof(double));
    double *ydos = (double *)calloc(npts, sizeof(double));
    ierr = LanDosG(nvec, msteps, npts, xdos, ydos, &ecount, xintv);
    t = evsl_timer() - t;
    if (ierr) {
      printf("Landos error %d\n", ierr);
      MFEM_ASSERT(false,"Landos error");
    }
    printf(" Time to build DOS (Landos) was : %10.2f  \n", t);
    printf(" estimated eig count in interval: %.15e \n", ecount);
    //-------------------- call splicer to slice the spectrum
    printf("DOS parameters: msteps = %d, nvec = %d, npnts = %d\n",
            msteps, nvec, npts);
    spslicer2(xdos, ydos, nslices, npts, sli);
   */
   /*
   nslices = 1;
   sli = (double*)malloc(2*sizeof(double));
   sli[0] = a;
   sli[1] = b;
   */
   /*
    printf("====================  SLICES FOUND  ====================\n");
    for (j = 0; j < nslices; j++) {
      printf(" %2d: [% .15e , % .15e]\n", j + 1, sli[j], sli[j + 1]);
    }
    //-------------------- # eigs per slice
    ev_int = (int)(1 + ecount / ((double)nslices));
    totcnt = 0;
    //-------------------- For each slice call RatLanrNr
    for (sl = 0; sl < nslices; sl++) {
      cout << "Beginning of Slice loop" << endl << flush;
      printf("======================================================\n");
      int nev2;
      double *lam, *Y, *res;
      int *ind;
      //--------------------
      a = sli[sl];
      b = sli[sl + 1];
      printf(" subinterval: [%.15e , %.15e]\n", a, b);
      xintv[0] = a;
      xintv[1] = b;
      xintv[2] = lmin;
      xintv[3] = lmax;
      //-------------------- set up default parameters for pol.
      set_pol_def(&pol);
      // can change default values here e.g.
      pol.damping = 2;
      pol.thresh_int = 0.8;
      pol.thresh_ext = 0.2;
      // pol.max_deg  = 300;
      //-------------------- Now determine polymomial
      find_pol(xintv, &pol);
      printf(" polynomial deg %d, bar %.15e gam %.15e\n", pol.deg,
              pol.bar, pol.gam);
      // save_vec(pol.deg+1, pol.mu, "OUT/mu.txt");
      //-------------------- approximate number of eigenvalues wanted
      nev = ev_int + 2;
      //-------------------- Dimension of Krylov subspace and maximal iterations
      mlan = max(5 * nev, 300);
      mlan = min(mlan, n);
      //-------------------- then call ChenLanNr
      ierr = ChebLanNr(xintv, mlan, tol, vinit, &pol, &nev2, &lam, &Y, &res,
                       fstats);
      if (ierr) {
        printf("ChebLanTr error %d\n", ierr);
        MFEM_ASSERT(false, "ChebLanTr error");
      }
   */
   /* sort the eigenvals: ascending order
    * ind: keep the orginal indices */
   /*
      ind = (int *)malloc(nev2 * sizeof(int));
      sort_double(nev2, lam, ind);
      printf(" number of eigenvalues found: %d\n", nev2);
   */
   /* print eigenvalues */
   /*
      printf("    Eigenvalues in [a, b]\n");
      printf("    Computed [%d]        ||Res||\n", nev2);
      for (i = 0; i < nev2; i++) {
        printf("% .15e  %.1e\n", lam[i], res[ind[i]]);
      }
      printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - "
        "- - - - - - - - - - - - - - - - - -\n");
      memcpy(&alleigs[totcnt], lam, nev2 * sizeof(double));
      totcnt += nev2;
      counts[sl] = nev2;
      //-------------------- free allocated space withing this scope
      if (lam)
        free(lam);
      if (Y)
        free(Y);
      if (res)
        free(res);
      free_pol(&pol);
      free(ind);
      cout << "End of Slice loop" << endl << flush;
    } // for (sl=0; sl<nslices; sl++)
    //-------------------- free other allocated space
    printf(" --> Total eigenvalues found = %d\n", totcnt);
   */
   // sprintf(path, "OUT/EigsOut_Lan_MMPLanN_(%s_%s)", io.MatNam1, io.MatNam2);
   /*
   FILE *fmtout = fopen(path, "w");
   if (fmtout) {
     for (j = 0; j < totcnt; j++)
       fprintf(fmtout, "%.15e\n", alleigs[j]);
     fclose(fmtout);
   }
   */
   /*
    double t4 = evsl_timer();
    cout << "EVSL Time: " << t4 - t3 << endl;

    free(vinit);
    free(sli);
    // free_coo(&Acoo);
    free_csr(&Acsr);
    // free_coo(&Bcoo);
    free_csr(&Bcsr);
    FreeBSolDirectData(Bsol);
    free(alleigs);
    free(counts);
    // free(xdos);
    // free(ydos);
    if (fstats != stdout) {
      fclose(fstats);
    }

   }
   delete Az;
   delete Mz;
      }
   */
   cout << "Leaving LOBPCG block of GetEigenvalues" << endl << flush;

}

void
MaxwellBlochWaveEquationUniAMR::GetEigenvector(unsigned int i,
                                         HypreParVector & Er,
                                         HypreParVector & Ei,
                                         HypreParVector & Br,
                                         HypreParVector & Bi)
{
   this->GetEigenvectorE(i, Er, Ei);
   this->GetEigenvectorB(i, Br, Bi);
}

void
MaxwellBlochWaveEquationUniAMR::GetEigenvectorE(unsigned int i,
                                          HypreParVector & Er,
                                          HypreParVector & Ei)
{
   double * data = NULL;
   if ( vecs_ != NULL )
   {
      data = (double*)*vecs_[i];
   }
   else
   {
      if ( lobpcg_ )
      {
         data = (double*)lobpcg_->GetEigenvector(i);
      }
      else if ( ame_ )
      {
         if ( i%2 == 0 )
         {
            data = (double*)ame_->GetEigenvector(i/2);
         }
         else
         {
            data = (double*)ame_->GetEigenvector((i-1)/2);
         }
      }
   }

   if ( lobpcg_ )
   {
      Er.SetData(&data[0]);
      Ei.SetData(&data[hcurl_loc_size_]);
   }
   else if ( ame_ )
   {
      if ( i%2 == 0 )
      {
         Er.SetData(&data[0]);
         Ei.SetData(vec0_->GetData());
      }
      else
      {
         Er.SetData(vec0_->GetData());
         Ei.SetData(&data[0]);
      }
   }
}

void
MaxwellBlochWaveEquationUniAMR::GetEigenvectorB(unsigned int i,
                                          HypreParVector & Br,
                                          HypreParVector & Bi)
{
   vector<double> eigenvalues;
   this->GetEigenvalues(eigenvalues);

   if ( lobpcg_ )
   {
      if ( vecs_ != NULL )
      {
         C_->Mult(*vecs_[i], *blkHDiv_);
      }
      else
      {
         C_->Mult(lobpcg_->GetEigenvector(i), *blkHDiv_);
      }
   }
   else if ( ame_ )
   {
      if ( i%2 == 0 )
      {
         blkHDiv_->GetBlock(1) = 0.0;
         Curl_->Mult(ame_->GetEigenvector(i/2),blkHDiv_->GetBlock(0));
      }
      else
      {
         Curl_->Mult(ame_->GetEigenvector((i-1)/2),blkHDiv_->GetBlock(1));
         blkHDiv_->GetBlock(0) = 0.0;
      }
   }

   if ( eigenvalues[i] != 0.0 ) { *blkHDiv_ /= sqrt(fabs(eigenvalues[i])); }

   double * data = (double*)*blkHDiv_;
   Bi.SetData(&data[0]);
   Br.SetData(&data[hdiv_loc_size_]); Br *= -1.0;
}

void
MaxwellBlochWaveEquationUniAMR::GetFourierCoefficients(HypreParVector & Vr,
                                                 HypreParVector & Vi,
                                                 Array2D<double> &f)
{
   f = 0.0;
   /*
   f[0][0] = InnerProduct(Vr,*X0_);
   f[0][1] = InnerProduct(Vi,*X0_);
   f[0][2] = InnerProduct(Vr,*Y0_);
   f[0][3] = InnerProduct(Vi,*Y0_);
   f[0][4] = InnerProduct(Vr,*Z0_);
   f[0][5] = InnerProduct(Vi,*Z0_);

   for (int i=0; i<3; i++)
   {
      f[2*i+1][0] = InnerProduct(Vr,*XC_[i]) - InnerProduct(Vi,*XS_[i]);
      f[2*i+1][1] = InnerProduct(Vi,*XC_[i]) + InnerProduct(Vr,*XS_[i]);
      f[2*i+1][2] = InnerProduct(Vr,*YC_[i]) - InnerProduct(Vi,*YS_[i]);
      f[2*i+1][3] = InnerProduct(Vi,*YC_[i]) + InnerProduct(Vr,*YS_[i]);
      f[2*i+1][4] = InnerProduct(Vr,*ZC_[i]) - InnerProduct(Vi,*ZS_[i]);
      f[2*i+1][5] = InnerProduct(Vi,*ZC_[i]) + InnerProduct(Vr,*ZS_[i]);

      f[2*i+2][0] = InnerProduct(Vr,*XC_[i]) + InnerProduct(Vi,*XS_[i]);
      f[2*i+2][1] = InnerProduct(Vi,*XC_[i]) - InnerProduct(Vr,*XS_[i]);
      f[2*i+2][2] = InnerProduct(Vr,*YC_[i]) + InnerProduct(Vi,*YS_[i]);
      f[2*i+2][3] = InnerProduct(Vi,*YC_[i]) - InnerProduct(Vr,*YS_[i]);
      f[2*i+2][4] = InnerProduct(Vr,*ZC_[i]) + InnerProduct(Vi,*ZS_[i]);
      f[2*i+2][5] = InnerProduct(Vi,*ZC_[i]) - InnerProduct(Vr,*ZS_[i]);
   }
   */
}

void
MaxwellBlochWaveEquationUniAMR::IdentifyDegeneracies(double zero_tol, double rel_tol,
                                               vector<set<int> > & degen)
{
   // Get the eigenvalues
   vector<double> eigenvalues;
   this->GetEigenvalues(eigenvalues);

   // Assume no degeneracies
   degen.resize(eigenvalues.size());

   // No eigenvalues means no degeneracies
   if ( eigenvalues.size() == 0 )
   {
      return;
   }

   // Place the first eigenvalue in the first set
   int nd = 0;
   degen[nd].insert(0);

   // Switch to select between zero_tol and rel_tol
   bool zeroes = eigenvalues[0] < zero_tol;

   // Scan the eigenvalues
   for (unsigned int i=1; i<eigenvalues.size(); i++)
   {
      if ( zeroes )
      {
         // The previous eigenvalue was a zero
         if ( eigenvalues[i] > zero_tol )
         {
            // This eigenvalue is not a zero
            nd++;
            zeroes = false;
         }
         // Place this eigenvalue in the appropriate grouping
         degen[nd].insert(i);
      }
      else
      {
         // The previous eigenvalue was non-zero
         if ( fabs( eigenvalues[i] - eigenvalues[i-1] ) >
              ( eigenvalues[i] + eigenvalues[i-1] ) * 0.5 * rel_tol )
         {
            // This eigenvalue belongs to a new grouping
            nd++;
         }
         // Place this eigenvalue in the appropriate grouping
         degen[nd].insert(i);
      }
   }

   // Adjust size down to the number of degeneracies identified
   degen.resize( nd + 1 );
}

void
MaxwellBlochWaveEquationUniAMR::GetFieldAverages(unsigned int i,
                                           Vector & Er, Vector & Ei,
                                           Vector & Br, Vector & Bi,
                                           Vector & Dr, Vector & Di,
                                           Vector & Hr, Vector & Hi)
{
   /*
   if ( fourierHCurl_ == NULL)
   {
     MFEM_ASSERT(bravais_ != NULL, "MaxwellBlochWaveEquationUniAMR: "
    "Field averages cannot be computed "
           "without a BravaisLattice object.");

     fourierHCurl_ = new HCurlFourierSeries(*bravais_, *HCurlFESpace_);
   }
   */
   // vector<double> eigs;
   // this->GetEigenvalues(eigs);

   // double omega = (eigs[i]>0.0)?sqrt(eigs[i]):0.0;

   // Vector arr(3), ari(3), air(3), aii(3);

   // fourierHCurl_->SetMode(0,0,0);

   HypreParVector  ParEr(HCurlFESpace_->GetComm(),
                         HCurlFESpace_->GlobalTrueVSize(),
                         NULL,
                         HCurlFESpace_->GetTrueDofOffsets());

   HypreParVector  ParEi(HCurlFESpace_->GetComm(),
                         HCurlFESpace_->GlobalTrueVSize(),
                         NULL,
                         HCurlFESpace_->GetTrueDofOffsets());

   HypreParVector  ParBr(HDivFESpace_->GetComm(),
                         HDivFESpace_->GlobalTrueVSize(),
                         NULL,
                         HDivFESpace_->GetTrueDofOffsets());

   HypreParVector  ParBi(HDivFESpace_->GetComm(),
                         HDivFESpace_->GlobalTrueVSize(),
                         NULL,
                         HDivFESpace_->GetTrueDofOffsets());

   this->GetEigenvector(i, ParEr, ParEi, ParBr, ParBi);

   Er.SetSize(3); Er = 0.0;
   Ei.SetSize(3); Ei = 0.0;
   Dr.SetSize(3); Dr = 0.0;
   Di.SetSize(3); Di = 0.0;
   Hr.SetSize(3); Hr = 0.0;
   Hi.SetSize(3); Hi = 0.0;
   Br.SetSize(3); Br = 0.0;
   Bi.SetSize(3); Bi = 0.0;

   for (int i=0; i<3; i++)
   {
      Er[i] += *AvgHCurl_coskx_[i] * ParEr;
      Er[i] -= *AvgHCurl_sinkx_[i] * ParEi;

      Ei[i] += *AvgHCurl_sinkx_[i] * ParEr;
      Ei[i] += *AvgHCurl_coskx_[i] * ParEi;

      Br[i] += *AvgHDiv_coskx_[i] * ParBr;
      Br[i] -= *AvgHDiv_sinkx_[i] * ParBi;

      Bi[i] += *AvgHDiv_sinkx_[i] * ParBr;
      Bi[i] += *AvgHDiv_coskx_[i] * ParBi;

      Dr[i] += *AvgHCurl_eps_coskx_[i] * ParEr;
      Dr[i] -= *AvgHCurl_eps_sinkx_[i] * ParEi;

      Di[i] += *AvgHCurl_eps_sinkx_[i] * ParEr;
      Di[i] += *AvgHCurl_eps_coskx_[i] * ParEi;

      Hr[i] += *AvgHDiv_muInv_coskx_[i] * ParBr;
      Hr[i] -= *AvgHDiv_muInv_sinkx_[i] * ParBi;

      Hi[i] += *AvgHDiv_muInv_sinkx_[i] * ParBr;
      Hi[i] += *AvgHDiv_muInv_coskx_[i] * ParBi;
   }
   /*
   // Compute the averages of the real and imaginary parts of E
   fourierHCurl_->GetCoefficient(ParEr, arr, ari);
   fourierHCurl_->GetCoefficient(ParEi, air, aii);

   Er = arr; Er -= aii;
   Ei = air; Ei += ari;

   // Compute the averages of the real and imaginary parts of B
   // using the fact that Curl E + i omega B = 0 which, for the averages,
   // translates to -i k x E + i omega B = 0 or B = k x E / omega.
   Br.SetSize(3); Bi.SetSize(3);
   Br[0] = kappa_[1] * Er[2] - kappa_[2] * Er[1];
   Br[1] = kappa_[2] * Er[0] - kappa_[0] * Er[2];
   Br[2] = kappa_[0] * Er[1] - kappa_[1] * Er[0];
   Br /= (omega>0.0)?omega:1.0;

   Bi[0] = kappa_[1] * Ei[2] - kappa_[2] * Ei[1];
   Bi[1] = kappa_[2] * Ei[0] - kappa_[0] * Ei[2];
   Bi[2] = kappa_[0] * Ei[1] - kappa_[1] * Ei[0];
   Bi /= (omega>0.0)?omega:1.0;

   // Compute the averages of the real and imaginary parts of H
   // using the fact that H = mu^{-1} B = i mu^{-1} Curl E / omega.
   // Note that in this case we cannot work strictly with field averages
   // because the value of mu might vary throughout the unit cell.

   ConstantCoefficient * kConst = dynamic_cast<ConstantCoefficient*>(kCoef_);
   if ( kConst != NULL )
   {
    // The coefficient mu is constant
    double muInv = kConst->constant;

    Hr = Br; Hr *= muInv;
    Hi = Bi; Hi *= muInv;
   }
   else
   {
    // fourierHCurl_->GetCoefficient(, arr, ari);
    // fourierHCurl_->GetCoefficient(, air, aii);

    // Hr = arr; Hr -= aii;
    // Hi = air; Hi += ari;
   }

   // Compute the averages of the real and imaginary parts of D
   // using the fact that Curl H - i omega D = 0 which, for the averages,
   // translates to -i k x H - i omega D = 0 or D = - k x H / omega.
   Dr.SetSize(3); Di.SetSize(3);
   Dr[0] = kappa_[1] * Hr[2] - kappa_[2] * Hr[1];
   Dr[1] = kappa_[2] * Hr[0] - kappa_[0] * Hr[2];
   Dr[2] = kappa_[0] * Hr[1] - kappa_[1] * Hr[0];
   Dr /= (omega>0.0)?-omega:1.0;

   Di[0] = kappa_[1] * Hi[2] - kappa_[2] * Hi[1];
   Di[1] = kappa_[2] * Hi[0] - kappa_[0] * Hi[2];
   Di[2] = kappa_[0] * Hi[1] - kappa_[1] * Hi[0];
   Di /= (omega>0.0)?-omega:1.0;
   */
}

void
MaxwellBlochWaveEquationUniAMR::ComputeHomogenizedCoefs()
{

}

void
MaxwellBlochWaveEquationUniAMR::DetermineBasis(const Vector & v1,
                                         std::vector<Vector> & e)
{
   e.resize(3);

   double kNorm = kappa_.Norml2();
   if ( kNorm < 1.0e-4 )
   {
      for (int i=0; i<3; i++)
      {
         e[i].SetSize(3); e[i] = 0.0; e[i][i] = 1.0;
      }
      return;
   }

   e[2].SetSize(3); e[2] = kappa_; e[2] /= kNorm;

   double e2dotv1 = e[2] * v1;
   e[1].SetSize(3); e[1] = v1; e[1].Add(-e2dotv1,v1);
   double e1norm  = e[1].Norml2();
   e[1] /= e1norm;

   e[0].SetSize(3);

   e[0][0] = e[1][1] * e[2][2] - e[1][2] * e[2][1];
   e[0][0] = e[1][2] * e[2][0] - e[1][0] * e[2][2];
   e[0][2] = e[1][0] * e[2][1] - e[1][1] * e[2][0];
}

void
MaxwellBlochWaveEquationUniAMR::WriteVisitFields(const string & prefix,
                                           const string & label)
{
   cout << "Writing VisIt data to: " << prefix  << " " << label << endl;

   ParGridFunction Er(this->GetHCurlFESpace());
   ParGridFunction Ei(this->GetHCurlFESpace());

   ParGridFunction Br(this->GetHDivFESpace());
   ParGridFunction Bi(this->GetHDivFESpace());

   HypreParVector ErVec(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());
   HypreParVector EiVec(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());

   HypreParVector BrVec(this->GetHDivFESpace()->GetComm(),
                        this->GetHDivFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHDivFESpace()->GetTrueDofOffsets());
   HypreParVector BiVec(this->GetHDivFESpace()->GetComm(),
                        this->GetHDivFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHDivFESpace()->GetTrueDofOffsets());

   VisItDataCollection visit_dc(label.c_str(), pmesh_);
   visit_dc.SetPrefixPath(prefix.c_str());

   if ( dynamic_cast<GridFunctionCoefficient*>(mCoef_) )
   {
      GridFunctionCoefficient * gfc =
         dynamic_cast<GridFunctionCoefficient*>(mCoef_);
      visit_dc.RegisterField("epsilon", gfc->GetGridFunction() );
   }
   if ( dynamic_cast<GridFunctionCoefficient*>(kCoef_) )
   {
      GridFunctionCoefficient * gfc =
         dynamic_cast<GridFunctionCoefficient*>(kCoef_);
      visit_dc.RegisterField("muInv", gfc->GetGridFunction() );
   }
   /*
   if ( cosKappaX_ )
   {
     visit_dc.RegisterField("CosKappaX", cosKappaX_);
   }
   if ( sinKappaX_ )
   {
     visit_dc.RegisterField("SinKappaX", sinKappaX_);
   }
   */
   visit_dc.RegisterField("E_r", &Er);
   visit_dc.RegisterField("E_i", &Ei);
   visit_dc.RegisterField("B_r", &Br);
   visit_dc.RegisterField("B_i", &Bi);

   vector<double> eigenvalues;
   this->GetEigenvalues(eigenvalues);

   // cout << "Number of eigenmodes: " << nev_ << endl;
   for (int i=0; i<nev_; i++)
   {
      // cout << "Writing mode " << i << " corresponding to eigenvalue "
      //  << eigenvalues[i] << endl;
      this->GetEigenvector(i, ErVec, EiVec, BrVec, BiVec);

      Er = ErVec;
      Ei = EiVec;

      Br = BrVec;
      Bi = BiVec;

      visit_dc.SetCycle(i+1);
      if ( eigenvalues[i] > 0.0 )
      {
         visit_dc.SetTime(sqrt(eigenvalues[i]));
      }
      else if ( eigenvalues[i] > -1.0e-6 )
      {
         visit_dc.SetTime(0.0);
      }
      else
      {
         visit_dc.SetTime(-1.0);
      }

      visit_dc.Save();
   }
}

void
MaxwellBlochWaveEquationUniAMR::GetSolverStats(double &meanTime, double &stdDevTime,
                                         double &meanIter, double &stdDevIter,
                                         int &nSolves)
{
   nSolves = (int)solve_times_.size();

   meanTime = 0.0;
   for (unsigned int i=0; i<solve_times_.size(); i++)
   {
      meanTime += solve_times_[i];
   }
   if ( nSolves > 0 ) { meanTime /= solve_times_.size(); }

   double var = 0.0;
   for (unsigned int i=0; i<solve_times_.size(); i++)
   {
      var += pow(solve_times_[i]-meanTime, 2.0);
   }
   if ( nSolves > 0 ) { var /= solve_times_.size(); }
   stdDevTime = sqrt(var);

   meanIter = 0.0;
   /*
   for (unsigned int i=0; i<solve_iters_.size(); i++)
   {
     meanIter += solve_iters_[i];
   }
   meanIter /= solve_iters_.size();
   */
   var = 0.0;
   /*
   for (unsigned int i=0; i<solve_iters_.size(); i++)
   {
     var += pow(solve_iters_[i]-meanIter, 2.0);
   }
   var /= solve_iters_.size();
   */
   stdDevIter = sqrt(var);
}

MaxwellBlochWaveEquationUniAMR::MaxwellBlochWavePrecond::
MaxwellBlochWavePrecond(ParFiniteElementSpace & HCurlFESpace,
                        BlockDiagonalPreconditioner & BDP,
                        Operator & subSpaceProj,
                        //BlockOperator & LU,
                        double w)
   : Solver(2*HCurlFESpace.GlobalTrueVSize()),
     myid_(0), BDP_(&BDP), subSpaceProj_(&subSpaceProj), u_(NULL)
{
   // Initialize MPI variables
   MPI_Comm comm = HCurlFESpace.GetComm();
   MPI_Comm_rank(comm, &myid_);
   int numProcs = HCurlFESpace.GetNRanks();

   if ( myid_ == 0 ) { cout << "MaxwellBlochWavePrecond" << endl; }

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

MaxwellBlochWaveEquationUniAMR::
MaxwellBlochWavePrecond::~MaxwellBlochWavePrecond()
{
   delete u_;
}

void
MaxwellBlochWaveEquationUniAMR::
MaxwellBlochWavePrecond::Mult(const Vector & x, Vector & y) const
{
   if ( subSpaceProj_ )
   {
      BDP_->Mult(x,*u_);
      subSpaceProj_->Mult(*u_,y);
   }
   else
   {
      BDP_->Mult(x,y);
   }

}

void
MaxwellBlochWaveEquationUniAMR::
MaxwellBlochWavePrecond::SetOperator(const Operator & A)
{
   A_ = &A;
}

MaxwellBlochWaveProjectorUniAMR::
MaxwellBlochWaveProjectorUniAMR(//ParFiniteElementSpace & HDivFESpace,
   ParFiniteElementSpace & HCurlFESpace,
   ParFiniteElementSpace & H1FESpace,
   BlockOperator & M,
   double beta, const Vector & zeta)
   : Operator(2*HCurlFESpace.GlobalTrueVSize()),
     newBeta_(true),
     newZeta_(true),
     // HDivFESpace_(&HDivFESpace),
     HCurlFESpace_(&HCurlFESpace),
     H1FESpace_(&H1FESpace),
     beta_(beta),
     zeta_(zeta),
     T01_(NULL),
     Z01_(NULL),
     A0_(NULL),
     DKZ_(NULL),
     DKZT_(NULL),
     amg_cos_(NULL),
     minres_(NULL),
     Grad_(NULL),
     Zeta_(NULL),
     S0_(NULL),
     M_(&M),
     G_(NULL),
     urDummy_(NULL),
     uiDummy_(NULL),
     vrDummy_(NULL),
     viDummy_(NULL),
     u0_(NULL),
     v0_(NULL),
     u1_(NULL),
     v1_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_rank(H1FESpace.GetParMesh()->GetComm(), &myid_);

   if ( myid_ == 0 )
   {
      cout << "Constructing MaxwellBlochWaveProjectorUniAMR" << endl;
   }

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

   if ( myid_ > 0 ) { cout << "done" << endl; }
}

MaxwellBlochWaveProjectorUniAMR::~MaxwellBlochWaveProjectorUniAMR()
{
   delete urDummy_; delete uiDummy_; delete vrDummy_; delete viDummy_;
   delete u0_; delete v0_;
   delete u1_; delete v1_;
   delete T01_;
   delete Z01_;
   delete A0_;
   delete DKZ_;
   delete DKZT_;
   delete Zeta_;
   delete Grad_;
   delete S0_;
   delete G_;
   delete amg_cos_;
   delete minres_;
}

void
MaxwellBlochWaveProjectorUniAMR::SetBeta(double beta)
{
   beta_ = beta; newBeta_ = true;
}

void
MaxwellBlochWaveProjectorUniAMR::SetZeta(const Vector & zeta)
{
   zeta_ = zeta; newZeta_ = true;
}

void
MaxwellBlochWaveProjectorUniAMR::Setup()
{
   if ( myid_ == 0 )
   {
      cout << "Setting up MaxwellBlochWaveProjectorUniAMR" << endl;
   }

   if ( Grad_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Grad operator" << endl; }
      Grad_ = new ParDiscreteGradOperator(H1FESpace_,HCurlFESpace_);
      Grad_->Assemble();
      Grad_->Finalize();
      T01_ = Grad_->ParallelAssemble();
   }

   if ( newZeta_ )
   {
      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Building zeta times operator" << endl; }
         Zeta_ = new ParDiscreteVectorProductOperator(H1FESpace_,
                                                      HCurlFESpace_,zeta_);
         Zeta_->Assemble();
         Zeta_->Finalize();
         Z01_ = Zeta_->ParallelAssemble();
      }
   }

   if ( G_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Block G" << endl; }
      G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
   }
   G_->SetBlock(0,0,T01_);
   G_->SetBlock(1,1,T01_);
   if ( fabs(beta_) > 0.0 )
   {
      // G_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_*M_PI/180.0);
      // G_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/180.0);
      G_->SetBlock(0,1,Z01_, beta_);
      G_->SetBlock(1,0,Z01_,-beta_);
   }
   G_->owns_blocks = 0;

   if ( newBeta_ || newZeta_ )
   {
      if ( myid_ == 0 ) { cout << "Forming GMG" << endl; }
      HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));
      HypreParMatrix * GMG = RAP(M1,T01_);

      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
         HypreParMatrix * ZMZ = RAP(M1,Z01_);

         HypreParMatrix * GMZ = RAP(T01_, M1, Z01_);
         HypreParMatrix * ZMG = RAP(Z01_, M1, T01_);
         *GMZ *= -1.0;
         DKZ_ = ParAdd(GMZ,ZMG);

         delete GMZ;
         delete ZMG;

         // *ZMZ *= beta_*beta_*M_PI*M_PI/32400.0;
         *ZMZ *= beta_*beta_;
         A0_ = ParAdd(GMG,ZMZ);
         delete GMG;
         delete ZMZ;
      }
      else
      {
         A0_ = GMG;
      }
   }

   if ( S0_ == NULL )
   {
      if ( myid_ > 0 ) { cout << "Building Block S0" << endl; }
      S0_ = new BlockOperator(block_trueOffsets0_);
   }
   S0_->SetDiagonalBlock(0,A0_,1.0);
   S0_->SetDiagonalBlock(1,A0_,1.0);
   if ( fabs(beta_) > 0.0 )
   {
      // S0_->SetBlock(0,1,DKZ_,-beta_*M_PI/180.0);
      // S0_->SetBlock(1,0,DKZ_, beta_*M_PI/180.0);
      S0_->SetBlock(0,1,DKZ_,-beta_);
      S0_->SetBlock(1,0,DKZ_, beta_);
   }
   S0_->owns_blocks = 0;

   if ( myid_ > 0 ) { cout << "Creating MINRES Solver" << endl; }
   delete minres_;
   minres_ = new MINRESSolver(H1FESpace_->GetComm());
   minres_->SetOperator(*S0_);
   minres_->SetRelTol(1e-13);
   minres_->SetMaxIter(3000);
   minres_->SetPrintLevel(0);

   newBeta_  = false;
   newZeta_  = false;

   if ( myid_ > 0 ) { cout << "done" << endl; }
}

void
MaxwellBlochWaveProjectorUniAMR::Update()
{
   // The finite element spaces have changed so we need to repopulate
   // these arrays.
   block_offsets0_.SetSize(3);
   block_offsets0_[0] = 0;
   block_offsets0_[1] = H1FESpace_->GetVSize();
   block_offsets0_[2] = H1FESpace_->GetVSize();
   block_offsets0_.PartialSum();

   block_offsets1_.SetSize(3);
   block_offsets1_[0] = 0;
   block_offsets1_[1] = HCurlFESpace_->GetVSize();
   block_offsets1_[2] = HCurlFESpace_->GetVSize();
   block_offsets1_.PartialSum();

   block_trueOffsets0_.SetSize(3);
   block_trueOffsets0_[0] = 0;
   block_trueOffsets0_[1] = H1FESpace_->TrueVSize();
   block_trueOffsets0_[2] = H1FESpace_->TrueVSize();
   block_trueOffsets0_.PartialSum();

   block_trueOffsets1_.SetSize(3);
   block_trueOffsets1_[0] = 0;
   block_trueOffsets1_[1] = HCurlFESpace_->TrueVSize();
   block_trueOffsets1_[2] = HCurlFESpace_->TrueVSize();
   block_trueOffsets1_.PartialSum();

   locSize_ = HCurlFESpace_->TrueVSize();

   // Reallocated the internal vectors
   delete u0_; delete v0_; delete u1_; delete v1_;
   u0_ = new BlockVector(block_trueOffsets0_);
   v0_ = new BlockVector(block_trueOffsets0_);
   u1_ = new BlockVector(block_trueOffsets1_);
   v1_ = new BlockVector(block_trueOffsets1_);

   Grad_->Update();
   delete T01_;
   T01_ = Grad_->ParallelAssemble();

   if ( Zeta_ != NULL )
   {
      Zeta_->Update();
      delete Z01_;
      Z01_ = Zeta_->ParallelAssemble();
   }

   delete G_;
   G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
   G_->SetBlock(0,0,T01_);
   G_->SetBlock(1,1,T01_);
   if ( fabs(beta_) > 0.0 )
   {
      // G_->SetBlock(0,1,Zeta_->ParallelAssemble(),beta_*M_PI/180.0);
      // G_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/180.0);
      G_->SetBlock(0,1,Z01_, beta_);
      G_->SetBlock(1,0,Z01_,-beta_);
   }
   G_->owns_blocks = 0;

   if ( myid_ == 0 ) { cout << "Forming GMG" << endl; }
   HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));
   HypreParMatrix * GMG = RAP(M1,T01_);

   delete A0_;
   if ( fabs(beta_) > 0.0 )
   {
      if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
      HypreParMatrix * ZMZ = RAP(M1,Z01_);

      HypreParMatrix * GMZ = RAP(T01_, M1, Z01_);
      HypreParMatrix * ZMG = RAP(Z01_, M1, T01_);

      *GMZ *= -1.0;
      DKZ_ = ParAdd(GMZ,ZMG);

      delete GMZ;
      delete ZMG;

      // *ZMZ *= beta_*beta_*M_PI*M_PI/32400.0;
      *ZMZ *= beta_*beta_;
      A0_ = ParAdd(GMG,ZMZ);
      delete GMG;
      delete ZMZ;
   }
   else
   {
      A0_ = GMG;
   }

   if ( myid_ > 0 ) { cout << "Building Block S0" << endl; }
   delete S0_;
   S0_ = new BlockOperator(block_trueOffsets0_);
   S0_->SetDiagonalBlock(0,A0_,1.0);
   S0_->SetDiagonalBlock(1,A0_,1.0);
   if ( fabs(beta_) > 0.0 )
   {
      // S0_->SetBlock(0,1,DKZ_,-beta_*M_PI/180.0);
      // S0_->SetBlock(1,0,DKZ_,beta_*M_PI/180.0);
      S0_->SetBlock(0,1,DKZ_,-beta_);
      S0_->SetBlock(1,0,DKZ_, beta_);
   }
   S0_->owns_blocks = 0;

   if ( myid_ > 0 ) { cout << "Creating MINRES Solver" << endl; }
   delete minres_;
   minres_ = new MINRESSolver(H1FESpace_->GetComm());
   minres_->SetOperator(*S0_);
   minres_->SetRelTol(1e-13);
   minres_->SetMaxIter(3000);
   minres_->SetPrintLevel(0);

   newBeta_  = false;
   newZeta_  = false;

   if ( myid_ > 0 ) { cout << "done" << endl; }
}

void
MaxwellBlochWaveProjectorUniAMR::Mult(const Vector &x, Vector &y) const
{
   M_->Mult(x,y);
   G_->MultTranspose(y,*u0_);
   *v0_ = 0.0;
   minres_->Mult(*u0_,*v0_);
   G_->Mult(*v0_,y);
   y *= -1.0;
   y += x;
}

void
ElementwiseEnergyNorm(BilinearFormIntegrator & bli,
                      ParGridFunction & x,
                      ParGridFunction & e)
{
   FiniteElementSpace *xfes = x.ParFESpace();
   FiniteElementSpace *efes = e.ParFESpace();
   Array<int> xdofs, edofs;

   Vector xvec;
   double loc_energy;

   DenseMatrix A;

   for (int i=0; i<xfes->GetNE(); i++)
   {
      xfes->GetElementVDofs(i, xdofs);
      x.GetSubVector(xdofs, xvec);

      bli.AssembleElementMatrix(*xfes->GetFE(i),
                                *xfes->GetElementTransformation(i),
                                A);

      loc_energy = A.InnerProduct(xvec,xvec);

      efes->GetElementVDofs(i, edofs);
      e.SetSubVector(edofs, &loc_energy);
   }
}
/*
LinearCombinationOperator::LinearCombinationOperator()
{}

LinearCombinationOperator::~LinearCombinationOperator()
{
   if ( owns_terms )
   {
      for (unsigned int i=0; i<ops_.size(); i++)
      {
         delete ops_[i];
      }
   }
}

void
LinearCombinationOperator::AddTerm(double coef, Operator & op)
{
   int h = op.Height();
   int w = op.Width();

   if ( ops_.size() == 0 )
   {
      this->height = h;
      this->width  = w;

      u_.SetSize(h);
   }
   else
   {
      MFEM_ASSERT(this->height == h, "the operators have differing heights");
      MFEM_ASSERT(this->width  == w, "the operators have differing widths");
   }

   coefs_.push_back(coef);
   ops_.push_back(&op);
}

void
LinearCombinationOperator::Mult(const Vector &x, Vector &y) const
{
   ops_[0]->Mult(x,y);
   y *= coefs_[0];

   for (unsigned int i=1; i<ops_.size(); i++)
   {
      ops_[i]->Mult(x,u_);
      u_ *= coefs_[i];
      y += u_;
   }
}
*/
} // namespace bloch
} // namespace mfem

#endif // MFEM_USE_MPI
