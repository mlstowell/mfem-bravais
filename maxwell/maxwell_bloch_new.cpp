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

#include "maxwell_bloch_new.hpp"
#include <fstream>

using namespace std;

namespace mfem
{

using namespace miniapps;

namespace bloch
{

MaxwellBlochWaveEquation::MaxwellBlochWaveEquation(ParMesh & pmesh,
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
     // alpha_a_(0.0),
     // alpha_i_(90.0),
     atol_(1.0e-6),
     beta_(0.0),
     omega_(-1.0),
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
     energy_(NULL),
     B_(NULL),
     minres_(NULL),
     gmres_(NULL)/*,
     cosKappaX_(NULL),
     sinKappaX_(NULL)*/
{
   // Initialize MPI variables
   comm_ = pmesh.GetComm();
   MPI_Comm_rank(comm_, &myid_);

   if ( myid_ == 0 )
   {
      cout << "Constructing MaxwellBlochWaveEquation" << endl;
   }

   int dim = pmesh.Dimension();

   zeta_.SetSize(dim);

   H1FESpace_    = new H1_ParFESpace(&pmesh,order,dim);
   HCurlFESpace_ = new ND_ParFESpace(&pmesh,order,dim);
   HDivFESpace_  = new RT_ParFESpace(&pmesh,order,dim);
   L2FESpace_    = new L2_ParFESpace(&pmesh,0,dim);

   hcurl_loc_size_ = HCurlFESpace_->TrueVSize();
   hdiv_loc_size_  = HDivFESpace_->TrueVSize();
   /*
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
   */
}

MaxwellBlochWaveEquation::~MaxwellBlochWaveEquation()
{
   delete lobpcg_;
   delete minres_;
   delete gmres_;
   delete B_;

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
   delete DKZ_;
   delete DKZT_;
   delete Curl_;
   delete Zeta_;
   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;
   delete L2FESpace_;
}

void
MaxwellBlochWaveEquation::SetKappa(const Vector & kappa)
{
   kappa_ = kappa;
   beta_  = kappa.Norml2();  newBeta_ = true;
   zeta_  = kappa;           newZeta_ = true;
   if ( fabs(beta_) > 0.0 )
   {
      zeta_ /= beta_;
   }
   /*
   if ( cosCoef_ == NULL )
   {
     cosCoef_ = new RealPhaseCoefficient();
   }
   if ( sinCoef_ == NULL )
   {
     sinCoef_ = new ImagPhaseCoefficient();
   }
   cosCoef_->SetKappa(kappa_);
   sinCoef_->SetKappa(kappa_);

   if ( cosKappaX_ == NULL )
   {
     cosKappaX_ = new ParGridFunction(H1FESpace_);
   }
   if ( sinKappaX_ == NULL )
   {
     sinKappaX_ = new ParGridFunction(H1FESpace_);
   }

   cosKappaX_->ProjectCoefficient(*cosCoef_);
   sinKappaX_->ProjectCoefficient(*sinCoef_);
   */
}

void
MaxwellBlochWaveEquation::SetBeta(double beta)
{
   beta_ = beta; newBeta_ = true;
}

void
MaxwellBlochWaveEquation::SetZeta(const Vector & zeta)
{
   zeta_ = zeta; newZeta_ = true;
}
/*
void
MaxwellBlochWaveEquation::SetAzimuth(double alpha_a)
{
   alpha_a_ = alpha_a; newAlpha_ = true;
}

void
MaxwellBlochWaveEquation::SetInclination(double alpha_i)
{
   alpha_i_ = alpha_i; newAlpha_ = true;
}
*/
/*
void
MaxwellBlochWaveEquation::SetOmega(double omega)
{
   omega_ = omega; newOmega_ = true;
}
*/
void
MaxwellBlochWaveEquation::SetAbsoluteTolerance(double atol)
{
   atol_ = atol;
}

void
MaxwellBlochWaveEquation::SetNumEigs(int nev)
{
   nev_ = nev;
}

void
MaxwellBlochWaveEquation::SetMassCoef(Coefficient & m)
{
   mCoef_ = &m; newMCoef_ = true;
}

void
MaxwellBlochWaveEquation::SetStiffnessCoef(Coefficient & k)
{
   kCoef_ = &k; newKCoef_ = true;
}

void
MaxwellBlochWaveEquation::Setup()
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
   }

   if ( Curl_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Curl operator" << endl; }
      Curl_ = new ParDiscreteCurlOperator(HCurlFESpace_,HDivFESpace_);
   }

   if ( newZeta_ || newBeta_ || newKCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Forming CMC" << endl; }
      HypreParMatrix * CMC = RAP(M2_,Curl_->ParallelAssemble());

      delete S1_;

      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
         HypreParMatrix * ZMZ = RAP(M2_,Zeta_->ParallelAssemble());

         HypreParMatrix * CMZ = RAP(Curl_->ParallelAssemble(),M2_,
                                    Zeta_->ParallelAssemble());
         HypreParMatrix * ZMC = RAP(Zeta_->ParallelAssemble(),M2_,
                                    Curl_->ParallelAssemble());

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
      C_->SetDiagonalBlock(0,Curl_->ParallelAssemble());
      C_->SetDiagonalBlock(1,Curl_->ParallelAssemble());
      if ( fabs(beta_) > 0.0 )
      {
         // C_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_*M_PI/(180.0*a_));
         // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/(180.0*a_));
         // C_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_/a_);
         // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_/a_);
         C_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_);
         C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_);
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
         SubSpaceProj_ = new MaxwellBlochWaveProjector(*HDivFESpace_,
                                                       *HCurlFESpace_,
                                                       *H1FESpace_,
                                                       *M_,beta_,zeta_);
         SubSpaceProj_->Setup();

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

   if ( omega_ >= 0.0 )
   {
      B_ = new LinearCombinationOperator(/*new BlockVector(block_offsets_)*/);
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

   newZeta_  = false;
   newBeta_  = false;
   newOmega_ = false;
   newMCoef_ = false;
   newKCoef_ = false;

   if ( myid_ == 0 ) { cout << "Leaving Setup" << endl; }
}

void
MaxwellBlochWaveEquation::SetInitialVectors(int num_vecs,
                                            HypreParVector ** vecs)
{
   if ( lobpcg_ )
   {
      lobpcg_->SetInitialVectors(num_vecs, vecs);
   }
}

void MaxwellBlochWaveEquation::Update()
{
   if ( myid_ == 0 ) { cout << "Building M2(k)" << endl; }
   ParBilinearForm m2(HDivFESpace_);
   m2.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoef_));
   m2.Assemble();
   m2.Finalize();
   delete M2_;
   M2_ = m2.ParallelAssemble();

   if ( Zeta_ ) { Zeta_->Update(); }
   if ( Curl_ ) { Curl_->Update(); }

   if ( myid_ == 0 ) { cout << "Forming CMC" << endl; }
   HypreParMatrix * CMC = RAP(M2_,Curl_->ParallelAssemble());

   delete S1_;

   if ( fabs(beta_) > 0.0 )
   {
      if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
      HypreParMatrix * ZMZ = RAP(M2_,Zeta_->ParallelAssemble());

      HypreParMatrix * CMZ = RAP(Curl_->ParallelAssemble(),M2_,
                                 Zeta_->ParallelAssemble());
      HypreParMatrix * ZMC = RAP(Zeta_->ParallelAssemble(),M2_,
                                 Curl_->ParallelAssemble());

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
   C_->SetDiagonalBlock(0,Curl_->ParallelAssemble());
   C_->SetDiagonalBlock(1,Curl_->ParallelAssemble());
   if ( fabs(beta_) > 0.0 )
   {
      // C_->SetBlock(0,1,Zeta_->ParallelAssemble(),beta_*M_PI/(180.0*a_));
      // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/(180.0*a_));
      // C_->SetBlock(0,1,Zeta_->ParallelAssemble(),beta_/a_);
      // C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_/a_);
      C_->SetBlock(0,1,Zeta_->ParallelAssemble(),beta_);
      C_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_);
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
void MaxwellBlochWaveEquation::TestVector(const HypreParVector & v)
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
MaxwellBlochWaveEquation::Solve()
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
MaxwellBlochWaveEquation::GetEigenvalues(vector<double> & eigenvalues)
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
MaxwellBlochWaveEquation::GetEigenvalues(int nev, const Vector & kappa,
                                         vector<HypreParVector*> & init_vecs,
                                         vector<double> & eigenvalues)
{
   this->SetNumEigs(nev);
   this->SetKappa(kappa);
   this->Setup();
   this->SetInitialVectors(nev, &init_vecs[0]);
   this->Solve();
   this->GetEigenvalues(eigenvalues);
}

void
MaxwellBlochWaveEquation::GetEigenvector(unsigned int i,
                                         HypreParVector & Er,
                                         HypreParVector & Ei,
                                         HypreParVector & Br,
                                         HypreParVector & Bi)
{
   this->GetEigenvectorE(i, Er, Ei);
   this->GetEigenvectorB(i, Br, Bi);
}

void
MaxwellBlochWaveEquation::GetEigenvectorE(unsigned int i,
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
MaxwellBlochWaveEquation::GetEigenvectorB(unsigned int i,
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
MaxwellBlochWaveEquation::GetFourierCoefficients(HypreParVector & Vr,
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
MaxwellBlochWaveEquation::WriteVisitFields(const string & prefix,
                                           const string & label)
{
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

   for (int i=0; i<nev_; i++)
   {
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

MaxwellBlochWaveEquation::MaxwellBlochWavePrecond::
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

MaxwellBlochWaveEquation::
MaxwellBlochWavePrecond::~MaxwellBlochWavePrecond()
{
   delete u_;
}

void
MaxwellBlochWaveEquation::
MaxwellBlochWavePrecond::Mult(const Vector & x, Vector & y) const
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
   subSpaceProj_->Mult(*u_,y);
}

void
MaxwellBlochWaveEquation::
MaxwellBlochWavePrecond::SetOperator(const Operator & A)
{
   A_ = &A;
}

MaxwellBlochWaveProjector::
MaxwellBlochWaveProjector(ParFiniteElementSpace & HDivFESpace,
                          ParFiniteElementSpace & HCurlFESpace,
                          ParFiniteElementSpace & H1FESpace,
                          BlockOperator & M,
                          double beta, const Vector & zeta)
   : Operator(2*HCurlFESpace.GlobalTrueVSize()),
     newBeta_(true),
     newZeta_(true),
     HDivFESpace_(&HDivFESpace),
     HCurlFESpace_(&HCurlFESpace),
     H1FESpace_(&H1FESpace),
     beta_(beta),
     zeta_(zeta),
     Z_(NULL),
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
      cout << "Constructing MaxwellBlochWaveProjector" << endl;
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

MaxwellBlochWaveProjector::~MaxwellBlochWaveProjector()
{
   delete urDummy_; delete uiDummy_; delete vrDummy_; delete viDummy_;
   delete u0_; delete v0_;
   delete u1_; delete v1_;
   delete Z_;
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
MaxwellBlochWaveProjector::SetBeta(double beta)
{
   beta_ = beta; newBeta_ = true;
}

void
MaxwellBlochWaveProjector::SetZeta(const Vector & zeta)
{
   zeta_ = zeta; newZeta_ = true;
}

void
MaxwellBlochWaveProjector::Setup()
{
   if ( myid_ == 0 )
   {
      cout << "Setting up MaxwellBlochWaveProjector" << endl;
   }

   if ( Grad_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Grad operator" << endl; }
      Grad_ = new ParDiscreteGradOperator(H1FESpace_,HCurlFESpace_);
   }

   if ( newZeta_ )
   {
      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Building zeta times operator" << endl; }
         Zeta_ = new ParDiscreteVectorProductOperator(H1FESpace_,
                                                      HCurlFESpace_,zeta_);
      }
   }

   if ( G_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Block G" << endl; }
      G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
   }
   G_->SetBlock(0,0,Grad_->ParallelAssemble());
   G_->SetBlock(1,1,Grad_->ParallelAssemble());
   if ( fabs(beta_) > 0.0 )
   {
      // G_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_*M_PI/180.0);
      // G_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/180.0);
      G_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_);
      G_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_);
   }
   G_->owns_blocks = 0;

   if ( newBeta_ || newZeta_ )
   {
      if ( myid_ == 0 ) { cout << "Forming GMG" << endl; }
      HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));
      HypreParMatrix * GMG = RAP(M1,Grad_->ParallelAssemble());

      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
         HypreParMatrix * ZMZ = RAP(M1,Zeta_->ParallelAssemble());

         HypreParMatrix * GMZ = RAP(Grad_->ParallelAssemble(),M1,
                                    Zeta_->ParallelAssemble());
         HypreParMatrix * ZMG = RAP(Zeta_->ParallelAssemble(),M1,
                                    Grad_->ParallelAssemble());
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
MaxwellBlochWaveProjector::Update()
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
   if ( Zeta_ != NULL ) { Zeta_->Update(); }

   delete G_;
   G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
   G_->SetBlock(0,0,Grad_->ParallelAssemble());
   G_->SetBlock(1,1,Grad_->ParallelAssemble());
   if ( fabs(beta_) > 0.0 )
   {
      // G_->SetBlock(0,1,Zeta_->ParallelAssemble(),beta_*M_PI/180.0);
      // G_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_*M_PI/180.0);
      G_->SetBlock(0,1,Zeta_->ParallelAssemble(), beta_);
      G_->SetBlock(1,0,Zeta_->ParallelAssemble(),-beta_);
   }
   G_->owns_blocks = 0;

   if ( myid_ == 0 ) { cout << "Forming GMG" << endl; }
   HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));
   HypreParMatrix * GMG = RAP(M1,Grad_->ParallelAssemble());

   delete A0_;
   if ( fabs(beta_) > 0.0 )
   {
      if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
      HypreParMatrix * ZMZ = RAP(M1,Zeta_->ParallelAssemble());

      HypreParMatrix * GMZ = RAP(Grad_->ParallelAssemble(),M1,
                                 Zeta_->ParallelAssemble());
      HypreParMatrix * ZMG = RAP(Zeta_->ParallelAssemble(),M1,
                                 Grad_->ParallelAssemble());
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
MaxwellBlochWaveProjector::Mult(const Vector &x, Vector &y) const
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

} // namespace bloch
} // namespace mfem

#endif // MFEM_USE_MPI
