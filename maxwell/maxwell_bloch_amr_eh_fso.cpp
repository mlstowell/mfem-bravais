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

#include "maxwell_bloch_amr_eh.hpp"
#include <fstream>

using namespace std;

namespace mfem
{

using namespace miniapps;

namespace bloch
{

MaxwellBlochWaveEquationAMR::MaxwellBlochWaveEquationAMR(MPI_Comm & comm,
                                                         Mesh & mesh,
                                                         int order,
                                                         int ar,
                                                         int logging)
   : comm_(comm),
     myid_(0),
     num_procs_(1),
     order_(order),
     ar_(ar),
     logging_(logging),
     nev_(-1),
     part_(NULL),
     newSizes_(true),
     newBeta_(true),
     newZeta_(true),
     // newOmega_(true),
     newACoef_(true),
     newMCoef_(true),
     currSizes_(false),
     currVecs_(false),
     currA1_(false),
     currB1_(false),
     currM1_(false),
     currM2_(false),
     currT12_(false),
     currD12_(false),
     currZ12_(false),
     currAMS_(false),
     currBOpA_(false),
     currBOpC_(false),
     currBOpM_(false),
     currBOpA12_(false),
     currBOpM12_(false),
     currBPC_(false),
     currBDP_(false),
     currSSP_(false),
     pmesh_(NULL),
     H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     L2FESpace_(NULL),
     bravais_(NULL),
     fourierHCurl_(NULL),
     atol_(1.0e-6),
     beta_(0.0),
     omega_max_(1.0),
     aCoef_(NULL),
     mCoef_(NULL),
     zCoef_(NULL),
     aCoefGF_(NULL),
     mCoefGF_(NULL),
     A_(NULL),
     M_(NULL),
     C_(NULL),
     A12_(NULL),
     M12_(NULL),
     blkHCurl_(NULL),
     blkHDiv_(NULL),
     M1_(NULL),
     M2_(NULL),
     A1_(NULL),
     T1_(NULL),
     T2_(NULL),
     T12_(NULL),
     D12_(NULL),
     D12T_(NULL),
     Z12_(NULL),
     Z12T_(NULL),
     B1_(NULL),
     // B1T_(NULL),
     CMC_(NULL),
     ZMZ_(NULL),
     CMZ_(NULL),
     ZMC_(NULL),
     T1Inv_ams_(NULL),
     T2Inv_ams_(NULL),
     // T1Inv_minres_(NULL),
     t12_(NULL),
     m1_(NULL),
     m2_(NULL),
     t1_(NULL),
     t2_(NULL),
     d12_(NULL),
     z12_(NULL),
     BDP_(NULL),
     Precond_(NULL),
     SubSpaceProj_(NULL),
     FSO_(NULL),
     FSOInv_(NULL),
     MInv_(NULL),
     MPC_(NULL),
     M1PC_(NULL),
     M2PC_(NULL),
     T2PC_(NULL),
     vecs_(NULL),
     vec0_(NULL),
     num_init_vecs_(0),
     init_vecs_(NULL),
     init_er_(NULL),
     init_ei_(NULL),
     init_hr_(NULL),
     init_hi_(NULL),
     lobpcg_(NULL),
     ame_(NULL)
     // energy_(NULL)
{
   // Initialize MPI variables
   // comm_ = pmesh.GetComm();
   MPI_Comm_rank(comm_, &myid_);
   MPI_Comm_size(comm_, &num_procs_);

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Constructing MaxwellBlochWaveEquation" << endl;
   }

   pmesh_ = new ParMesh(comm, mesh);

   int dim = pmesh_->Dimension();

   zeta_.SetSize(dim); zeta_ = 0.0;

   H1FESpace_    = new H1_ParFESpace(pmesh_, order, dim);
   HCurlFESpace_ = new ND_ParFESpace(pmesh_, order, dim);
   HDivFESpace_  = new RT_ParFESpace(pmesh_, order, dim);
   L2FESpace_    = new L2_ParFESpace(pmesh_,     0, dim);
   /*
   hcurl_loc_size_ = HCurlFESpace_->TrueVSize();
   hdiv_loc_size_  = HDivFESpace_->TrueVSize();

   block_trueOffsets1_.SetSize(3);
   block_trueOffsets1_[0] = 0;
   block_trueOffsets1_[1] = HCurlFESpace_->TrueVSize();
   block_trueOffsets1_[2] = HCurlFESpace_->TrueVSize();
   block_trueOffsets1_.PartialSum();

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
   */
   // blkHCurl_ = new BlockVector(block_trueOffsets_);
   // blkHDiv_  = new BlockVector(block_trueOffsets2_);
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Done constructing MaxwellBlochWaveEquation" << endl;
   }
}

MaxwellBlochWaveEquationAMR::~MaxwellBlochWaveEquationAMR()
{
   delete lobpcg_;
   delete ame_;

   if ( vecs_ != NULL )
   {
      for (int i=0; i<nev_; i++) { delete vecs_[i]; }
      delete [] vecs_;
   }

   delete FSO_;
   delete FSOInv_;
   delete MInv_;
   delete MPC_;
   delete M1PC_;
   delete M2PC_;
   delete T2PC_;
   delete SubSpaceProj_;
   delete Precond_;
   delete vec0_;

   delete blkHCurl_;
   delete blkHDiv_;
   delete A_;
   delete M_;
   delete C_;
   delete A12_;
   delete M12_;
   delete BDP_;
   delete T1Inv_ams_;
   delete T2Inv_ams_;
   // delete T1Inv_minres_;

   delete M1_;
   delete M2_;
   if ( A1_ != CMC_ ) { delete A1_; }
   delete T1_;
   delete T2_;
   delete T12_;
   delete D12_;
   delete D12T_;
   delete Z12_;
   delete Z12T_;
   delete B1_;
   // delete B1T_;
   delete CMC_;
   delete ZMZ_;
   delete CMZ_;
   delete ZMC_;

   delete t12_;
   delete m1_;
   delete m2_;
   delete t1_;
   delete t2_;
   delete d12_;
   delete z12_;

   delete zCoef_;

   delete aCoefGF_;
   delete mCoefGF_;

   if ( init_er_ != NULL )
   {
      for (int i=0; i<num_init_vecs_; i++)
      {
         delete init_er_[i];
      }
      delete init_er_;
   }
   if ( init_ei_ != NULL )
   {
      for (int i=0; i<num_init_vecs_; i++)
      {
         delete init_ei_[i];
      }
      delete init_ei_;
   }
   if ( init_hr_ != NULL )
   {
      for (int i=0; i<num_init_vecs_; i++)
      {
         delete init_hr_[i];
      }
      delete init_hr_;
   }
   if ( init_hi_ != NULL )
   {
      for (int i=0; i<num_init_vecs_; i++)
      {
         delete init_hi_[i];
      }
      delete init_hi_;
   }
   if ( init_vecs_ != NULL )
   {
      for (int i=0; i<num_init_vecs_; i++)
      {
         delete init_vecs_[i];
      }
      delete init_vecs_;
   }

   delete fourierHCurl_;

   delete part_;

   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;
   delete L2FESpace_;
   delete pmesh_;
}

void
MaxwellBlochWaveEquationAMR::SetKappa(const Vector & kappa)
{
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Setting Kappa: ";
      kappa.Print(cout);
      cout << endl;
   }
   kappa_ = kappa;

   double beta  = kappa_.Norml2();
   this->SetBeta(beta);

   if ( fabs(beta_) > 0.0 )
   {
      Vector zeta(kappa_); zeta /= beta_;
      this->SetZeta(zeta);
   }
}

void
MaxwellBlochWaveEquationAMR::SetBeta(double beta)
{
   double diffb = fabs(beta_ - beta);
   double avgb  = 0.5 * ( beta_ + beta );

   if ( avgb < 1.0e-4 )
   {
      if ( diffb > 1.0e-4 )
      {
         if ( myid_ == 0  && logging_ > 0 )
         {
            cout << "Setting Beta: " << beta << endl;
         }
         beta_ = beta; newBeta_ = true;
      }
   }
   else
   {
      if ( diffb / avgb > 1.0e-4 )
      {
         if ( myid_ == 0  && logging_ > 0 )
         {
            cout << "Setting Beta: " << beta << endl;
         }
         beta_ = beta; newBeta_ = true;
      }
   }
   if ( newBeta_ )
   {
      currA1_   = false;
      currB1_   = false;
      currAMS_  = false;
      currBDP_  = false;
      currBPC_  = false;
      currSSP_  = false;
      currBOpA_ = false;
      currBOpA12_ = false;
      currBOpC_ = false;

      if ( SubSpaceProj_ ) { SubSpaceProj_->SetBeta(beta_); }
   }
}

void
MaxwellBlochWaveEquationAMR::SetZeta(const Vector & zeta)
{
   Vector diffz(zeta_); diffz -= zeta;

   if ( diffz.Norml2() > 1.0e-4 )
   {
      if ( myid_ == 0  && logging_ > 0 )
      {
         cout << "Setting Zeta: ";
         zeta.Print(cout);
         cout << endl;
      }
      zeta_ = zeta; newZeta_ = true;
      currA1_   = false;
      currB1_   = false;
      currZ12_  = false;
      currAMS_  = false;
      currBDP_  = false;
      currBPC_  = false;
      currSSP_  = false;
      currBOpA_ = false;
      currBOpA12_ = false;
      currBOpC_ = false;

      delete zCoef_; zCoef_ = new VectorConstantCoefficient(zeta_);

      if ( SubSpaceProj_ ) { SubSpaceProj_->SetZeta(zeta_); }
   }
}

void
MaxwellBlochWaveEquationAMR::SetAbsoluteTolerance(double atol)
{
   atol_ = atol;
}

void
MaxwellBlochWaveEquationAMR::SetNumEigs(int nev)
{
   cout << "Setting nev = " << nev << endl;
   nev_ = nev;
}

void
MaxwellBlochWaveEquationAMR::SetMassCoef(Coefficient & m)
{
   mCoef_ = &m; newMCoef_ = true;
   currM1_   = false;
   currBOpM_ = false;
   currBOpM12_ = false;
   currSSP_  = false;
}

void
MaxwellBlochWaveEquationAMR::SetInvMassCoef(Coefficient & m)
{
   mInvCoef_ = &m; newMCoef_ = true;
   currM1_   = false;
   currBOpM_ = false;
   currBOpM12_ = false;
   currSSP_  = false;
}

void
MaxwellBlochWaveEquationAMR::SetStiffnessCoef(Coefficient & a)
{
   aCoef_ = &a; newACoef_ = true;
   currA1_   = false;
   currB1_   = false;
   currM2_   = false;
   currD12_  = false;
   currBOpA_ = false;
   currBOpA12_ = false;
   currBOpM12_ = false;
   currBPC_  = false;
   currBDP_  = false;
   currAMS_  = false;
}

void
MaxwellBlochWaveEquationAMR::SetMaximumLightSpeed(double c)
{
   // Estimate smallest edge length
   int ne = pmesh_->GetNE();

   Vector xmin, xmax;

   pmesh_->GetBoundingBox(xmin, xmax);

   double vol = 1.0;
   for (int i=0; i<pmesh_->SpaceDimension(); i++)
   {
      vol *= xmax[i] - xmin[i];
   }

   double h = pow(vol / ne, 1.0/3.0);

   omega_max_ = 2.0 * M_PI * c / h;

   if ( myid_ == 0 )
   {
      cout << "Estimate of Maximum Frequency: " << omega_max_ << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::Setup()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::Setup" << endl;
   }

   if ( !currVecs_ ) { this->SetupTmpVectors(); }

   if ( fabs(beta_) > 0.0 || true )
   {
      if ( myid_ == 0  && logging_ > 1 )
      {
         cout << "  Setting up the Block system for beta > 0" << endl;
      }
      // if ( !currBOpC_ ) { this->SetupBlockOperatorC(); }
      this->SetupBlockSolver();
   }
   else
   {
      if ( myid_ == 0  && logging_ > 1 )
      {
         cout << "  Setting up the linear system for beta == 0" << endl;
      }
      if ( !currT12_ ) { this->SetupT12(); }
      this->SetupSolver();
   }

   newSizes_ = false;
   newBeta_  = false;
   newZeta_  = false;
   newACoef_ = false;
   newMCoef_ = false;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::Setup" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::OldSetup()
{
   /*
   if ( newACoef_ )
   {
      if ( myid_ == 0 ) { cout << "Building M2(k)" << endl; }
      ParBilinearForm m2(HDivFESpace_);
      m2.AddDomainIntegrator(new VectorFEMassIntegrator(*aCoef_));
      m2.Assemble();
      m2.Finalize();
      delete M2_;
      M2_ = m2.ParallelAssemble();
   }
   */
   /*
   if ( newZeta_ )
   {
      if ( myid_ == 0 ) { cout << "Building zeta cross operator" << endl; }
      delete z12_;
      z12_ = new ParDiscreteVectorCrossProductOperator(HCurlFESpace_,
                                                       HDivFESpace_,zeta_);
      z12_->Assemble();
      z12_->Finalize();
      Z12_ = z12_->ParallelAssemble();
   }
   */
   /*
   if ( t12_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Curl operator" << endl; }
      t12_ = new ParDiscreteCurlOperator(HCurlFESpace_,HDivFESpace_);
      t12_->Assemble();
      t12_->Finalize();
      T12_ = t12_->ParallelAssemble();
   }
   */
   /*
   if ( newZeta_ || newBeta_ || newACoef_ )
   {
      if ( myid_ == 0 ) { cout << "Forming CMC" << endl; }
      HypreParMatrix * CMC = RAP(M2_,T12_);

      delete A1_;

      if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
      if ( fabs(beta_) > 0.0 )
      {
         HypreParMatrix * ZMZ = RAP(M2_, Z12_);
         HypreParMatrix * CMZ = RAP(T12_, M2_, Z12_);
         HypreParMatrix * ZMC = RAP(Z12_, M2_, T12_);

         *ZMC *= -1.0;
         delete B1_;
         B1_ = ParAdd(CMZ,ZMC);
         delete CMZ;
         delete ZMC;

         *ZMZ *= beta_*beta_;
         A1_ = ParAdd(CMC,ZMZ);
         delete CMC;
         delete ZMZ;
      }
      else
      {
         A1_ = CMC;
      }
   }
   */
   /*
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
   */
   /*
   if ( newZeta_ || newBeta_ || newACoef_ )
   {
      if ( A_ == NULL )
      {
         if ( myid_ == 0 ) { cout << "Building Block A" << endl; }
         A_ = new BlockOperator(block_trueOffsets1_);
      }
      A_->SetDiagonalBlock(0, A1_);
      A_->SetDiagonalBlock(1, A1_);
      if ( fabs(beta_) > 0.0 )
      {
         A_->SetBlock(0, 1, B1_,  beta_);
         A_->SetBlock(1, 0, B1_, -beta_);
      }
      A_->owns_blocks = 0;
   }
   */
   /*
   if ( newMCoef_ )
   {
      if ( M_ == NULL )
      {
         if ( myid_ == 0 ) { cout << "Building Block M" << endl; }
         M_ = new BlockOperator(block_trueOffsets1_);
      }
      M_->SetDiagonalBlock(0, M1_);
      M_->SetDiagonalBlock(1, M1_);
      M_->owns_blocks = 0;
   }
   */
   /*
   if ( newZeta_ || newBeta_ )
   {
      if ( C_ == NULL )
      {
         if ( myid_ == 0 ) { cout << "Building Block C" << endl; }
         C_ = new BlockOperator(block_trueOffsets2_, block_trueOffsets1_);
      }
      C_->SetDiagonalBlock(0, T12_);
      C_->SetDiagonalBlock(1, T12_);
      if ( fabs(beta_) > 0.0 )
      {
         C_->SetBlock(0, 1, Z12_,  beta_);
         C_->SetBlock(1, 0, Z12_, -beta_);
      }
      C_->owns_blocks = 0;
   }
   */
   if ( newZeta_ || newBeta_ || newACoef_ )
   {
      /*
       if ( myid_ == 0 ) { cout << "Building T1Inv" << endl; }

       delete T1Inv_ams_;
       T1Inv_ams_ = new HypreAMS(*A1_,HCurlFESpace_);

       if ( fabs(beta_*180.0) < M_PI )
       {
          cout << "HypreAMS::SetSingularProblem()" << endl;
          T1Inv_ams_->SetSingularProblem();
       }
      */
      /*
      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Building BDP" << endl; }
         delete BDP_;
         BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets1_);
         BDP_->SetDiagonalBlock(0, T1Inv_ams_);
         BDP_->SetDiagonalBlock(1, T1Inv_ams_);
         BDP_->owns_blocks = 0;
      }
      */
   }
   /*
   if ( ( newZeta_ || newBeta_ || newMCoef_ || newACoef_ ) && nev_ > 0 )
   {
      if ( fabs(beta_) > 0.0 )
      {
         delete SubSpaceProj_;
         if ( myid_ == 0 ) { cout << "Building Subspace Projector" << endl; }
         SubSpaceProj_ = new MaxwellBlochWaveProjectorAMR(*HCurlFESpace_,
                                                          *H1FESpace_,
                                                          *M_, beta_, zeta_, 1);
         // SubSpaceProj_->Setup();

         if ( myid_ == 0 ) { cout << "Building Preconditioner" << endl; }
         delete Precond_;
         Precond_ = new MaxwellBlochWavePrecond(*HCurlFESpace_, *BDP_,
                                                SubSpaceProj_, 0.5);
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
         ame_->SetPreconditioner(*T1Inv_ams_);
         ame_->SetMaxIter(2000);
         ame_->SetTol(atol_);
         ame_->SetRelTol(1e-8);
         ame_->SetPrintLevel(1);

         // Set the matrices which define the linear system
         ame_->SetMassMatrix(*M1_);
         ame_->SetOperator(*A1_);

         if ( vec0_ == NULL )
         {
            vec0_ = new HypreParVector(*M1_);
         }
         *vec0_ = 0.0;
      }
   }
   */
   /*
   Vector xHat(3), yHat(3), zHat(3);
   xHat = yHat = zHat = 0.0;
   xHat(0) = 1.0; yHat(1) = 1.0; zHat(2) = 1.0;

   newZeta_  = false;
   newBeta_  = false;
   // newOmega_ = false;
   newMCoef_ = false;
   newACoef_ = false;
   */
   if ( myid_ == 0 ) { cout << "Leaving Setup" << endl; }
}

void
MaxwellBlochWaveEquationAMR::SetupSizes()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupSizes" << endl;
   }
   if ( ! currSizes_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Setting new sizes" << endl;
      }
      if ( block_trueOffsets1_.Size() < 3 )
      {
         block_trueOffsets1_.SetSize(3);
         block_trueOffsets2_.SetSize(3);
         block_trueOffsets12_.SetSize(5);
      }

      block_trueOffsets1_[0] = 0;
      block_trueOffsets1_[1] = HCurlFESpace_->TrueVSize();
      block_trueOffsets1_[2] = HCurlFESpace_->TrueVSize();
      block_trueOffsets1_.PartialSum();

      block_trueOffsets2_[0] = 0;
      block_trueOffsets2_[1] = HDivFESpace_->TrueVSize();
      block_trueOffsets2_[2] = HDivFESpace_->TrueVSize();
      block_trueOffsets2_.PartialSum();

      block_trueOffsets12_[0] = 0;
      block_trueOffsets12_[1] = HCurlFESpace_->TrueVSize();
      block_trueOffsets12_[2] = HCurlFESpace_->TrueVSize();
      block_trueOffsets12_[3] = HCurlFESpace_->TrueVSize();
      block_trueOffsets12_[4] = HCurlFESpace_->TrueVSize();
      block_trueOffsets12_.PartialSum();

      hcurl_loc_size_ = HCurlFESpace_->TrueVSize();
      hdiv_loc_size_  = HDivFESpace_->TrueVSize();

      MPI_Allreduce(&hcurl_loc_size_, &hcurl_glb_size_, 1,
                    HYPRE_MPI_INT, MPI_SUM, comm_);
      MPI_Allreduce(&hdiv_loc_size_, &hdiv_glb_size_, 1,
                    HYPRE_MPI_INT, MPI_SUM, comm_);

      int locSize = 4 * hcurl_loc_size_;

      if (HYPRE_AssumedPartitionCheck())
      {
         if ( part_ == NULL ) { part_ = new HYPRE_Int[2]; }

         MPI_Scan(&locSize, &part_[1], 1, HYPRE_MPI_INT, MPI_SUM, comm_);

         part_[0] = part_[1] - locSize;
      }
      else
      {
         if ( part_ == NULL ) { part_ = new HYPRE_Int[num_procs_+1]; }

         MPI_Allgather(&locSize, 1, MPI_INT,
                       &part_[1], 1, HYPRE_MPI_INT, comm_);

         part_[0] = 0;
         for (int i=0; i<num_procs_; i++)
         {
            part_[i+1] += part_[i];
         }

         // hcurl_glb_size_ = part_[num_procs_];
      }
      // hcurl_glb_size_ /= 2;

      tdof_offsets_.SetSize(num_procs_+1);
      HYPRE_Int * hcurl_tdof_offsets = HCurlFESpace_->GetTrueDofOffsets();
      for (int i=0; i<tdof_offsets_.Size(); i++)
      {
         tdof_offsets_[i] = 2 * hcurl_tdof_offsets[i];
      }
      currSizes_ = true;
   }
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupSizes" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupTmpVectors()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupTmpVectors" << endl;
   }
   if ( !currSizes_ )
   {
      this->SetupSizes();
   }

   if ( !currVecs_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Allocating temporary vectors" << endl;
      }
      delete blkHCurl_; blkHCurl_ = new BlockVector(block_trueOffsets1_);
      delete blkHDiv_;   blkHDiv_ = new BlockVector(block_trueOffsets2_);
      delete vec0_;         vec0_ = new HypreParVector(HCurlFESpace_);

      if ( aCoefGF_ == NULL )
      {
         aCoefGF_ = new ParGridFunction(L2FESpace_);
      }
      else
      {
         aCoefGF_->Update();
      }
      if ( mCoefGF_ == NULL )
      {
         mCoefGF_ = new ParGridFunction(L2FESpace_);
      }
      else
      {
         mCoefGF_->Update();
      }

      currVecs_ = true;
   }
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupTmpVectors" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupT12()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupT12" << endl;
   }
   if ( t12_ == NULL )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building Curl discrete operator" << endl;
      }
      t12_ = new ParDiscreteCurlOperator(HCurlFESpace_,HDivFESpace_);
   }
   else
   {
      t12_->Update();
   }

   if ( ! currT12_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Assembling Curl operator" << endl;
      }
      t12_->Assemble();
      t12_->Finalize();
      delete T12_; T12_ = t12_->ParallelAssemble();
      T12_->Print("T12.mat");
      currT12_ = true;
   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupT12" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupD12()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupD12" << endl;
   }

   if ( ! currD12_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building curl operator" << endl;
      }
      if ( newACoef_ )
      {
         delete d12_;
         d12_ = new ParMixedBilinearForm(HCurlFESpace_,
                                         HCurlFESpace_);
         d12_->AddDomainIntegrator(new MixedVectorCurlIntegrator(*aCoef_));
      }
      else
      {
         d12_->Update();
      }

      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Assembling curl operator" << endl;
      }
      d12_->Assemble();
      d12_->Finalize();
      delete D12_; D12_ = d12_->ParallelAssemble();
      delete D12T_; D12T_ = D12_->Transpose();

      D12_->Print("D12.mat");
      D12T_->Print("D12T.mat");
      currD12_ = true;
   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupD12" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupZ12()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupZ12" << endl;
   }

   if ( ! currZ12_ && fabs(beta_) > 0.0 )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building zeta cross discrete operator" << endl;
      }
      /*
      if ( newZeta_ )
      {
         delete z12_;
         z12_ = new ParDiscreteVectorCrossProductOperator(HCurlFESpace_,
                                                          HDivFESpace_,zeta_);
      }
      else
      {
         z12_->Update();
      }

      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Assembling zeta cross operator" << endl;
      }
      z12_->Assemble();
      z12_->Finalize();
      delete Z12_; Z12_ = z12_->ParallelAssemble();
      */
      if ( newZeta_ )
      {
         delete z12_;
         z12_ = new ParMixedBilinearForm(HCurlFESpace_,
                                         HCurlFESpace_);
         z12_->AddDomainIntegrator(new MixedCrossProductIntegrator(*zCoef_));
      }
      else
      {
         z12_->Update();
      }

      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Assembling zeta cross operator" << endl;
      }
      z12_->Assemble();
      z12_->Finalize();
      delete Z12_; Z12_ = z12_->ParallelAssemble();
      delete Z12T_; Z12T_ = Z12_->Transpose();

      Z12_->Print("Z12.mat");
      Z12T_->Print("Z12T.mat");
      currZ12_ = true;

   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupZ12" << endl;
   }
}
/*
void
MaxwellBlochWaveEquationAMR::SetupA1()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupA1" << endl;
   }
   if ( !currT12_ ) { this->SetupT12(); }
   if ( !currZ12_ ) { this->SetupZ12(); }
   if ( !currM2_  ) { this->SetupM2(); }

   if ( ! currA1_ )
   {
      if ( myid_ == 0 && logging_ > 1 ) { cout << "  Forming CMC" << endl; }

      if ( A1_ != CMC_ ) { delete A1_; }
      delete CMC_; CMC_ = RAP(M2_,T12_);



      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 && logging_ > 1 )
         { cout << "  Forming ZMZ" << endl; }
         delete ZMZ_; ZMZ_ = RAP(M2_, Z12_);

         *ZMZ_ *= beta_*beta_;
         A1_ = ParAdd(CMC_, ZMZ_);
      }
      else
      {
         A1_ = CMC_;
      }
      // A1_->Print("A1.mat");
      currA1_   = true;
      currBOpA_ = false;
      currAMS_  = false;
   }
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupA1" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupB1()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupB1" << endl;
   }
   if ( !currT12_ ) { this->SetupT12(); }
   if ( !currZ12_ ) { this->SetupZ12(); }
   if ( !currM2_  ) { this->SetupM2(); }

   if ( ! currB1_ )
   {
      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 && logging_ > 1 )
         { cout << "  Forming CMZ and ZMC" << endl; }
         delete CMZ_; CMZ_ = RAP(T12_, M2_, Z12_);
         delete ZMC_; ZMC_ = RAP(Z12_, M2_, T12_);

         *ZMC_ *= -1.0;
         delete B1_; B1_ = ParAdd(CMZ_,ZMC_);
      }

      currB1_   = true;
      currBOpA_ = false;
      currAMS_  = false;
   }
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupB1" << endl;
   }
}
*/
void
MaxwellBlochWaveEquationAMR::SetupM1()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupM1" << endl;
   }
   if ( ! currM1_ )
   {
      if ( myid_ == 0 && logging_ > 1 ) { cout << "  Building M1(m)" << endl; }
      ParBilinearForm m1(HCurlFESpace_);
      m1.AddDomainIntegrator(new VectorFEMassIntegrator(*mCoef_));

      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Assembling M1(m)" << endl;
      }
      m1.Assemble();
      m1.Finalize();
      delete M1_;
      M1_ = m1.ParallelAssemble();
      currM1_  = true;
      currAMS_ = false;
   }
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupM1" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupM2()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupM2" << endl;
   }
   if ( ! currM2_ )
   {
      if ( myid_ == 0 && logging_ > 1 ) { cout << "  Building M2(a)" << endl; }
      ParBilinearForm m2(HCurlFESpace_);
      m2.AddDomainIntegrator(new VectorFEMassIntegrator(*aCoef_));

      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Assembling M2(a)" << endl;
      }
      m2.Assemble();
      m2.Finalize();
      delete M2_;
      M2_ = m2.ParallelAssemble();
      currM2_ = true;
      currA1_ = false;
      currB1_ = false;
   }
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupM2" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupSolver()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupSolver" << endl;
   }
   if ( !currAMS_ ) { this->SetupAMS(); }
   if ( !currM1_  ) { this->SetupM1(); }
   /*
   if ( ame_ == NULL )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building HypreAME solver" << endl;
      }
      ame_ = new HypreAME(comm_);
      ame_->SetNumModes(nev_/2);
      ame_->SetMaxIter(2000);
      ame_->SetTol(atol_);
      ame_->SetRelTol(1e-8);
      ame_->SetPrintLevel(1);
   }
   if ( newSizes_ || newMCoef_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Setting mass matrix" << endl;
      }
      ame_->SetMassMatrix(*M1_);
   }
   if ( newSizes_ || newACoef_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Setting preconditioner and operator" << endl;
      }
      cout << flush;
      ame_->SetPreconditioner(*T1Inv_ams_);
      A1_->Print("A1_ams.mat");
      cout << "set op" << endl << flush;
      ame_->SetOperator(*A1_);
      cout << "back from set op" << endl << flush;
   }
   */
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "  Building HypreAME solver" << endl;
   }
   delete ame_;
   ame_ = new HypreAME(comm_);
   ame_->SetNumModes(nev_/2);
   ame_->SetMaxIter(2000);
   ame_->SetTol(atol_);
   ame_->SetRelTol(1e-8);
   ame_->SetPrintLevel(1);
   ame_->SetMassMatrix(*M1_);
   ame_->SetPreconditioner(*T1Inv_ams_);
   ame_->SetOperator(*A1_);

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupSolver" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupBlockOperatorC()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupBlockOperatorC"
           << endl;
   }
   if ( !currSizes_ ) { this->SetupSizes(); }
   if ( !currT12_   ) { this->SetupT12();   }
   if ( !currZ12_   ) { this->SetupZ12();   }

   if ( C_ == NULL || newSizes_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building Block Operator C" << endl;
      }
      delete C_;
      C_ = new BlockOperator(block_trueOffsets2_, block_trueOffsets1_);
      C_->owns_blocks = 0;
   }
   if ( ! currBOpC_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Adding Diag blocks to Block Operator C" << endl;
      }
      C_->SetBlock(0, 0, T12_);
      C_->SetBlock(1, 1, T12_);
   }
   if ( newSizes_ || ( fabs(beta_) > 0.0 && (newBeta_ || newZeta_) ) )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Adding Off-Diag blocks to Block Operator C" << endl;
      }
      C_->SetBlock(0, 1, Z12_,  beta_);
      C_->SetBlock(1, 0, Z12_, -beta_);
   }
   currBOpC_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupBlockOperatorC"
           << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupBlockOperatorA()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupBlockOperatorA"
           << endl;
   }

   if ( !currSizes_ ) { this->SetupSizes(); }
   /*
   if ( !currA1_    ) { this->SetupA1();    }
   if ( !currB1_    ) { this->SetupB1();    }

   if ( A_ == NULL || newSizes_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building Block A" << endl;
      }
      delete A_; A_ = new BlockOperator(block_trueOffsets1_);
      A_->owns_blocks = 0;
   }
   if ( newZeta_ || newBeta_ || newACoef_ || newSizes_ )
   {
      A_->SetDiagonalBlock(0, A1_);
      A_->SetDiagonalBlock(1, A1_);
      if ( fabs(beta_) > 0.0 )
      {
         A_->SetBlock(0, 1, B1_,  beta_);
         A_->SetBlock(1, 0, B1_, -beta_);
      }
   }
   */
   currBOpA_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupBlockOperatorA"
           << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupBlockOperatorA12()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupBlockOperatorA12"
           << endl;
   }

   if ( !currSizes_ ) { this->SetupSizes(); }
   if ( !currD12_    ) { this->SetupD12();    }
   if ( !currZ12_    ) { this->SetupZ12();    }

   if ( A12_ == NULL || newSizes_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building Block A12" << endl;
      }
      delete A12_; A12_ = new BlockOperator(block_trueOffsets12_);
      A12_->owns_blocks = 0;
   }
   if ( newZeta_ || newBeta_ || newACoef_ || newSizes_ )
   {
      A12_->SetBlock(0, 3, D12T_,  1.0);
      A12_->SetBlock(1, 2, D12T_, -1.0);
      A12_->SetBlock(2, 1,  D12_, -1.0);
      A12_->SetBlock(3, 0,  D12_,  1.0);
      if ( fabs(beta_) > 0.0 )
      {
         A12_->SetBlock(0, 2, Z12T_,  beta_);
         A12_->SetBlock(1, 3, Z12T_,  beta_);
         A12_->SetBlock(2, 0,  Z12_,  beta_);
         A12_->SetBlock(3, 1,  Z12_,  beta_);
      }
   }
   currBOpA12_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupBlockOperatorA12"
           << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupBlockOperatorM()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupBlockOperatorM"
           << endl;
   }

   if ( !currSizes_ ) { this->SetupSizes(); }
   if ( !currM1_    ) { this->SetupM1();    }
   if ( !currM2_    ) { this->SetupM2();    }

   if ( M_ == NULL || newSizes_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building Block M" << endl;
      }
      delete M_;
      M_ = new BlockOperator(block_trueOffsets1_);
      M_->owns_blocks = 0;
   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "  Adding Diag blocks to Block Operator M" << endl;
   }
   M_->SetDiagonalBlock(0, M1_);
   M_->SetDiagonalBlock(1, M1_);

   currBOpM_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupBlockOperatorM"
           << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupBlockOperatorM12()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupBlockOperatorM12"
           << endl;
   }

   if ( !currSizes_ ) { this->SetupSizes(); }
   if ( !currM1_    ) { this->SetupM1();    }
   if ( !currM2_    ) { this->SetupM2();    }

   if ( M12_ == NULL || newSizes_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building Block M12" << endl;
      }
      delete M12_;
      M12_ = new BlockOperator(block_trueOffsets12_);
      M12_->owns_blocks = 0;
   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "  Adding Diag blocks to Block Operator M12" << endl;
   }
   M12_->SetDiagonalBlock(0, M1_);
   M12_->SetDiagonalBlock(1, M1_);
   M12_->SetDiagonalBlock(2, M2_);
   M12_->SetDiagonalBlock(3, M2_);

   currBOpM12_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupBlockOperatorM12"
           << endl;
   }
}
/*
void
MaxwellBlochWaveEquationAMR::SetupSubSpaceProjector()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupSubSpaceProjector"
           << endl;
   }
   if ( !currBOpM_ ) { this->SetupBlockOperatorM(); }

   if ( SubSpaceProj_ == NULL )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building Subspace Projector" << endl;
      }
      SubSpaceProj_ = new MaxwellBlochWaveProjectorAMR(*HCurlFESpace_,
                                                       *H1FESpace_,
                                                       *M_, beta_, zeta_,
                                                       logging_);
   }
   else
   {
      if ( newBeta_ )
      {
         SubSpaceProj_->SetBeta(beta_);
      }
      if ( newZeta_ )
      {
         SubSpaceProj_->SetZeta(zeta_);
      }
      if ( newMCoef_ )
      {
         SubSpaceProj_->SetM(*M_);
      }
   }
   if ( newBeta_ || newZeta_ || newMCoef_ )
   {
      SubSpaceProj_->Setup();
      currBPC_ = false;
   }
   else
   {
      SubSpaceProj_->SetM(*M_);
      SubSpaceProj_->Update();
      currBPC_ = false;
   }

   currSSP_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupSubSpaceProjector"
           << endl;
   }
}
  */
void
MaxwellBlochWaveEquationAMR::SetupAMS()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupAMS" << endl;
   }

   // if ( !currA1_ ) { this->SetupA1(); }

   if ( !currAMS_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building HypreAMS Solver" << endl;
      }

      delete T1Inv_ams_;
      T1Inv_ams_ = new HypreAMS(*A1_, HCurlFESpace_);

      if ( fabs(beta_*180.0) < M_PI )
      {
         if ( myid_ == 0 && logging_ > 1 )
         {
            cout << "  HypreAMS::SetSingularProblem()" << endl;
         }
         T1Inv_ams_->SetSingularProblem();
      }
      currAMS_ = true;
      currBDP_ = false;
   }
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupAMS" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupBlockDiagPrecond()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupBlockDiagPrecond"
           << endl;
   }

   if ( !currSizes_ ) { this->SetupSizes(); }
   if ( !currAMS_ ) { this->SetupAMS(); }

   if ( BDP_ == NULL || newSizes_ )
   {
      if ( myid_ == 0 ) { cout << "Building BDP" << endl; }
      delete BDP_;
      BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets1_);
      BDP_->owns_blocks = 0;
   }
   if ( newBeta_ || newZeta_ || newACoef_ || newSizes_ )
   {
      BDP_->SetDiagonalBlock(0, T1Inv_ams_);
      BDP_->SetDiagonalBlock(1, T1Inv_ams_);

      currBPC_ = false;
   }
   currBDP_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupBlockDiagPrecond"
           << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupBlockPrecond()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupBlockPrecond"
           << endl;
   }
   if ( !currBOpM_ ) { this->SetupBlockOperatorM();    }
   if ( !currBOpA_ ) { this->SetupBlockOperatorA();    }
   // if ( !currSSP_  ) { this->SetupSubSpaceProjector(); }
   if ( !currBDP_  ) { this->SetupBlockDiagPrecond();  }

   if ( Precond_ == NULL || newSizes_ )
   {
      if ( myid_ == 0 ) { cout << "Building Preconditioner" << endl; }
      delete Precond_;
      Precond_ = new MaxwellBlochWavePrecond(*HCurlFESpace_, *BDP_,
                                             SubSpaceProj_, 0.5);
   }
   if ( newBeta_ || newZeta_ || newACoef_ || newSizes_ )
   {
      Precond_->SetOperator(*A_);
   }

   currBPC_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupBlockPrecond"
           << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetupBlockSolver()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::SetupBlockSolver"
           << endl;
   }
   // if ( !currBPC_  ) { this->SetupBlockPrecond();   }
   if ( !currBOpA12_ ) { this->SetupBlockOperatorA12(); }
   if ( !currBOpM12_ ) { this->SetupBlockOperatorM12(); }
   /*
   if ( lobpcg_ == NULL )
   {
      lobpcg_ = new HypreLOBPCG(comm_);
      lobpcg_->SetNumModes(nev_);
      lobpcg_->SetMaxIter(2000);
      lobpcg_->SetTol(atol_);
      lobpcg_->SetPrintLevel(1);
   }
   if ( newSizes_ || newMCoef_ )
   {
      lobpcg_->SetMassMatrix(*this->GetMOperator());
   }
   if ( newSizes_ || newBeta_ || newZeta_ || newACoef_ )
   {
      lobpcg_->SetPreconditioner(*this->GetPreconditioner());
      lobpcg_->SetPrecondUsageMode(1);
      lobpcg_->SetOperator(*this->GetAOperator());
      lobpcg_->SetSubSpaceProjector(*this->GetSubSpaceProjector());
   }
   */
   delete M1PC_;
   M1PC_ = new HypreDiagScale(*M1_);

   delete M2PC_;
   M2PC_ = new HypreDiagScale(*M2_);

   delete MPC_;
   MPC_ = new BlockDiagonalPreconditioner(block_trueOffsets12_);
   MPC_->SetDiagonalBlock(0, M1PC_);
   MPC_->SetDiagonalBlock(1, M1PC_);
   MPC_->SetDiagonalBlock(2, M2PC_);
   MPC_->SetDiagonalBlock(3, M2PC_);
   MPC_->owns_blocks = false;

   delete MInv_;
   MInv_ = new CGSolver(comm_);
   MInv_->SetRelTol(1e-12);
   MInv_->SetMaxIter(500);
   // MInv_->SetOperator(*this->GetMOperator());
   MInv_->SetOperator(*M12_);
   MInv_->SetPreconditioner(*MPC_);

   double kSqrMin = MAXFLOAT;
   double kSqrMax = 0.0;
   Vector kEff(3);

   // This will only work for Cubic lattices
   for (int k=-1; k<=1; k++)
   {
      for (int j=-1; j<=1; j++)
      {
         for (int i=-1; i<=1; i++)
         {
            kEff = kappa_;
            kEff[0] += 2.0 * M_PI * i;
            kEff[1] += 2.0 * M_PI * j;
            kEff[2] += 2.0 * M_PI * k;
            double kSqr = kEff * kEff;
            kSqrMin = min(kSqrMin, kSqr);
            kSqrMax = max(kSqrMax, kSqr);
         }
      }
   }

   // double lambda0 = 0.5 * (kSqrMin + kSqrMax) / 5.5;
   // double lambda0 = 0.5 * (sqrt(kSqrMin) + sqrt(kSqrMax)) * 1.2;
   double lambda0 = 2.6;
   cout << "lambda0 = " << lambda0 << endl;
   omega_shift_ = lambda0;

   delete FSO_;
   FSO_ = new FoldedSpectrumOperator(*A12_,
                                     *M12_,
                                     *MInv_,
                                     lambda0);
   /*
   delete FSOInv_;
   FSOInv_ = new MINRESSolver(comm_);
   FSOInv_->SetRelTol(1e-4);
   FSOInv_->SetMaxIter(500);
   FSOInv_->SetOperator(*FSO_);
   */
   ConstantCoefficient sCoef(lambda0*lambda0);
   ProductCoefficient tmCoef(sCoef, *mCoef_);
   ProductCoefficient taCoef(sCoef, *aCoef_);

   DenseMatrix kk(3);
   kk(0,0) = zeta_[1] * zeta_[1] + zeta_[2] * zeta_[2];
   kk(1,1) = zeta_[2] * zeta_[2] + zeta_[0] * zeta_[0];
   kk(2,2) = zeta_[0] * zeta_[0] + zeta_[1] * zeta_[1];
   kk(0,1) = -zeta_[0] * zeta_[1];
   kk(0,2) = -zeta_[0] * zeta_[2];
   kk(1,0) = -zeta_[1] * zeta_[0];
   kk(1,2) = -zeta_[1] * zeta_[2];
   kk(2,0) = -zeta_[2] * zeta_[0];
   kk(2,1) = -zeta_[2] * zeta_[1];
   kk *= beta_ * beta_;
   MatrixFunctionCoefficient tkCoef(kk, *aCoef_);
   MatrixFunctionCoefficient tk2Coef(kk, *mInvCoef_);

   delete t1_;
   t1_ = new ParBilinearForm(HCurlFESpace_);
   t1_->AddDomainIntegrator(new VectorFEMassIntegrator(tmCoef));
   t1_->AddDomainIntegrator(new VectorFEMassIntegrator(tkCoef));
   t1_->AddDomainIntegrator(new CurlCurlIntegrator(*aCoef_));
   t1_->Assemble();
   t1_->Finalize();
   delete T1_; T1_ = t1_->ParallelAssemble();

   delete T1Inv_ams_;
   T1Inv_ams_ = new HypreAMS(*T1_, HCurlFESpace_);
   T1Inv_ams_->SetMaxIter(3);
   T1Inv_ams_->SetPrintLevel(0);

   delete t2_;
   t2_ = new ParBilinearForm(HCurlFESpace_);
   t2_->AddDomainIntegrator(new VectorFEMassIntegrator(taCoef));
   t2_->AddDomainIntegrator(new VectorFEMassIntegrator(tk2Coef));
   t2_->AddDomainIntegrator(new CurlCurlIntegrator(*mInvCoef_));
   t2_->Assemble();
   t2_->Finalize();
   delete T2_; T2_ = t2_->ParallelAssemble();

   delete T2Inv_ams_;
   T2Inv_ams_ = new HypreAMS(*T2_, HCurlFESpace_);
   T2Inv_ams_->SetMaxIter(3);
   T2Inv_ams_->SetPrintLevel(0);

   BlockOperator T12(block_trueOffsets12_);
   T12.owns_blocks = false;
   T12.SetDiagonalBlock(0, T1_);
   T12.SetDiagonalBlock(1, T1_);
   T12.SetDiagonalBlock(2, T2_);
   T12.SetDiagonalBlock(3, T2_);

   delete T2PC_;
   T2PC_ = new HypreDiagScale(*T2_);

   delete BDP_;
   BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets12_);
   BDP_->SetDiagonalBlock(0, T1Inv_ams_);
   BDP_->SetDiagonalBlock(1, T1Inv_ams_);
   BDP_->SetDiagonalBlock(2, T2Inv_ams_);
   BDP_->SetDiagonalBlock(3, T2Inv_ams_);
   // BDP_->SetDiagonalBlock(2, T2PC_);
   // BDP_->SetDiagonalBlock(3, T2PC_);
   BDP_->owns_blocks = false;

   if ( false && num_procs_ == 1 )
   {
      // Test preconditioner
      HypreParVector tmpa(comm_, 4 * hcurl_glb_size_,
                          part_);
      HypreParVector tmpb(comm_, 4 * hcurl_glb_size_,
                          part_);
      HypreParVector tmpc(comm_, 4 * hcurl_glb_size_,
                          part_);
      tmpa = 0.0;
      tmpc = 0.0;

      int size = 4 * hcurl_glb_size_;
      // double maxDiff = 0.0;
      // double minDiff = MAXFLOAT;
      // double avgDiff = 0.0;

      double delta0 = MAXFLOAT;
      double delta1 = 0.0;

      for (int i=0; i<size; i++)
      {
         tmpa(i) = 1.0;

         FSO_->Mult(tmpa, tmpb);
         T12.Mult(tmpa,tmpc);

         double uAu = tmpa * tmpb;
         double uTu = tmpa * tmpc;

         delta0 = min(delta0, uAu / uTu);
         delta1 = max(delta1, uAu / uTu);

         /*
         // BDP_->Mult(tmpb, tmpc);
         FSOInv_->Mult(tmpb, tmpc);
         tmpc(i) -= 1.0;

         double diff = tmpc.Normlinf();

         maxDiff = max(diff, maxDiff);
         minDiff = min(diff, minDiff);
         avgDiff += diff;
         */

         tmpa(i) = 0.0;
      }
      // avgDiff /= size;

      // cout << "Precond diffs: "
      //    << minDiff << " " << maxDiff << " " << avgDiff << endl;
      cout << "delta1 / delta0: "
           << delta1 << ", " << delta0 << " = " << delta1/delta0 << endl;
   }
   /*
   if ( init_vecs_ != NULL )
   {
     for (int i=0; i<num_init_vecs_; i++)
     {
       delete init_vecs_[i];
     }
     delete init_vecs_; init_vecs_ = NULL;
   }
   */
   if ( init_vecs_ == NULL )
   {
      if ( num_init_vecs_ < nev_ ) { num_init_vecs_ = nev_; }

      HypreParVector tmp(comm_, 2 * (hcurl_glb_size_ +
                                     hdiv_glb_size_),
                         part_);

      init_vecs_ = new HypreParVector*[num_init_vecs_];
      for (int i=0; i<num_init_vecs_; i++)
      {
         init_vecs_[i] = new HypreParVector(comm_,
                                            2 * (hcurl_glb_size_ +
                                                 hdiv_glb_size_),
                                            part_);
         tmp.Randomize(555 + i);
         A12_->Mult(tmp, *init_vecs_[i]);
      }
   }


   delete lobpcg_;
   lobpcg_ = new HypreLOBPCG(comm_);
   lobpcg_->SetNumModes(nev_);
   lobpcg_->SetMaxIter(200);
   // lobpcg_->SetTol(atol_);
   lobpcg_->SetTol(1.0e-2);
   lobpcg_->SetPrintLevel(1);
   lobpcg_->SetMassMatrix(*M12_);
   // lobpcg_->SetPreconditioner(*this->GetPreconditioner());
   // lobpcg_->SetPrecondUsageMode(1);
   // lobpcg_->SetPreconditioner(*FSOInv_);
   lobpcg_->SetPreconditioner(*BDP_);
   // lobpcg_->SetPrecondUsageMode(1);
   // lobpcg_->SetOperator(*this->GetAOperator());
   // lobpcg_->SetSubSpaceProjector(*this->GetSubSpaceProjector());
   lobpcg_->SetOperator(*FSO_);
   // lobpcg_->SetInitialVectors(num_init_vecs_, init_vecs_);

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::SetupBlockSolver"
           << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::SetInitialVectors(int num_vecs,
                                               HypreParVector ** vecs)
{
   num_init_vecs_ = num_vecs;
   init_vecs_     = vecs;
   if ( lobpcg_ )
   {
      lobpcg_->SetInitialVectors(num_vecs, vecs);
   }
}

void MaxwellBlochWaveEquationAMR::Update()
{
   if ( myid_ == 0 )
   { cout << "Entering MaxwellBlochWaveEquationAMR::Update" << endl; }

   this->UpdateFES();

   if ( !currSizes_ )
   {
      currVecs_ = false;
      currA1_   = false;
      currB1_   = false;
      currM1_   = false;
      currM2_   = false;
      currT12_  = false;
      currD12_  = false;
      currZ12_  = false;
      currAMS_  = false;
      currBOpA_ = false;
      currBOpC_ = false;
      currBOpM_ = false;
      currBOpA12_ = false;
      currBOpM12_ = false;
      currBDP_  = false;
      currBPC_  = false;
      currSSP_  = false;
   }
   this->Setup();
   // this->UpdateTmpVectors();
   /*
   if ( myid_ == 0 ) { cout << "Building M2(k)" << endl; }
   ParBilinearForm m2(HDivFESpace_);
   m2.AddDomainIntegrator(new VectorFEMassIntegrator(*aCoef_));
   m2.Assemble();
   m2.Finalize();
   delete M2_;
   M2_ = m2.ParallelAssemble();

   if ( z12_ )
   {
      z12_->Update();
      delete Z12_;
      Z12_ = z12_->ParallelAssemble();
   }
   if ( t12_ )
   {
      t12_->Update();
      delete T12_;
      T12_ = t12_->ParallelAssemble();
   }

   if ( myid_ == 0 ) { cout << "  Forming CMC" << endl; }
   HypreParMatrix * CMC = RAP(M2_, T12_);

   delete A1_;

   if ( fabs(beta_) > 0.0 )
   {
      if ( myid_ == 0 ) { cout << "  Forming ZMZ, CMZ, and ZMC" << endl; }
      HypreParMatrix * ZMZ = RAP(M2_,Z12_);

      HypreParMatrix * CMZ = RAP(T12_, M2_, Z12_);
      HypreParMatrix * ZMC = RAP(Z12_, M2_, T12_);

      *ZMC *= -1.0;
      delete B1_;
      B1_ = ParAdd(CMZ,ZMC);
      delete CMZ;
      delete ZMC;

      *ZMZ *= beta_*beta_;
      A1_ = ParAdd(CMC,ZMZ);
      delete CMC;
      delete ZMZ;
   }
   else
   {
      A1_ = CMC;
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
   A_ = new BlockOperator(block_trueOffsets1_);
   A_->SetDiagonalBlock(0, A1_);
   A_->SetDiagonalBlock(1, A1_);
   if ( fabs(beta_) > 0.0 )
   {
      A_->SetBlock(0, 1, B1_, beta_);
      A_->SetBlock(1, 0, B1_, -beta_);
   }
   A_->owns_blocks = 0;

   if ( myid_ == 0 ) { cout << "Building Block M" << endl; }
   delete M_;
   M_ = new BlockOperator(block_trueOffsets1_);
   M_->SetDiagonalBlock(0, M1_);
   M_->SetDiagonalBlock(1, M1_);
   M_->owns_blocks = 0;

   if ( myid_ == 0 ) { cout << "Building Block C" << endl; }
   delete C_;
   C_ = new BlockOperator(block_trueOffsets2_, block_trueOffsets1_);
   C_->SetDiagonalBlock(0, T12_);
   C_->SetDiagonalBlock(1, T12_);
   if ( fabs(beta_) > 0.0 )
   {
      C_->SetBlock(0, 1, Z12_, beta_);
      C_->SetBlock(1, 0, Z12_, -beta_);
   }
   C_->owns_blocks = 0;

   if ( myid_ == 0 ) { cout << "Building T1Inv" << endl; }
   delete T1Inv_ams_;
   if ( fabs(beta_) < 1.0 )
   {
      T1Inv_ams_ = new HypreAMS(*A1_,HCurlFESpace_);
      T1Inv_ams_->SetSingularProblem();
   }
   else
   {
      T1Inv_ams_ = new HypreAMS(*A1_,HCurlFESpace_);
      T1Inv_ams_->SetSingularProblem();
   }

   if ( myid_ == 0 ) { cout << "Building BDP" << endl; }
   delete BDP_;
   BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets1_);
   BDP_->SetDiagonalBlock(0, T1Inv_ams_);
   BDP_->SetDiagonalBlock(1, T1Inv_ams_);
   BDP_->owns_blocks = 0;

   if ( SubSpaceProj_ ) { SubSpaceProj_->Update(); }

   if ( myid_ == 0 ) { cout << "Building Preconditioner" << endl; }
   delete Precond_;
   Precond_ = new MaxwellBlochWavePrecond(*HCurlFESpace_, *BDP_,
                                          SubSpaceProj_, 0.5);
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
   newACoef_ = false;
   newMCoef_ = false;
   */
   if ( myid_ == 0 )
   { cout << "Leaving MaxwellBlochWaveEquationAMR::Update" << endl; }
}

void
MaxwellBlochWaveEquationAMR::UpdateFES()
{
   H1FESpace_->Update();
   HCurlFESpace_->Update();
   HDivFESpace_->Update();
   L2FESpace_->Update();

   newSizes_ = true;
}

void
MaxwellBlochWaveEquationAMR::UpdateTmpVectors()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::UpdateTmpVectors" << endl;
   }
   /*
   this->SetupSizes();

   delete blkHCurl_; blkHCurl_ = new BlockVector(block_trueOffsets1_);
   delete blkHDiv_;   blkHDiv_ = new BlockVector(block_trueOffsets2_);
   delete vec0_;         vec0_ = new HypreParVector(HCurlFESpace_);

   mCoefGF_->Update();
   aCoefGF_->Update();
   */
   this->Setup();

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::UpdateTmpVectors" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::Solve()
{
   if ( nev_ > 0 )
   {
      if ( fabs(beta_) > 0.0 ||true)
      {
         lobpcg_->Solve();
         vecs_ = lobpcg_->StealEigenvectors();
         cout << "lobpcg done" << endl;

         // DenseMatrix B(nev_);
         DenseMatrix vAv(nev_);
         // DenseMatrix vMv(nev_);
         BlockVector Av(block_trueOffsets12_);
         // BlockVector Mv(block_trueOffsets12_);
         for (int i=0; i<nev_; i++)
         {
            A12_->Mult(*vecs_[i], Av);
            // M12_->Mult(*vecs_[i], Mv);

            for (int j=i; j<nev_; j++)
            {
               vAv(i,j) = vAv(j,i) = *vecs_[j] * Av;
               // vMv(i,j) = vMv(j,i) = *vecs_[j] * Mv;
            }
         }
         //cout << "vAv "; vAv.Print(cout, nev_);
         //cout << "vMv "; vMv.Print(cout, nev_);
         /*
         vMv.Invert();
         Mult(vMv, vAv, B);
         B.Eigensystem(lambdaB_, vecB_);
         */
         vAv.Eigensystem(lambdaB_, vecB_);
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
MaxwellBlochWaveEquationAMR::GetEigenvalues(vector<double> & eigenvalues)
{
   if ( myid_ == 0 )
   { cout << "Entering MaxwellBlochWaveEquationAMR::GetEigenvalues" << endl; }
   if ( lobpcg_ )
   {
      Array<double> eigs;
      lobpcg_->GetEigenvalues(eigs);

      eigenvalues.resize(lambdaB_.Size());
      cout << "Eigen frequencies:" << endl;
      for (int i=0; i<lambdaB_.Size(); i++)
      {
         eigenvalues[i] = lambdaB_[i];
         cout << lambdaB_[i]
              << "\t" << -sqrt(fabs(eigs[i]))+omega_shift_
              << "\t" <<  sqrt(fabs(eigs[i]))+omega_shift_ << endl;
      }
      /*
            eigenvalues.resize(eigs.Size());
            for (int i=0; i<eigs.Size(); i++)
            {
               eigenvalues[i] = eigs[i];
            }
            */
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
   if ( myid_ == 0 )
   { cout << "Entering MaxwellBlochWaveEquationAMR::GetEigenvalues" << endl; }
}

int
MaxwellBlochWaveEquationAMR::numModes(int lattice_type)
{
   int n = 0;

   switch (lattice_type)
   {
      case 1:
      {
         n = 7;
      }
      break;
      case 2:
      {
         n = 13;
      }
      break;
      case 3:
      {
         n = 15;
      }
      break;
      default:
         cout << "Unsupported lattice type:  " << lattice_type << endl;
   };
   return 4 * n;
}

void
MaxwellBlochWaveEquationAMR::GetEigenvalues(/*int nev,*/ const Vector & kappa,
                                                         // vector<HypreParVector*> & init_vecs,
                                                         vector<double> & eigenvalues)
{
   if ( myid_ == 0 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::GetEigenvalues"
           << ", convenience version" << endl;
   }
   this->SetKappa(kappa);

   int nev = numModes(1);

   if ( fabs(beta_) == 0.0 ) { nev += 2; }

   nev /= 2;

   this->SetNumEigs(nev);

   this->Setup();
   // this->SetInitialVectors(nev, &init_vecs[0]);
   this->Solve();
   this->GetEigenvalues(eigenvalues);
   if ( myid_ == 0 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::GetEigenvalues"
           << ", convenience version" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::GetEigenvalues(/*int nev,*/ const Vector & kappa,
                                                         // vector<HypreParVector*> & init_vecs,
                                                         const set<int> & modes, double tol,
                                                         vector<double> & eigenvalues)
{
   if ( myid_ == 0 )
   {
      cout << "Entering MaxwellBlochWaveEquationAMR::GetEigenvalues"
           << ", AMR version" << endl;
   }

   this->SetKappa(kappa);

   int nev = numModes(1);

   if ( fabs(beta_) == 0.0 ) { nev += 2; }

   nev /= 2;

   this->SetNumEigs(nev);
   this->Setup();
   /*
   this->SetInitialVectors(nev, &init_vecs[0]);
   this->Solve();
   this->GetEigenvalues(eigenvalues);
   */
   int it = 0;
   while ( it <= ar_ )
   {
      if ( myid_ == 0 )
      { cout << "AMR loop:  iteration " << it << endl; }

      if ( it > 0 )
      {
         if ( myid_ == 0 )
         { cout << "Estimating errors" << endl; }

         Vector errors(pmesh_->GetNE());
         Vector errors_r(pmesh_->GetNE());
         Vector errors_i(pmesh_->GetNE());

         // Space for the discontinuous (original) flux
         CurlCurlIntegrator flux_integrator(*aCoef_);
         RT_FECollection flux_fec(order_-1, pmesh_->SpaceDimension());
         ParFiniteElementSpace flux_fes(pmesh_, &flux_fec);

         // Space for the smoothed (conforming) flux
         ND_FECollection smooth_flux_fec(order_, pmesh_->Dimension());
         ParFiniteElementSpace smooth_flux_fes(pmesh_, &smooth_flux_fec);

         ParGridFunction er(HCurlFESpace_);
         ParGridFunction ei(HCurlFESpace_);

         HypreParVector Er(HCurlFESpace_->GetComm(),
                           HCurlFESpace_->GlobalTrueVSize(),
                           NULL,
                           HCurlFESpace_->GetTrueDofOffsets());
         HypreParVector Ei(HCurlFESpace_->GetComm(),
                           HCurlFESpace_->GlobalTrueVSize(),
                           NULL,
                           HCurlFESpace_->GetTrueDofOffsets());
         /*
              ParGridFunction hr(HCurlFESpace_);
              ParGridFunction hi(HCurlFESpace_);
         */
         HypreParVector Hr(HCurlFESpace_->GetComm(),
                           HCurlFESpace_->GlobalTrueVSize(),
                           NULL,
                           HCurlFESpace_->GetTrueDofOffsets());
         HypreParVector Hi(HCurlFESpace_->GetComm(),
                           HCurlFESpace_->GlobalTrueVSize(),
                           NULL,
                           HCurlFESpace_->GetTrueDofOffsets());

         double norm_p = 1;
         errors = 0.0;

         set<int>::const_iterator sit;
         for (sit=modes.begin(); sit!=modes.end(); sit++)
         {
            // convert eigenvector from HypreParVector to ParGridFunction
            this->GetEigenvectorE(*sit, Er, Ei);
            er = Er;
            ei = Ei;

            L2ZZErrorEstimator(flux_integrator, er,
                               smooth_flux_fes, flux_fes, errors_r, norm_p);
            L2ZZErrorEstimator(flux_integrator, ei,
                               smooth_flux_fes, flux_fes, errors_i, norm_p);

            for (int j=0; j<errors.Size(); j++)
            {
               errors[j] += pow(errors_r[j], norm_p) + pow(errors_i[j], norm_p);
            }
         }
         for (int j=0; j<errors.Size(); j++)
         {
            errors[j] = pow(errors[j], 1.0/norm_p);
         }

         double local_max_err = errors.Max();
         double global_max_err;
         MPI_Allreduce(&local_max_err, &global_max_err, 1,
                       MPI_DOUBLE, MPI_MAX, pmesh_->GetComm());

         if ( myid_ == 0 ) { cout << "Maximum error: " << global_max_err << endl; }

         // Refine the elements whose error is larger than a fraction of the
         // maximum element error.
         const double frac = 0.5;
         double threshold = frac * global_max_err;
         if ( myid_ == 0 )
         {
            cout << "AMR iteration " << it+1 << ", refining from "
                 << pmesh_->GetNE() << " elements" << flush;
         }
         pmesh_->RefineByError(errors, threshold, 0);
         currSizes_ = false;
         if ( myid_ == 0 )
         {
            cout << " to " << pmesh_->GetNE() << "." << endl;
         }

         num_init_vecs_ = nev_;

         if ( init_er_ == NULL )
         {
            init_er_ = new ParGridFunction*[num_init_vecs_];
            for ( int i=0; i<num_init_vecs_; i++)
            {
               init_er_[i] = new ParGridFunction(HCurlFESpace_);
            }
         }
         if ( init_ei_ == NULL )
         {
            init_ei_ = new ParGridFunction*[num_init_vecs_];
            for ( int i=0; i<num_init_vecs_; i++)
            {
               init_ei_[i] = new ParGridFunction(HCurlFESpace_);
            }
         }
         if ( init_hr_ == NULL )
         {
            init_hr_ = new ParGridFunction*[num_init_vecs_];
            for ( int i=0; i<num_init_vecs_; i++)
            {
               init_hr_[i] = new ParGridFunction(HCurlFESpace_);
            }
         }
         if ( init_hi_ == NULL )
         {
            init_hi_ = new ParGridFunction*[num_init_vecs_];
            for ( int i=0; i<num_init_vecs_; i++)
            {
               init_hi_[i] = new ParGridFunction(HCurlFESpace_);
            }
         }
         if ( myid_ == 0 && logging_ > 1 )
         {
            cout << "Caching eigenvectors as GridFunctions" << endl;
         }
         for (int i=0; i<num_init_vecs_; i++)
         {
            this->GetEigenvector(i, Er, Ei, Hr, Hi);
            *init_er_[i] = Er;
            *init_ei_[i] = Ei;
            *init_hr_[i] = Hr;
            *init_hi_[i] = Hi;
         }

         this->Update();

         for (int i=0; i<num_init_vecs_; i++)
         {
            delete init_vecs_[i];
            init_vecs_[i] = new HypreParVector(comm_,
                                               4*hcurl_glb_size_, part_);
         }
         if ( myid_ == 0 && logging_ > 1 )
         {
            cout << "Projecting coarse GridFunctions onto refined vectors"
                 << endl;
         }
         for (int i=0; i<num_init_vecs_; i++)
         {
            init_er_[i]->Update();
            init_ei_[i]->Update();
            init_hr_[i]->Update();
            init_hi_[i]->Update();

            Er.SetDataAndSize(&(*init_vecs_[i])(0 * hcurl_loc_size_),
                              hcurl_loc_size_);
            Ei.SetDataAndSize(&(*init_vecs_[i])(1 * hcurl_loc_size_),
                              hcurl_loc_size_);
            Hr.SetDataAndSize(&(*init_vecs_[i])(2 * hcurl_loc_size_),
                              hcurl_loc_size_);
            Hi.SetDataAndSize(&(*init_vecs_[i])(3 * hcurl_loc_size_),
                              hcurl_loc_size_);

            init_er_[i]->ParallelAssemble(Er);
            init_ei_[i]->ParallelAssemble(Ei);
            init_hr_[i]->ParallelAssemble(Hr);
            init_hi_[i]->ParallelAssemble(Hi);
         }
         /*
         if ( lobpcg_ )
         {
           lobpcg_->SetInitialVectors(num_init_vecs_, init_vecs_);
         }
         */
         /*
              if ( myid_ == 0 )
              { cout << "Updating initial vectors" << endl; }

              HypreParVector Wr(this->GetHCurlFESpace()->GetComm(),
                                this->GetHCurlFESpace()->GlobalTrueVSize(),
                                NULL,
                                this->GetHCurlFESpace()->GetTrueDofOffsets());
              HypreParVector Wi(this->GetHCurlFESpace()->GetComm(),
                                this->GetHCurlFESpace()->GlobalTrueVSize(),
                                NULL,
                                this->GetHCurlFESpace()->GetTrueDofOffsets());

              for (int i=0; i<num_init_vecs_; i++)
              {
                 vr[i]->Update();
                 vi[i]->Update();

                 init_vecs_[i] = new HypreParVector(comm_, 2*hcurl_glb_size_, part_);

                 Wr.SetData(&(*init_vecs_[i])(0));
                 Wi.SetData(&(*init_vecs_[i])(hcurl_loc_size_));

                 vr[i]->ParallelAssemble(Wr);
                 vi[i]->ParallelAssemble(Wi);
              }
         */
      }
      /*
      if ( it == 0 )
      {
         if ( myid_ == 0 )
         { cout << "Setting initial vectors" << endl; }
         this->SetInitialVectors(nev, &init_vecs[0]);
      }
      else
      {
         if ( myid_ == 0 )
         { cout << "Setting new initial vectors" << endl; }
         this->SetInitialVectors(nev, &init_vecs_[0]);
      }
      */
      /*
      if ( SubSpaceProj_ )
      {
      if ( myid_ == 0 )
      { cout << "Testing Discrete Deriv Opertors" << endl; }

      SubSpaceProj_->GetBlockVector()->Randomize();
      double nrm0 = SubSpaceProj_->GetBlockVector()->Norml2();
      if ( myid_ == 0 )
        { cout << "Norm of random vector: " << nrm0 << endl; }
      SubSpaceProj_->GetGOperator()->Mult(*SubSpaceProj_->GetBlockVector(),
                     *blkHCurl_);
      double nrm1 = blkHCurl_->Norml2();
      if ( myid_ == 0 )
        { cout << "Norm of Gradient of random vector: " << nrm1 << endl; }
      C_->Mult(*blkHCurl_, *blkHDiv_);
      double nrm2 = blkHDiv_->Norml2();
      if ( myid_ == 0 )
        { cout << "Norm of Curl of Gradient of random vector: "
         << nrm2 << endl; }

      HypreParMatrix *Z12Z01 = ParMult(Z12_,
                  dynamic_cast<HypreParMatrix*>(&SubSpaceProj_->GetGOperator()->GetBlock(0,1)));
      Z12Z01->Print("Z12Z01.mat");
      delete Z12Z01;

      HypreParMatrix* Z12T01 = ParMult(Z12_,
                  dynamic_cast<HypreParMatrix*>(&SubSpaceProj_->GetGOperator()->GetBlock(0,0)));
      Z12T01->Print("Z12T01.mat");

      HypreParMatrix* T12Z01 = ParMult(T12_,
                  dynamic_cast<HypreParMatrix*>(&SubSpaceProj_->GetGOperator()->GetBlock(0,1)));
      T12Z01->Print("T12Z01.mat");

      HypreParMatrix * D02 = ParAdd(Z12T01, T12Z01);
      //D02->Threshold(1.0e-14);
      D02->Print("D02.mat");
      delete D02;
      delete Z12T01;
      delete T12Z01;
           }
           */
      if ( lobpcg_ && init_vecs_ )
      {
         if ( myid_ == 0 && logging_ > 1 )
         {
            cout << "Setting initial vectors in LOBPCG" << endl;
         }
         lobpcg_->SetInitialVectors(num_init_vecs_, &init_vecs_[0]);
      }
      if ( myid_ == 0 )
      { cout << "Calling Solve" << endl; }
      this->Solve();
      if ( myid_ == 0 )
      { cout << "Calling GetEigenvalues" << endl; }
      this->GetEigenvalues(eigenvalues);

      it++;
   }
   if ( myid_ == 0 )
   {
      cout << "Leaving MaxwellBlochWaveEquationAMR::GetEigenvalues"
           << ", AMR version" << endl;
   }
}

void
MaxwellBlochWaveEquationAMR::GetEigenvector(unsigned int i,
                                            HypreParVector & Er,
                                            HypreParVector & Ei,
                                            HypreParVector & Hr,
                                            HypreParVector & Hi)
{
   this->GetEigenvectorE(i, Er, Ei);
   this->GetEigenvectorH(i, Hr, Hi);
}

void
MaxwellBlochWaveEquationAMR::GetEigenvectorE(unsigned int i,
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
MaxwellBlochWaveEquationAMR::GetEigenvectorH(unsigned int i,
                                             HypreParVector & Hr,
                                             HypreParVector & Hi)
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
         /*
          if ( i%2 == 0 )
               {
                  data = (double*)ame_->GetEigenvector(i/2);
               }
               else
               {
                  data = (double*)ame_->GetEigenvector((i-1)/2);
               }
         */
      }
   }

   if ( lobpcg_ )
   {
      Hr.SetData(&data[2*hcurl_loc_size_]);
      Hi.SetData(&data[3*hcurl_loc_size_]);
   }
   else if ( ame_ )
   {
      /*
       if ( i%2 == 0 )
       {
          Hr.SetData(&data[0]);
          Hi.SetData(vec0_->GetData());
       }
       else
       {
          Hr.SetData(vec0_->GetData());
          Hi.SetData(&data[0]);
       }
      */
   }
}
/*
void
MaxwellBlochWaveEquationAMR::GetEigenvectorB(unsigned int i,
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
         t12_->Mult(ame_->GetEigenvector(i/2),blkHDiv_->GetBlock(0));
      }
      else
      {
         t12_->Mult(ame_->GetEigenvector((i-1)/2),blkHDiv_->GetBlock(1));
         blkHDiv_->GetBlock(0) = 0.0;
      }
   }

   if ( eigenvalues[i] != 0.0 ) { *blkHDiv_ /= sqrt(fabs(eigenvalues[i])); }

   double * data = (double*)*blkHDiv_;
   Bi.SetData(&data[0]);
   Br.SetData(&data[hdiv_loc_size_]); Br *= -1.0;
}
*/
void
MaxwellBlochWaveEquationAMR::GetInnerProducts(DenseMatrix & mat)
{
   mat.SetSize(nev_, num_init_vecs_); mat = 0.0;
   /*
   if ( M_ != NULL )
   {
      HypreParVector Mv(comm_, init_vecs_[0]->GlobalSize(),
                        blkHCurl_->GetData(),
                        init_vecs_[0]->Partitioning());
      for (int i=0; i<nev_; i++)
      {
         if ( vecs_[i] != NULL )
         {
            M_->Mult(*vecs_[i], *blkHCurl_);
         }
         for (int j=0; j<num_init_vecs_; j++)
         {
            mat(i,j) = InnerProduct(Mv, *init_vecs_[j]);
         }
      }
   }
   */
}

void
MaxwellBlochWaveEquationAMR::GetFourierCoefficients(HypreParVector & Vr,
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
MaxwellBlochWaveEquationAMR::GetFieldAverages(unsigned int i,
                                              Vector & Er, Vector & Ei,
                                              Vector & Br, Vector & Bi,
                                              Vector & Dr, Vector & Di,
                                              Vector & Hr, Vector & Hi)
{
   if ( fourierHCurl_ == NULL)
   {
      MFEM_ASSERT(bravais_ != NULL, "MaxwellBlochWaveEquationAMR: "
                  "Field averages cannot be computed "
                  "without a BravaisLattice object.");

      fourierHCurl_ = new HCurlFourierSeries(*bravais_, *HCurlFESpace_);
   }

   vector<double> eigs;
   this->GetEigenvalues(eigs);

   double omega = (eigs[i]>0.0)?sqrt(eigs[i]):0.0;

   Vector arr(3), ari(3), air(3), aii(3);

   fourierHCurl_->SetMode(0,0,0);

   HypreParVector  ParEr(HCurlFESpace_->GetComm(),
                         HCurlFESpace_->GlobalTrueVSize(),
                         NULL,
                         HCurlFESpace_->GetTrueDofOffsets());

   HypreParVector  ParEi(HCurlFESpace_->GetComm(),
                         HCurlFESpace_->GlobalTrueVSize(),
                         NULL,
                         HCurlFESpace_->GetTrueDofOffsets());

   this->GetEigenvectorE(i, ParEr, ParEi);

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

   ConstantCoefficient * aConst = dynamic_cast<ConstantCoefficient*>(aCoef_);
   if ( aConst != NULL )
   {
      // The coefficient mu is constant
      double muInv = aConst->constant;

      Hr = Br; Hr *= muInv;
      Hi = Bi; Hi *= muInv;
   }
   else
   {
      /*
      fourierHCurl_->GetCoefficient(, arr, ari);
      fourierHCurl_->GetCoefficient(, air, aii);

      Hr = arr; Hr -= aii;
      Hi = air; Hi += ari;
      */
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
}

void
MaxwellBlochWaveEquationAMR::WriteVisitFields(const string & prefix,
                                              const string & label)
{
   ParGridFunction Er(this->GetHCurlFESpace());
   ParGridFunction Ei(this->GetHCurlFESpace());

   ParGridFunction Hr(this->GetHCurlFESpace());
   ParGridFunction Hi(this->GetHCurlFESpace());

   HypreParVector ErVec(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());
   HypreParVector EiVec(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());

   HypreParVector HrVec(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());
   HypreParVector HiVec(this->GetHCurlFESpace()->GetComm(),
                        this->GetHCurlFESpace()->GlobalTrueVSize(),
                        NULL,
                        this->GetHCurlFESpace()->GetTrueDofOffsets());

   VisItDataCollection visit_dc(label.c_str(), pmesh_);
   visit_dc.SetPrefixPath(prefix.c_str());

   if ( dynamic_cast<GridFunctionCoefficient*>(mCoef_) )
   {
      GridFunctionCoefficient * gfc =
         dynamic_cast<GridFunctionCoefficient*>(mCoef_);
      visit_dc.RegisterField("epsilon", gfc->GetGridFunction() );
   }
   if ( dynamic_cast<GridFunctionCoefficient*>(aCoef_) )
   {
      GridFunctionCoefficient * gfc =
         dynamic_cast<GridFunctionCoefficient*>(aCoef_);
      visit_dc.RegisterField("muInv", gfc->GetGridFunction() );
   }

   visit_dc.RegisterField("E_r", &Er);
   visit_dc.RegisterField("E_i", &Ei);
   visit_dc.RegisterField("H_r", &Hr);
   visit_dc.RegisterField("H_i", &Hi);

   vector<double> eigenvalues;
   this->GetEigenvalues(eigenvalues);

   for (int i=0; i<nev_; i++)
   {
      this->GetEigenvector(i, ErVec, EiVec, HrVec, HiVec);

      Er = ErVec;
      Ei = EiVec;

      Hr = HrVec;
      Hi = HiVec;

      visit_dc.SetCycle(i+1);
      visit_dc.SetTime(eigenvalues[i]);
      visit_dc.Save();
   }
}

void
MaxwellBlochWaveEquationAMR::DisplayToGLVis(socketstream & a_sock,
                                            socketstream & m_sock,
                                            char vishost[], int visport,
                                            int Wx, int Wy, int Ww, int Wh,
                                            int offx, int offy)
{
   mCoefGF_->ProjectCoefficient(*mCoef_);
   aCoefGF_->ProjectCoefficient(*aCoef_);

   VisualizeField(m_sock, vishost, visport, *mCoefGF_,
                  "Mass Coefficient", Wx, Wy, Ww, Wh);
   VisualizeField(a_sock, vishost, visport, *aCoefGF_,
                  "Stiffness Coefficient", Wx, Wy+offy, Ww, Wh);
}

MaxwellBlochWaveEquationAMR::MaxwellBlochWavePrecond::
MaxwellBlochWavePrecond(ParFiniteElementSpace & HCurlFESpace,
                        BlockDiagonalPreconditioner & BDP,
                        Operator * subSpaceProj,
                        double w)
   : Solver(2*HCurlFESpace.GlobalTrueVSize()),
     comm_(HCurlFESpace.GetComm()), myid_(0), numProcs_(-1), part_(NULL),
     HCurlFESpace_(&HCurlFESpace), BDP_(&BDP),
     subSpaceProj_(subSpaceProj), u_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_rank(comm_, &myid_);
   numProcs_ = HCurlFESpace.GetNRanks();

   if ( myid_ == 0 ) { cout << "MaxwellBlochWavePrecond" << endl; }

   if (HYPRE_AssumedPartitionCheck())
   {
      part_ = new HYPRE_Int[2];
   }
   else
   {
      part_ = new HYPRE_Int[numProcs_+1];
   }

   this->Update();
}

MaxwellBlochWaveEquationAMR::
MaxwellBlochWavePrecond::~MaxwellBlochWavePrecond()
{
   delete u_;
   delete part_;
}

void
MaxwellBlochWaveEquationAMR::MaxwellBlochWavePrecond::Update()
{
   int locSize = 2*HCurlFESpace_->TrueVSize();
   int glbSize = 0;

   if (HYPRE_AssumedPartitionCheck())
   {
      MPI_Scan(&locSize, &part_[1], 1, HYPRE_MPI_INT, MPI_SUM, comm_);

      part_[0] = part_[1] - locSize;

      MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm_);
   }
   else
   {
      MPI_Allgather(&locSize, 1, MPI_INT,
                    &part_[1], 1, HYPRE_MPI_INT, comm_);

      part_[0] = 0;
      for (int i=0; i<numProcs_; i++)
      {
         part_[i+1] += part_[i];
      }

      glbSize = part_[numProcs_];
   }
   Operator::height = Operator::width = glbSize;

   delete u_;
   u_ = new HypreParVector(comm_, glbSize, part_);
}

void
MaxwellBlochWaveEquationAMR::
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
MaxwellBlochWaveEquationAMR::
MaxwellBlochWavePrecond::SetOperator(const Operator & A)
{
   A_ = &A;
}

FoldedSpectrumOperator::FoldedSpectrumOperator(Operator & A, Operator & M,
                                               IterativeSolver & MInv,
                                               double lambda0)
   : Operator(A.Height()),
     lambda0_(lambda0),
     A_(&A),
     M_(&M),
     MInv_(&MInv),
     z0_(A.Height()),
     z1_(A.Height())
{
   z0_ = 0.0;
   z1_ = 0.0;
}

void
FoldedSpectrumOperator::Mult(const Vector &x, Vector &y) const
{
   // Apply y = (A - lambda0 M) x
   M_->Mult(x, y);
   y *= -lambda0_;
   A_->Mult(x, z0_);
   y += z0_;

   // z1 = M^{-1} (A - lambda0 M) x
   MInv_->Mult(y, z1_);
   // cout << "MInv its: " << MInv_->GetNumIterations() << endl;

   // y = (A - lambda0 M) M^{-1} (A - lambda0 M) x
   M_->Mult(z1_, y);
   y *= -lambda0_;
   A_->Mult(z1_, z0_);
   y += z0_;
}

MaxwellBlochWaveProjectorAMR::
MaxwellBlochWaveProjectorAMR(ParFiniteElementSpace & HCurlFESpace,
                             ParFiniteElementSpace & H1FESpace,
                             BlockOperator & M,
                             double beta, const Vector & zeta,
                             int logging)
   : Operator(2*HCurlFESpace.GlobalTrueVSize()),
     logging_(logging),
     newSizes_(true),
     newM_(true),
     newBeta_(true),
     newZeta_(true),
     currSizes_(false),
     currVecs_(false),
     currA0_(false),
     currB0_(false),
     currT01_(false),
     currZ01_(false),
     currBOpA_(false),
     currBOpG_(false),
     HCurlFESpace_(&HCurlFESpace),
     H1FESpace_(&H1FESpace),
     beta_(beta),
     zeta_(zeta),
     T01_(NULL),
     Z01_(NULL),
     A0_(NULL),
     B0_(NULL),
     // DKZT_(NULL),
     GMG_(NULL),
     ZMZ_(NULL),
     GMZ_(NULL),
     ZMG_(NULL),
     // amg_cos_(NULL),
     t01_(NULL),
     z01_(NULL),
     A_(NULL),
     M_(&M),
     G_(NULL),
     AInv_(NULL),
     // urDummy_(NULL),
     // uiDummy_(NULL),
     // vrDummy_(NULL),
     // viDummy_(NULL),
     u_(NULL),
     v_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_rank(H1FESpace.GetParMesh()->GetComm(), &myid_);

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Constructing MaxwellBlochWaveProjectorAMR" << endl;
   }

   this->Setup();
   // this->Update();
   /*
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
   */
   if ( myid_ == 0 && logging_ > 1 ) { cout << "done" << endl; }
}

MaxwellBlochWaveProjectorAMR::~MaxwellBlochWaveProjectorAMR()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Destroying MaxwellBlochWaveProjectorAMR" << endl;
   }
   // delete urDummy_; delete uiDummy_; delete vrDummy_; delete viDummy_;
   delete u_; delete v_;
   delete T01_;
   delete Z01_;
   if ( A0_ != GMG_ ) { delete A0_; }
   delete B0_;
   // delete DKZT_;
   delete GMG_;
   delete ZMZ_;
   delete GMZ_;
   delete ZMG_;

   delete z01_;
   delete t01_;
   delete A_;
   delete G_;
   // delete amg_cos_;
   delete AInv_;

   if ( myid_ == 0 && logging_ > 1 ) { cout << "done" << endl; }
}

void
MaxwellBlochWaveProjectorAMR::SetM(BlockOperator & M)
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Setting new M pointer" << endl;
   }
   M_ = &M; newM_ = true;
   currA0_   = false;
   currB0_   = false;
   currBOpA_ = false;
}

void
MaxwellBlochWaveProjectorAMR::SetBeta(double beta)
{
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Setting new Beta" << endl;
   }
   beta_ = beta; newBeta_ = true;
   currA0_   = false;
   currB0_   = false;
   currBOpA_ = false;
   currBOpG_ = false;
}

void
MaxwellBlochWaveProjectorAMR::SetZeta(const Vector & zeta)
{
   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << "Setting new Zeta" << endl;
   }
   zeta_ = zeta; newZeta_ = true;
   currA0_   = false;
   currB0_   = false;
   currZ01_  = false;
   currBOpA_ = false;
   currBOpG_ = false;
}

void
MaxwellBlochWaveProjectorAMR::Setup()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::Setup" << endl;
   }
   if ( !currVecs_ ) { this->SetupTmpVectors(); }
   if ( !currBOpG_ ) { this->SetupBlockOperatorG(); }
   this->SetupSolver();

   newSizes_ = false;
   newM_     = false;
   newBeta_  = false;
   newZeta_  = false;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::Setup" << endl;
   }
}

void
MaxwellBlochWaveProjectorAMR::SetupTmpVectors()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::SetupTmpVectors" << endl;
   }
   if ( !currSizes_ ) { this->SetupSizes(); }

   if ( !currVecs_ )
   {
      delete u_; u_ = new BlockVector(block_trueOffsets0_);
      delete v_; v_ = new BlockVector(block_trueOffsets0_);

      currVecs_ = true;
   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::SetupTmpVectors" << endl;
   }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateTmpVectors()
{
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Entering MaxwellBlochWaveProjectorAMR::UpdateTmpVectors" << endl;
 }
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Leaving MaxwellBlochWaveProjectorAMR::UpdateTmpVectors" << endl;
 }
}
*/
void
MaxwellBlochWaveProjectorAMR::SetupBlockOperatorG()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::SetupBlockOperatorG"
           << endl;
   }
   if ( !currSizes_ ) { this->SetupSizes(); }
   if ( !currT01_   ) { this->SetupT01();   }
   if ( !currZ01_   ) { this->SetupZ01();   }

   if ( newSizes_ )
   {
      delete G_;
      G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
      G_->SetBlock(0,0,T01_);
      G_->SetBlock(1,1,T01_);
      G_->owns_blocks = 0;
   }
   if ( newSizes_ || ( fabs(beta_) > 0.0 && (newBeta_ || newZeta_) ) )
   {
      G_->SetBlock(0,1,Z01_, beta_);
      G_->SetBlock(1,0,Z01_,-beta_);
   }
   currBOpG_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::SetupBlockOperatorG"
           << endl;
   }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateBlockOperatorG()
{
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Entering MaxwellBlochWaveProjectorAMR::UpdateBlockOperatorG"
         << endl;
 }
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Leaving MaxwellBlochWaveProjectorAMR::UpdateBlockOperatorG"
         << endl;
 }
}
*/
void
MaxwellBlochWaveProjectorAMR::SetupBlockOperatorA()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::SetupBlockOperatorA"
           << endl;
   }
   if ( !currSizes_ ) { this->SetupSizes(); }
   if ( !currB0_    ) { this->SetupB0();    }
   if ( !currA0_    ) { this->SetupA0();    }

   if ( newSizes_ )
   {
      delete A_;
      A_ = new BlockOperator(block_trueOffsets0_);
      A_->owns_blocks = 0;
   }
   if ( newSizes_ || newM_ || newBeta_ || newZeta_ )
   {
      A_->SetDiagonalBlock(0, A0_, 1.0);
      A_->SetDiagonalBlock(1, A0_, 1.0);

      if ( fabs(beta_) > 0.0 )
      {
         A_->SetBlock(0, 1, B0_, -beta_);
         A_->SetBlock(1, 0, B0_,  beta_);
      }
   }
   currBOpA_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::SetupBlockOperatorA"
           << endl;
   }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateBlockOperatorA()
{
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Entering MaxwellBlochWaveProjectorAMR::UpdateBlockOperatorA"
         << endl;
 }
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Leaving MaxwellBlochWaveProjectorAMR::UpdateBlockOperatorA"
         << endl;
 }
}
*/
void
MaxwellBlochWaveProjectorAMR::SetupSolver()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::SetupSolver" << endl;
   }
   if ( !currBOpA_ ) { this->SetupBlockOperatorA(); }

   if ( AInv_ == NULL )
   {
      if ( myid_ > 0 && logging_ > 1 )
      {
         cout << "  Creating MINRES Solver" << endl;
      }
      AInv_ = new MINRESSolver(H1FESpace_->GetComm());
      AInv_->SetRelTol(1e-13);
      AInv_->SetMaxIter(3000);
      AInv_->SetPrintLevel(0);
   }

   if ( newM_ || newBeta_ || newZeta_ || newSizes_ )
   {
      AInv_->SetOperator(*A_);
   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::SetupSolver" << endl;
   }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateSolver()
{
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Entering MaxwellBlochWaveProjectorAMR::UpdateSolver" << endl;
 }
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Leaving MaxwellBlochWaveProjectorAMR::UpdateSolver" << endl;
 }
}
*/
void
MaxwellBlochWaveProjectorAMR::SetupSizes()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::SetupSizes" << endl;
   }
   if ( !currSizes_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Setting new sizes" << endl;
      }
      if ( block_trueOffsets0_.Size() < 3 )
      {
         block_trueOffsets0_.SetSize(3);
         block_trueOffsets1_.SetSize(3);
      }

      block_trueOffsets0_[0] = 0;
      block_trueOffsets0_[1] = H1FESpace_->TrueVSize();
      block_trueOffsets0_[2] = H1FESpace_->TrueVSize();
      block_trueOffsets0_.PartialSum();

      block_trueOffsets1_[0] = 0;
      block_trueOffsets1_[1] = HCurlFESpace_->TrueVSize();
      block_trueOffsets1_[2] = HCurlFESpace_->TrueVSize();
      block_trueOffsets1_.PartialSum();

      locSize_ = HCurlFESpace_->TrueVSize();

      Operator::height = Operator::width = 2*HCurlFESpace_->GlobalTrueVSize();

      currSizes_ = true;
      currVecs_  = false;
      currT01_   = false;
      currZ01_   = false;
      currA0_    = false;
      currB0_    = false;
      currBOpA_  = false;
      currBOpG_  = false;
   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::SetupSizes" << endl;
   }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateSizes()
{
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Entering MaxwellBlochWaveProjectorAMR::UpdateSizes" << endl;
 }
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Leaving MaxwellBlochWaveProjectorAMR::UpdateSizes" << endl;
 }
}
*/
void
MaxwellBlochWaveProjectorAMR::SetupA0()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::SetupA0" << endl;
   }
   if ( !currT01_ ) { this->SetupT01(); }
   if ( !currZ01_ ) { this->SetupZ01(); }

   if ( ! currA0_ )
   {
      if ( myid_ == 0 && logging_ > 1 ) { cout << "  Forming GMG" << endl; }

      HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));

      if ( A0_ != GMG_ ) { delete A0_; }
      delete GMG_; GMG_ = RAP(M1,T01_);

      if ( fabs(beta_) > 0.0 )
      {
         if ( newSizes_ || newM_ || newZeta_ )
         {
            if ( myid_ == 0 && logging_ > 1 )
            {
               cout << "  Forming ZMZ" << endl;
            }
            HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));
            delete ZMZ_; ZMZ_ = RAP(M1,Z01_);
         }

         if ( newSizes_ || newM_ || newZeta_ || newBeta_ )
         {
            *ZMZ_ *= beta_*beta_;
            A0_ = ParAdd(GMG_, ZMZ_);
            *ZMZ_ *= 1.0 / (beta_*beta_);
         }
      }
      else
      {
         A0_ = GMG_;
      }
   }
   currA0_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::SetupA0" << endl;
   }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateA0()
{
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Entering MaxwellBlochWaveProjectorAMR::UpdateA0" << endl;
 }
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Leaving MaxwellBlochWaveProjectorAMR::UpdateA0" << endl;
 }
}
*/
void
MaxwellBlochWaveProjectorAMR::SetupB0()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::SetupB0" << endl;
   }
   if ( !currT01_ ) { this->SetupT01(); }
   if ( !currZ01_ ) { this->SetupZ01(); }

   if ( newSizes_ || newM_ || newZeta_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Forming GMZ and ZMG" << endl;
      }
      HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));
      delete GMZ_; GMZ_ = RAP(T01_, M1, Z01_);
      delete ZMG_; ZMG_ = RAP(Z01_, M1, T01_);
      *GMZ_ *= -1.0;
      delete B0_; B0_ = ParAdd(GMZ_, ZMG_);
   }
   currB0_ = true;

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::SetupB0" << endl;
   }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateB0()
{
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Entering MaxwellBlochWaveProjectorAMR::UpdateB0" << endl;
 }
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Leaving MaxwellBlochWaveProjectorAMR::UpdateB0" << endl;
 }
}
*/
void
MaxwellBlochWaveProjectorAMR::SetupT01()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::SetupT01" << endl;
   }
   if ( t01_ == NULL )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building Grad operator" << endl;
      }
      t01_ = new ParDiscreteGradOperator(H1FESpace_,HCurlFESpace_);
   }
   else
   {
      t01_->Update();
   }

   if ( ! currT01_ )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Assembling Gradient operator" << endl;
      }
      t01_->Assemble();
      t01_->Finalize();
      delete T01_; T01_ = t01_->ParallelAssemble();
      T01_->Print("T01.mat");
      currT01_ = true;
   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::SetupT01" << endl;
   }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateT01()
{
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Entering MaxwellBlochWaveProjectorAMR::UpdateT01" << endl;
 }
 // t01_->Update();

 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Leaving MaxwellBlochWaveProjectorAMR::UpdateT01" << endl;
 }
}
*/
void
MaxwellBlochWaveProjectorAMR::SetupZ01()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::SetupZ01" << endl;
   }

   if ( ! currZ01_ && fabs(beta_) > 0.0 )
   {
      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Building zeta times operator" << endl;
      }
      if ( newZeta_ )
      {
         delete z01_;
         z01_ = new ParDiscreteVectorProductOperator(H1FESpace_,
                                                     HCurlFESpace_,zeta_);
      }
      else
      {
         z01_->Update();
      }

      if ( myid_ == 0 && logging_ > 1 )
      {
         cout << "  Assembling zeta times operator" << endl;
      }
      z01_->Assemble();
      z01_->Finalize();
      delete Z01_; Z01_ = z01_->ParallelAssemble();
      Z01_->Print("Z01.mat");
      currZ01_ = true;
   }

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::SetupZ01" << endl;
   }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateZ01()
{
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Entering MaxwellBlochWaveProjectorAMR::UpdateZ01" << endl;
 }
 if ( myid_ == 0 && logging_ > 0 )
 {
    cout << "Leaving MaxwellBlochWaveProjectorAMR::UpdateZ01" << endl;
 }
}
*/
void
MaxwellBlochWaveProjectorAMR::OldSetup()
{
   MFEM_ASSERT(false, "Deprecated code");
   if ( myid_ == 0 )
   {
      cout << "Setting up MaxwellBlochWaveProjector" << endl;
   }

   if ( t01_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Grad operator" << endl; }
      t01_ = new ParDiscreteGradOperator(H1FESpace_,HCurlFESpace_);
      t01_->Assemble();
      t01_->Finalize();
      // T01_ = Grad_->ParallelAssemble();
   }

   if ( newZeta_ )
   {
      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Building zeta times operator" << endl; }
         z01_ = new ParDiscreteVectorProductOperator(H1FESpace_,
                                                     HCurlFESpace_,zeta_);
         z01_->Assemble();
         z01_->Finalize();
         // Z01_ = Zeta_->ParallelAssemble();
      }
   }
   /*
   if ( G_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Block G" << endl; }
      G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
   }
   G_->SetBlock(0,0,T01_);
   G_->SetBlock(1,1,T01_);
   if ( fabs(beta_) > 0.0 )
   {
      G_->SetBlock(0,1,Z01_, beta_);
      G_->SetBlock(1,0,Z01_,-beta_);
   }
   G_->owns_blocks = 0;
   */
   if ( newBeta_ || newZeta_ )
   {
      if ( myid_ == 0 ) { cout << "  Forming GMG" << endl; }
      HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));
      HypreParMatrix * GMG = RAP(M1,T01_);

      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "  Forming ZMZ, GMZ, and ZMG" << endl; }
         HypreParMatrix * ZMZ = RAP(M1,Z01_);

         HypreParMatrix * GMZ = RAP(T01_, M1, Z01_);
         HypreParMatrix * ZMG = RAP(Z01_, M1, T01_);
         *GMZ *= -1.0;
         B0_ = ParAdd(GMZ,ZMG);

         delete GMZ;
         delete ZMG;

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

   if ( A_ == NULL )
   {
      if ( myid_ > 0 ) { cout << "Building Block A" << endl; }
      A_ = new BlockOperator(block_trueOffsets0_);
   }
   A_->SetDiagonalBlock(0,A0_,1.0);
   A_->SetDiagonalBlock(1,A0_,1.0);
   if ( fabs(beta_) > 0.0 )
   {
      A_->SetBlock(0,1,B0_,-beta_);
      A_->SetBlock(1,0,B0_, beta_);
   }
   A_->owns_blocks = 0;

   if ( myid_ > 0 ) { cout << "Creating MINRES Solver" << endl; }
   delete AInv_;
   AInv_ = new MINRESSolver(H1FESpace_->GetComm());
   AInv_->SetOperator(*A_);
   AInv_->SetRelTol(1e-13);
   AInv_->SetMaxIter(3000);
   AInv_->SetPrintLevel(0);

   newBeta_  = false;
   newZeta_  = false;

   if ( myid_ > 0 ) { cout << "done" << endl; }
}
/*
void
MaxwellBlochWaveProjectorAMR::UpdateG()
{
 t01_->Update();
 delete T01_;
 T01_ = t01_->ParallelAssemble();

 if ( z01_ != NULL )
 {
    z01_->Update();
    delete Z01_;
    Z01_ = z01_->ParallelAssemble();
 }

 delete G_;
 G_ = new BlockOperator(block_trueOffsets1_,block_trueOffsets0_);
 G_->SetBlock(0,0,T01_);
 G_->SetBlock(1,1,T01_);
 if ( fabs(beta_) > 0.0 )
 {
    G_->SetBlock(0,1,Z01_, beta_);
    G_->SetBlock(1,0,Z01_,-beta_);
 }
 G_->owns_blocks = 0;
}
*/
void
MaxwellBlochWaveProjectorAMR::Update()
{
   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::Update" << endl;
   }

   // These are updated in their home object
   // H1FESpace_->Update();
   // HCurlFESpace_->Update();

   newSizes_  = true;
   currSizes_ = false;
   currVecs_  = false;
   currA0_    = false;
   currB0_    = false;
   currT01_   = false;
   currZ01_   = false;
   currBOpA_  = false;
   currBOpG_  = false;

   this->Setup();

   if ( myid_ == 0 && logging_ > 1 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::Update" << endl;
   }
}

void
MaxwellBlochWaveProjectorAMR::OldUpdate()
{
   MFEM_ASSERT(false, "Deprecated code");

   // The finite element spaces have changed so we need to repopulate
   // these arrays.
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
   delete u_; delete v_;
   u_ = new BlockVector(block_trueOffsets0_);
   v_ = new BlockVector(block_trueOffsets0_);

   // this->UpdateG();

   if ( myid_ == 0 ) { cout << "  Forming GMG" << endl; }
   HypreParMatrix *  M1 = dynamic_cast<HypreParMatrix*>(&M_->GetBlock(0,0));
   HypreParMatrix * GMG = RAP(M1,T01_);

   delete A0_;
   if ( fabs(beta_) > 0.0 )
   {
      if ( myid_ == 0 ) { cout << "  Forming ZMZ, GMZ, and ZMG" << endl; }
      HypreParMatrix * ZMZ = RAP(M1,Z01_);

      HypreParMatrix * GMZ = RAP(T01_, M1, Z01_);
      HypreParMatrix * ZMG = RAP(Z01_, M1, T01_);

      *GMZ *= -1.0;
      B0_ = ParAdd(GMZ,ZMG);

      delete GMZ;
      delete ZMG;

      *ZMZ *= beta_*beta_;
      A0_ = ParAdd(GMG,ZMZ);
      delete GMG;
      delete ZMZ;
   }
   else
   {
      A0_ = GMG;
   }

   if ( myid_ > 0 ) { cout << "Building Block A" << endl; }
   delete A_;
   A_ = new BlockOperator(block_trueOffsets0_);
   A_->SetDiagonalBlock(0,A0_,1.0);
   A_->SetDiagonalBlock(1,A0_,1.0);
   if ( fabs(beta_) > 0.0 )
   {
      A_->SetBlock(0,1,B0_,-beta_);
      A_->SetBlock(1,0,B0_, beta_);
   }
   A_->owns_blocks = 0;

   if ( myid_ > 0 ) { cout << "Creating MINRES Solver" << endl; }
   delete AInv_;
   AInv_ = new MINRESSolver(H1FESpace_->GetComm());
   AInv_->SetOperator(*A_);
   AInv_->SetRelTol(1e-13);
   AInv_->SetMaxIter(3000);
   AInv_->SetPrintLevel(0);

   newM_     = false;
   newBeta_  = false;
   newZeta_  = false;

   Operator::height = Operator::width = 2*HCurlFESpace_->GlobalTrueVSize();

   if ( myid_ > 0 ) { cout << "done" << endl; }
}

void
MaxwellBlochWaveProjectorAMR::Mult(const Vector &x, Vector &y) const
{
   if ( myid_ == 0 && logging_ > 2 )
   {
      cout << "Entering MaxwellBlochWaveProjectorAMR::Mult" << endl;
   }
   M_->Mult(x, y);
   G_->MultTranspose(y, *u_);
   *v_ = 0.0;
   AInv_->Mult(*u_, *v_);

   if ( myid_ == 0 && logging_ > 2 )
   {
      cout << "MINRES Iterations: " << AInv_->GetNumIterations() << endl;
   }

   G_->Mult(*v_, y);
   y *= -1.0;
   y += x;
   if ( myid_ == 0 && logging_ > 2 )
   {
      cout << "Leaving MaxwellBlochWaveProjectorAMR::Mult" << endl;
   }
}
/*
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
*/
} // namespace bloch
} // namespace mfem

#endif // MFEM_USE_MPI
