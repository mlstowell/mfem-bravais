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

#include "maxwell_bloch_shift.hpp"
#include <fstream>

using namespace std;

namespace mfem
{

using namespace miniapps;

namespace bloch
{

MaxwellBlochWaveEquationShift::MaxwellBlochWaveEquationShift(ParMesh & pmesh,
                                                             int order,
                                                             bool sns)
   : myid_(0),
     nev_(-1),
     newBeta_(true),
     newZeta_(true),
     newOmega_(true),
     newMCoef_(true),
     newKCoef_(true),
     shiftNullSpace_(sns),
     pmesh_(&pmesh),
     H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     L2FESpace_(NULL),
     bravais_(NULL),
     fourierHCurl_(NULL),
     atol_(1.0e-6),
     beta_(0.0),
     omega_max_(1.0),
     mCoef_(NULL),
     kCoef_(NULL),
     A_(NULL),
     M_(NULL),
     C_(NULL),
     blkHCurl_(NULL),
     blkHDiv_(NULL),
     M0_(NULL),
     M1_(NULL),
     M2_(NULL),
     S1_(NULL),
     T1_(NULL),
     T01_(NULL),
     T01T_(NULL),
     T12_(NULL),
     Z01_(NULL),
     Z01T_(NULL),
     Z12_(NULL),
     DKZ_(NULL),
     DKZT_(NULL),
     T1Inv_ams_(NULL),
     T1Inv_minres_(NULL),
     t01_(NULL),
     t12_(NULL),
     z01_(NULL),
     z12_(NULL),
     BDP_(NULL),
     Precond_(NULL),
     SubSpaceProj_(NULL),
     vecs_(NULL),
     vec0_(NULL),
     lobpcg_(NULL),
     ame_(NULL),
     energy_(NULL)
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

MaxwellBlochWaveEquationShift::~MaxwellBlochWaveEquationShift()
{
   delete lobpcg_;

   if ( vecs_ != NULL )
   {
      for (int i=0; i<nev_; i++) { delete vecs_[i]; }
      delete [] vecs_;
   }

   delete SubSpaceProj_;
   delete Precond_;

   delete blkHCurl_;
   delete blkHDiv_;
   delete A_;
   delete M_;
   delete C_;
   delete BDP_;
   delete T1Inv_ams_;
   delete T1Inv_minres_;

   delete M0_;
   delete M1_;
   delete M2_;
   delete S1_;
   delete T1_;
   delete T01_;
   delete T01T_;
   delete T12_;
   delete Z01_;
   delete Z01T_;
   delete Z12_;
   delete DKZ_;
   delete DKZT_;
   delete t01_;
   delete t12_;
   delete z01_;
   delete z12_;

   delete fourierHCurl_;

   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;
   delete L2FESpace_;
}

void
MaxwellBlochWaveEquationShift::SetKappa(const Vector & kappa)
{
   if ( myid_ == 0 )
   {
      cout << "Setting Kappa: ";
      kappa.Print(cout);
      cout << endl;
   }
   kappa_ = kappa;
   beta_  = kappa.Norml2();  newBeta_ = true;
   zeta_  = kappa;           newZeta_ = true;
   if ( fabs(beta_) > 0.0 )
   {
      zeta_ /= beta_;
   }
}

void
MaxwellBlochWaveEquationShift::SetBeta(double beta)
{
   beta_ = beta; newBeta_ = true;
}

void
MaxwellBlochWaveEquationShift::SetZeta(const Vector & zeta)
{
   zeta_ = zeta; newZeta_ = true;
}

void
MaxwellBlochWaveEquationShift::SetAbsoluteTolerance(double atol)
{
   atol_ = atol;
}

void
MaxwellBlochWaveEquationShift::SetNumEigs(int nev)
{
   nev_ = nev;
}

void
MaxwellBlochWaveEquationShift::SetMassCoef(Coefficient & m)
{
   mCoef_ = &m; newMCoef_ = true;
}

void
MaxwellBlochWaveEquationShift::SetStiffnessCoef(Coefficient & k)
{
   kCoef_ = &k; newKCoef_ = true;
}

void
MaxwellBlochWaveEquationShift::SetMaximumLightSpeed(double c)
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
MaxwellBlochWaveEquationShift::Setup()
{
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
   if ( newKCoef_ && shiftNullSpace_ )
   {
      if ( myid_ == 0 ) { cout << "Building M0(k)" << endl; }
      ParBilinearForm m0(H1FESpace_);
      m0.AddDomainIntegrator(new MassIntegrator(*kCoef_));
      m0.Assemble();
      m0.Finalize();
      delete M0_;
      M0_ = m0.ParallelAssemble();
   }

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
   if ( newZeta_ && shiftNullSpace_ )
   {
      if ( myid_ == 0 ) { cout << "Building zeta product operator" << endl; }
      delete z01_;
      z01_ = new ParDiscreteVectorProductOperator(H1FESpace_,
                                                  HCurlFESpace_,zeta_);
      z01_->Assemble();
      z01_->Finalize();
      Z01_ = z01_->ParallelAssemble();
      Z01T_ = Z01_->Transpose();
   }

   if ( t12_ == NULL )
   {
      if ( myid_ == 0 ) { cout << "Building Curl operator" << endl; }
      t12_ = new ParDiscreteCurlOperator(HCurlFESpace_,HDivFESpace_);
      t12_->Assemble();
      t12_->Finalize();
      T12_ = t12_->ParallelAssemble();
   }
   if ( t01_ == NULL && shiftNullSpace_ )
   {
      if ( myid_ == 0 ) { cout << "Building Gradient operator" << endl; }
      t01_ = new ParDiscreteGradOperator(H1FESpace_, HCurlFESpace_);
      t01_->Assemble();
      t01_->Finalize();
      T01_ = t01_->ParallelAssemble();
      T01T_ = T01_->Transpose();
   }

   if ( newZeta_ || newBeta_ || newKCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Forming CMC" << endl; }
      HypreParMatrix * CMC = RAP(M2_,T12_);
      HypreParMatrix * GMG = NULL;

      if ( shiftNullSpace_ )
      {
         GMG = RAP(M0_,T01T_);
      }

      delete S1_;

      if ( myid_ == 0 ) { cout << "Forming 2nd order operators" << endl; }
      if ( fabs(beta_) > 0.0 )
      {
         HypreParMatrix * ZMZ = RAP(M2_, Z12_);
         HypreParMatrix * CMZ = RAP(T12_, M2_, Z12_);
         HypreParMatrix * ZMC = RAP(Z12_, M2_, T12_);

         *ZMC *= -1.0;
         delete DKZ_;
         DKZ_ = ParAdd(CMZ,ZMC);
         delete CMZ;
         delete ZMC;

         *ZMZ *= beta_*beta_;
         S1_ = ParAdd(CMC,ZMZ);
         delete CMC;
         delete ZMZ;

         if ( shiftNullSpace_ )
         {
            HypreParMatrix * ZM0Z = RAP(M0_, Z01T_);
            HypreParMatrix * GM0Z = RAP(T01T_, M0_, Z01T_);
            HypreParMatrix * ZM0G = RAP(Z01T_, M0_, T01T_);

            HypreParMatrix * DKZ_tmp = DKZ_;

            *GM0Z *= -1.0;
            HypreParMatrix * DKZ_shift = ParAdd(ZM0G, GM0Z);
            delete ZM0G;
            delete GM0Z;

            *DKZ_shift *= 1000.0 * omega_max_ * omega_max_;
            DKZ_ = ParAdd(DKZ_shift,DKZ_tmp);
            delete DKZ_shift;
            delete DKZ_tmp;

            HypreParMatrix * S1_tmp = S1_;

            *ZM0Z *= beta_*beta_;
            HypreParMatrix * S1_shift = ParAdd(GMG, ZM0Z);
            S1_shift->Print("S1_shift.mat");
            delete GMG;
            delete ZM0Z;

            *S1_shift *= 1000.0 * omega_max_ * omega_max_;
            S1_ = ParAdd(S1_shift, S1_tmp);
            delete S1_shift;
            delete S1_tmp;
         }

      }
      else
      {
         if ( shiftNullSpace_ )
         {
            *GMG *= 1000.0 * omega_max_ * omega_max_;
            S1_ = ParAdd(GMG, CMC);
         }
         else
         {
            S1_ = CMC;
         }
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
         C_->SetBlock(0,1,Z12_, beta_);
         C_->SetBlock(1,0,Z12_,-beta_);
      }
      C_->owns_blocks = 0;
   }

   if ( newZeta_ || newBeta_ || newKCoef_ )
   {
      if ( myid_ == 0 ) { cout << "Building T1Inv" << endl; }

      if ( !shiftNullSpace_ || fabs(beta_) == 0.0 )
      {
         delete T1Inv_ams_;
         T1Inv_ams_ = new HypreAMS(*S1_,HCurlFESpace_);

         if ( fabs(beta_*180.0) < M_PI )
         {
            cout << "HypreAMS::SetSingularProblem()" << endl;
            T1Inv_ams_->SetSingularProblem();
         }
      }
      else
      {
         delete T1Inv_minres_;
         T1Inv_minres_ = new MINRESSolver(comm_);
         T1Inv_minres_->SetOperator(*S1_);
      }

      if ( fabs(beta_) > 0.0 )
      {
         if ( myid_ == 0 ) { cout << "Building BDP" << endl; }
         delete BDP_;
         BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets_);
         if ( !shiftNullSpace_ )
         {
            BDP_->SetDiagonalBlock(0,T1Inv_ams_);
            BDP_->SetDiagonalBlock(1,T1Inv_ams_);
         }
         else
         {
            BDP_->SetDiagonalBlock(0,T1Inv_minres_);
            BDP_->SetDiagonalBlock(1,T1Inv_minres_);
         }
         BDP_->owns_blocks = 0;
      }
   }

   if ( ( newZeta_ || newBeta_ || newMCoef_ || newKCoef_ ) && nev_ > 0 )
   {
      if ( fabs(beta_) > 0.0 )
      {
         if ( ! shiftNullSpace_ )
         {
            delete SubSpaceProj_;
            if ( myid_ == 0 ) { cout << "Building Subspace Projector" << endl; }
            SubSpaceProj_ = new MaxwellBlochWaveProjectorShift(*HCurlFESpace_,
                                                               *H1FESpace_,
                                                               *M_,beta_,zeta_);
            SubSpaceProj_->Setup();
         }
         if ( myid_ == 0 ) { cout << "Building Preconditioner" << endl; }
         delete Precond_;
         Precond_ = new MaxwellBlochWavePrecond(*HCurlFESpace_,*BDP_,SubSpaceProj_,0.5);
         Precond_->SetOperator(*A_);

         if ( myid_ == 0 ) { cout << "Building HypreLOBPCG solver" << endl; }
         delete lobpcg_;
         lobpcg_ = new HypreLOBPCG(comm_);

         lobpcg_->SetNumModes(nev_);
         if ( ! shiftNullSpace_ )
         {
            lobpcg_->SetPreconditioner(*this->GetPreconditioner());
         }
         lobpcg_->SetMaxIter(2000);
         lobpcg_->SetTol(atol_);
         lobpcg_->SetPrecondUsageMode(1);
         lobpcg_->SetPrintLevel(1);

         // Set the matrices which define the linear system
         lobpcg_->SetMassMatrix(*this->GetMOperator());
         lobpcg_->SetOperator(*this->GetAOperator());
         if ( ! shiftNullSpace_ )
         {
            lobpcg_->SetSubSpaceProjector(*this->GetSubSpaceProjector());
         }

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

   newZeta_  = false;
   newBeta_  = false;
   newOmega_ = false;
   newMCoef_ = false;
   newKCoef_ = false;

   if ( myid_ == 0 ) { cout << "Leaving Setup" << endl; }
}

void
MaxwellBlochWaveEquationShift::SetInitialVectors(int num_vecs,
                                                 HypreParVector ** vecs)
{
   if ( lobpcg_ )
   {
      lobpcg_->SetInitialVectors(num_vecs, vecs);
   }
}

void MaxwellBlochWaveEquationShift::Update()
{
   if ( myid_ == 0 ) { cout << "Building M2(k)" << endl; }
   ParBilinearForm m2(HDivFESpace_);
   m2.AddDomainIntegrator(new VectorFEMassIntegrator(*kCoef_));
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
      C_->SetBlock(0,1,Z12_,beta_);
      C_->SetBlock(1,0,Z12_,-beta_);
   }
   C_->owns_blocks = 0;

   if ( myid_ == 0 ) { cout << "Building T1Inv" << endl; }
   if ( !shiftNullSpace_ )
   {
      delete T1Inv_ams_;
      if ( fabs(beta_) < 1.0 )
      {
         T1Inv_ams_ = new HypreAMS(*S1_,HCurlFESpace_);
         T1Inv_ams_->SetSingularProblem();
      }
      else
      {
         T1Inv_ams_ = new HypreAMS(*S1_,HCurlFESpace_);
         T1Inv_ams_->SetSingularProblem();
      }
   }
   else
   {
      delete T1Inv_minres_;
      T1Inv_minres_ = new MINRESSolver(comm_);
      T1Inv_minres_->SetOperator(*S1_);
   }

   if ( myid_ == 0 ) { cout << "Building BDP" << endl; }
   delete BDP_;
   BDP_ = new BlockDiagonalPreconditioner(block_trueOffsets_);
   if ( !shiftNullSpace_ )
   {
      BDP_->SetDiagonalBlock(0,T1Inv_ams_);
      BDP_->SetDiagonalBlock(1,T1Inv_ams_);
   }
   else
   {
      BDP_->SetDiagonalBlock(0,T1Inv_minres_);
      BDP_->SetDiagonalBlock(1,T1Inv_minres_);
   }
   BDP_->owns_blocks = 0;

   if ( SubSpaceProj_ ) { SubSpaceProj_->Update(); }

   if ( myid_ == 0 ) { cout << "Building Preconditioner" << endl; }
   delete Precond_;
   Precond_ = new MaxwellBlochWavePrecond(*HCurlFESpace_,*BDP_,
                                          SubSpaceProj_,0.5);
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
   if ( ! shiftNullSpace_ )
   {
      lobpcg_->SetSubSpaceProjector(*this->GetSubSpaceProjector());
   }

   newZeta_  = false;
   newBeta_  = false;
   newMCoef_ = false;
   newKCoef_ = false;
}

void
MaxwellBlochWaveEquationShift::Solve()
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
MaxwellBlochWaveEquationShift::GetEigenvalues(vector<double> & eigenvalues)
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
MaxwellBlochWaveEquationShift::GetEigenvalues(int nev, const Vector & kappa,
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
MaxwellBlochWaveEquationShift::GetEigenvector(unsigned int i,
                                              HypreParVector & Er,
                                              HypreParVector & Ei,
                                              HypreParVector & Br,
                                              HypreParVector & Bi)
{
   this->GetEigenvectorE(i, Er, Ei);
   this->GetEigenvectorB(i, Br, Bi);
}

void
MaxwellBlochWaveEquationShift::GetEigenvectorE(unsigned int i,
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
MaxwellBlochWaveEquationShift::GetEigenvectorB(unsigned int i,
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

void
MaxwellBlochWaveEquationShift::GetFourierCoefficients(HypreParVector & Vr,
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
MaxwellBlochWaveEquationShift::GetFieldAverages(unsigned int i,
                                                Vector & Er, Vector & Ei,
                                                Vector & Br, Vector & Bi,
                                                Vector & Dr, Vector & Di,
                                                Vector & Hr, Vector & Hi)
{
   if ( fourierHCurl_ == NULL)
   {
      MFEM_ASSERT(bravais_ != NULL, "MaxwellBlochWaveEquationShift: "
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
MaxwellBlochWaveEquationShift::WriteVisitFields(const string & prefix,
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

MaxwellBlochWaveEquationShift::MaxwellBlochWavePrecond::
MaxwellBlochWavePrecond(ParFiniteElementSpace & HCurlFESpace,
                        BlockDiagonalPreconditioner & BDP,
                        Operator * subSpaceProj,
                        double w)
   : Solver(2*HCurlFESpace.GlobalTrueVSize()),
     myid_(0), BDP_(&BDP), subSpaceProj_(subSpaceProj), u_(NULL)
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

MaxwellBlochWaveEquationShift::
MaxwellBlochWavePrecond::~MaxwellBlochWavePrecond()
{
   delete u_;
}

void
MaxwellBlochWaveEquationShift::
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
MaxwellBlochWaveEquationShift::
MaxwellBlochWavePrecond::SetOperator(const Operator & A)
{
   A_ = &A;
}

MaxwellBlochWaveProjectorShift::
MaxwellBlochWaveProjectorShift(ParFiniteElementSpace & HCurlFESpace,
                               ParFiniteElementSpace & H1FESpace,
                               BlockOperator & M,
                               double beta, const Vector & zeta)
   : Operator(2*HCurlFESpace.GlobalTrueVSize()),
     newBeta_(true),
     newZeta_(true),
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

MaxwellBlochWaveProjectorShift::~MaxwellBlochWaveProjectorShift()
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
MaxwellBlochWaveProjectorShift::SetBeta(double beta)
{
   beta_ = beta; newBeta_ = true;
}

void
MaxwellBlochWaveProjectorShift::SetZeta(const Vector & zeta)
{
   zeta_ = zeta; newZeta_ = true;
}

void
MaxwellBlochWaveProjectorShift::Setup()
{
   if ( myid_ == 0 )
   {
      cout << "Setting up MaxwellBlochWaveProjector" << endl;
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
MaxwellBlochWaveProjectorShift::Update()
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
MaxwellBlochWaveProjectorShift::Mult(const Vector &x, Vector &y) const
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
