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

#include "mfem.hpp"
#include "pfem_extras.hpp"

using namespace std;

namespace mfem_bloch
{

H1_ParFESpace::H1_ParFESpace(ParMesh *m,
                             const int p, const int space_dim, const int type,
                             int vdim, int order)
   : ParFiniteElementSpace(m, new H1_FECollection(p,space_dim,type),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

H1_ParFESpace::~H1_ParFESpace()
{
   delete FEC_;
}

ND_ParFESpace::ND_ParFESpace(ParMesh *m, const int p, const int space_dim,
                             int vdim, int order)
   : ParFiniteElementSpace(m, new ND_FECollection(p,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

ND_ParFESpace::~ND_ParFESpace()
{
   delete FEC_;
}

RT_ParFESpace::RT_ParFESpace(ParMesh *m, const int p, const int space_dim,
                             int vdim, int order)
   : ParFiniteElementSpace(m, new RT_FECollection(p-1,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

RT_ParFESpace::~RT_ParFESpace()
{
   delete FEC_;
}

L2_ParFESpace::L2_ParFESpace(ParMesh *m, const int p, const int space_dim,
                             int vdim, int order)
   : ParFiniteElementSpace(m, new L2_FECollection(p,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

L2_ParFESpace::~L2_ParFESpace()
{
   delete FEC_;
}

ParDiscreteInterpolationOperator::~ParDiscreteInterpolationOperator()
{
   if ( pdlo_ != NULL ) { delete pdlo_; }
   if ( mat_  != NULL ) { delete mat_; }
}

HYPRE_Int
ParDiscreteInterpolationOperator::Mult(HypreParVector &x, HypreParVector &y,
                                       double alpha, double beta)
{
   return mat_->Mult( x, y, alpha, beta);
}

HYPRE_Int
ParDiscreteInterpolationOperator::Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                                       double alpha, double beta)
{
   return mat_->Mult( x, y, alpha, beta);
}

HYPRE_Int
ParDiscreteInterpolationOperator::MultTranspose(HypreParVector &x,
                                                HypreParVector &y,
                                                double alpha, double beta)
{
   return mat_->MultTranspose( x, y, alpha, beta);
}

void
ParDiscreteInterpolationOperator::Mult(double a, const Vector &x,
                                       double b, Vector &y) const
{
   mat_->Mult( a, x, b, y);
}

void
ParDiscreteInterpolationOperator::MultTranspose(double a, const Vector &x,
                                                double b, Vector &y) const
{
   mat_->MultTranspose( a, x, b, y);
}

void
ParDiscreteInterpolationOperator::Mult(const Vector &x, Vector &y) const
{
   mat_->Mult( x, y);
}

void
ParDiscreteInterpolationOperator::MultTranspose(const Vector &x,
                                                Vector &y) const
{
   mat_->MultTranspose( x, y);
}

ParDiscreteGradOperator::ParDiscreteGradOperator(ParFiniteElementSpace *dfes,
                                                 ParFiniteElementSpace *rfes)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new GradientInterpolator);
   pdlo_->Assemble();
   pdlo_->Finalize();
   mat_ = pdlo_->ParallelAssemble();
}

ParDiscreteCurlOperator::ParDiscreteCurlOperator(ParFiniteElementSpace *dfes,
                                                 ParFiniteElementSpace *rfes)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new CurlInterpolator);
   pdlo_->Assemble();
   pdlo_->Finalize();
   mat_ = pdlo_->ParallelAssemble();
}

ParDiscreteDivOperator::ParDiscreteDivOperator(ParFiniteElementSpace *dfes,
                                               ParFiniteElementSpace *rfes)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new DivergenceInterpolator);
   pdlo_->Assemble();
   pdlo_->Finalize();
   mat_ = pdlo_->ParallelAssemble();
}

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
VectorProductInterpolator::AssembleElementMatrix2(const FiniteElement &h1_fe,
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

void
VectorProductInterpolator::ScalarProduct_::Eval(Vector & vs,
                                                ElementTransformation &T,
                                                const IntegrationPoint &ip)
{
   vs.SetSize(v_.Size());

   h1_->CalcShape(ip, shape_);

   for (int i=0; i<v_.Size(); i++)
   {
      vs(i) = v_(i) * shape_(ind_);
   }
}

void
VectorCrossProductInterpolator::AssembleElementMatrix2(const FiniteElement
                                                       &nd_fe,
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

void
VectorCrossProductInterpolator::CrossProduct_::Eval(Vector & vxw,
                                                    ElementTransformation &T,
                                                    const IntegrationPoint &ip)
{
   vxw.SetSize(3);

   nd_->CalcVShape(T, vshape_);

   vxw(0) = v_(1) * vshape_(ind_,2) - v_(2) * vshape_(ind_,1);
   vxw(1) = v_(2) * vshape_(ind_,0) - v_(0) * vshape_(ind_,2);
   vxw(2) = v_(0) * vshape_(ind_,1) - v_(1) * vshape_(ind_,0);
}

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h)
{
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys maaAc" << endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

} // namespace mfem_bloch

#endif
