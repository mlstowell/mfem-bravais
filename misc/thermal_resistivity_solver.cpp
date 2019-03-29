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

#include "thermal_resistivity_solver.hpp"

using namespace std;
using namespace mfem::miniapps;

namespace mfem
{

class BdrGradIntegrator : public LinearFormIntegrator
{
public:
   BdrGradIntegrator()
      : Q(NULL) {}
   BdrGradIntegrator(Coefficient & a)
      : Q(&a) {}

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect)
   {
   }
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect)
   {
      int order;
      int nd = el.GetDof();
      int dim = el.GetDim();
      double w, nor_data[3], nort_data[3];

      Vector nor(nor_data, dim);
      Vector nort(nort_data, dim);

      dshape.SetSize(nd,dim);
      invdfdx.SetSize(dim);
      elvect.SetSize(nd);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         // Assuming order(u)==order(mesh)
         // order = Tr.Elem1->OrderW() + 2*el.GetOrder();
         order = 2*el.GetOrder() + dim - 1;
         if (el.Space() == FunctionSpace::Pk)
         {
            order = 2*el.GetOrder() - 2;
         }
         ir = &IntRules.Get(Tr.FaceGeom, order);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         IntegrationPoint eip;
         Tr.Loc1.Transform(ip, eip);
         el.CalcDShape(eip, dshape);

         Tr.Face->SetIntPoint(&ip);
         Tr.Elem1->SetIntPoint(&eip);

         CalcInverse(Tr.Elem1->Jacobian(), invdfdx);

         if (dim == 1)
         {
            nor(0) = 2*eip.x - 1.0;
         }
         else
         {
            CalcOrtho(Tr.Face->Jacobian(), nor);
         }
         w = ip.weight;// / Tr.Elem1->Weight();
         if ( Q )
         {
            w *= Q->Eval(*Tr.Face, ip);
         }

         invdfdx.Mult(nor,nort);

         nort *= w;
         dshape.AddMult(nort, elvect);
      }
   }

private:
   Coefficient * Q;
   DenseMatrix dshape;
   DenseMatrix invdfdx;
};

/// Coefficient defined on a subset of domain or boundary attributes
class BdrRestrictedCoefficient : public Coefficient
{
private:
   ParMesh * pmesh;
   Coefficient *c;
   Array<int> active_attr;

public:
   BdrRestrictedCoefficient(ParMesh & _pmesh, Coefficient &_c, Array<int> &attr)
      : pmesh(&_pmesh), c(&_c) { attr.Copy(active_attr); }

   BdrRestrictedCoefficient(Array<int> &attr)
   { pmesh = NULL; c = NULL; attr.Copy(active_attr); }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      if ( c )
      {
         if ( !active_attr[T.Attribute-1] )
         {
            return 0.0;
         }

         FaceElementTransformations & Tr =
            pmesh->GetFaceElementTransformations();

         IntegrationPoint eip;
         Tr.Loc1.Transform(ip, eip);
         Tr.Elem1->SetIntPoint(&eip);

         return c->Eval(*Tr.Elem1, eip, GetTime());
      }
      else
      {
         return active_attr[T.Attribute-1] ? 1.0 : 0.0;
      }
   }

};

ThermalResistivity::ThermalResistivity(MPI_Comm & comm, ParMesh & pmesh,
                                       ParFiniteElementSpace & L2FESpace,
                                       Coefficient & kCoef,
                                       int dim, int order,
                                       double a)
   : commPtr_(&comm),
     myid_(0),
     numProcs_(1),
     dim_(dim),
     order_(order),
     a_(a),
     pmesh_(&pmesh),
     H1FESpace_(NULL),
     L2FESpace_(&L2FESpace),
     kCoef_(&kCoef),
     rkCoef_(NULL),
     rCoef_(NULL),
     ak_(NULL),
     gk_(NULL),
     t_(NULL),
     rhs_(NULL),
     k_(NULL),
     w_(NULL),
     A_(NULL)
{
   MPI_Comm_rank(*commPtr_, &myid_);
   MPI_Comm_size(*commPtr_, &numProcs_);

   ess_bdr_.SetSize(pmesh_->bdr_attributes.Max());
   ess_bdr_ = 0;

   ess_bdr2_.SetSize(pmesh_->bdr_attributes.Max());
   ess_bdr2_ = 0;

   switch (dim_)
   {
      case 1:
         ess_bdr_[0] = 1; ess_bdr_[1] = 1;
         ess_bdr2_[1] = 1;
         break;
      case 2:
         ess_bdr_[1] = 1; ess_bdr_[3] = 1;
         ess_bdr2_[1] = 1;
         break;
      case 3:
         ess_bdr_[2] = 1; ess_bdr_[4] = 1;
         ess_bdr2_[2] = 1;
         break;
   }

   H1FESpace_    = new H1_ParFESpace(pmesh_,order_,dim_);

   ak_ = new ParBilinearForm(H1FESpace_);
   ak_->AddDomainIntegrator(new DiffusionIntegrator(*kCoef_));

   rkCoef_ = new BdrRestrictedCoefficient(*pmesh_, *kCoef_, ess_bdr2_);
   rCoef_ = new BdrRestrictedCoefficient(ess_bdr2_);
   gk_ = new ParLinearForm(H1FESpace_);
   gk_->AddBdrFaceIntegrator(new BdrGradIntegrator(*rkCoef_));

   t_   = new ParGridFunction(H1FESpace_);
   rhs_ = new ParGridFunction(H1FESpace_);
   k_   = new ParGridFunction(L2FESpace_);
   w_   = new ParGridFunction(H1FESpace_);

   this->InitSecondaryObjects();
}

ThermalResistivity::~ThermalResistivity()
{
   delete A_;

   delete t_;
   delete rhs_;
   delete k_;
   delete w_;

   delete ak_;
   delete gk_;

   delete rkCoef_;
   delete rCoef_;

   delete H1FESpace_;

   delete pmesh_;
}

void
ThermalResistivity::SetConductivityCoef(Coefficient & kCoef)
{
   kCoef_ = &kCoef;

   delete ak_;
   ak_ = new ParBilinearForm(H1FESpace_);
   ak_->AddDomainIntegrator(new DiffusionIntegrator(*kCoef_));
   ak_->Assemble();
   ak_->Finalize();

   delete A_; A_ = new HypreParMatrix;

   delete rkCoef_;
   rkCoef_ = new BdrRestrictedCoefficient(*pmesh_, *kCoef_, ess_bdr2_);

   delete gk_;
   gk_ = new ParLinearForm(H1FESpace_);
   gk_->AddBdrFaceIntegrator(new BdrGradIntegrator(*rkCoef_));
   gk_->Assemble();
}

void
ThermalResistivity::ConductivityChanged()
{
   delete A_; A_ = new HypreParMatrix;

   ak_->Update();
   ak_->Assemble();
   ak_->Finalize();

   gk_->Assemble();
}

void
ThermalResistivity::InitSecondaryObjects()
{
   H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_tdof_list_);

   ak_->Assemble();
   ak_->Finalize();

   A_ = new HypreParMatrix;

   gk_->Assemble();
}

void
ThermalResistivity::Solve(ParGridFunction * w)
{
   /// Set the Temperature to 0.0 on surface 1 and 1.0 pn surface 2
   ConstantCoefficient one(1.0);
   *t_ = 0.0;
   t_->ProjectBdrCoefficient(one, ess_bdr2_);

   *rhs_ = 0.0;

   ak_->FormLinearSystem(ess_tdof_list_, *t_, *rhs_, *A_, T_, RHS_);

   HypreBoomerAMG amg(*A_);
   amg.SetPrintLevel(0);
   HyprePCG pcg(*A_);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(200);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(amg);
   pcg.Mult(RHS_, T_);

   ak_->RecoverFEMSolution(T_, *rhs_, *t_);

   if ( w != NULL )
   {
      H1FESpace_->Dof_TrueDof_Matrix()->MultTranspose(*gk_, RHS_);
      pcg.Mult(RHS_, T_);
      for (int i=0; i<ess_tdof_list_.Size(); i++)
      {
         T_[ess_tdof_list_[i]] = 0.0;
      }
      H1FESpace_->Dof_TrueDof_Matrix()->Mult(T_, *w);
   }
}

double
ThermalResistivity::GetResistivity(Vector * dR)
{
   double R = this->CalcResistivity(dR);

   if ( myid_ == 0 )
   {
      cout << "R = " << R << endl;
   }

   return R;
}

void
ThermalResistivity::CalcSensitivity(Vector & dF)
{
   dF.SetSize(pmesh_->GetNE());
   dF = 0.0;

   Array<int> h1Vdofs, l2Vdofs;
   DenseMatrix sMat;
   Vector tVec, wVec, gVec, stVec;

   BdrGradIntegrator gInt(*rCoef_);
   DiffusionIntegrator sInt;

   for (int i=0; i<pmesh_->GetNBE(); i++)
   {
      FaceElementTransformations * tr = pmesh_->GetBdrFaceTransformations(i);

      int el1 = tr -> Elem1No;
      H1FESpace_->GetElementVDofs(el1,h1Vdofs);;
      L2FESpace_->GetElementVDofs(el1,l2Vdofs);;

      t_->GetSubVector(h1Vdofs, tVec);

      gInt.AssembleRHSElementVect(*H1FESpace_->GetFE(el1),
                                  *tr, gVec);
      dF[l2Vdofs[0]] += gVec * tVec;
   }
   for (int i=0; i<pmesh_->GetNE(); i++)
   {
      H1FESpace_->GetElementVDofs(i,h1Vdofs);;
      L2FESpace_->GetElementVDofs(i,l2Vdofs);;

      t_->GetSubVector(h1Vdofs, tVec);
      w_->GetSubVector(h1Vdofs, wVec);

      sInt.AssembleElementMatrix(*H1FESpace_->GetFE(i),
                                 *H1FESpace_->GetElementTransformation(i),
                                 sMat);

      stVec.SetSize(h1Vdofs.Size());
      sMat.Mult(tVec, stVec);
      dF[l2Vdofs[0]] -= wVec * stVec;
   }

}

double
ThermalResistivity::CalcResistivity(Vector * dR)
{
   double R = -1.0;

   if ( dR == NULL)
   {
      this->Solve();
   }
   else
   {
      this->Solve(w_);
      this->CalcSensitivity(*dR);
      cout << "back from calc sens" << endl;
   }

   double loc_flux = *gk_ * *t_;
   double glb_flux = 0.0;
   MPI_Allreduce(&loc_flux, &glb_flux, 1, MPI_DOUBLE, MPI_SUM, *commPtr_);
   cout << "global flux " << glb_flux << endl;
   if ( glb_flux > 0.0 )
   {
      switch (dim_)
      {
         case 1:
            R = 2.0 / (a_ * glb_flux);
            break;
         case 2:
            R = 1.0 / glb_flux;
            break;
         case 3:
            R = 0.5 * a_ / glb_flux;
            break;
      }
   }
   if ( dR != NULL && fabs(glb_flux) > 0.0 )
   {
      *dR *= - R / glb_flux;
   }

   return R;
}

void
ThermalResistivity::Visualize(ParGridFunction * dR)
{
   socketstream t_sock, k_sock, w_sock, dr_sock;
   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10;    // window offsets
   int offy = Wh+45; // window offsets

   VisualizeField(t_sock, vishost, visport, *t_,
                  "Temperature", Wx, Wy, Ww, Wh);

   Wx += offx;

   k_->ProjectCoefficient(*kCoef_);

   VisualizeField(k_sock, vishost, visport, *k_,
                  "Thermal Conductivity", Wx, Wy, Ww, Wh);

   Wx -= offx; Wy += offy;
   VisualizeField(w_sock, vishost, visport, *w_,
                  "Adjoint Vector", Wx, Wy, Ww, Wh);

   if ( dR )
   {
      Wx += offx;
      VisualizeField(dr_sock, vishost, visport, *dR,
                     "dR", Wx, Wy, Ww, Wh);
   }
}

} // namespace mfem

#endif // MFEM_USE_MPI
