#include "mfem.hpp"
#include "../common/pfem_extras.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;

class ThermalResistivity
{
public:
   ThermalResistivity(MPI_Comm & comm, Coefficient & kCoef,
                      int dim = 1, int order = 1, int n = 8,
                      double a = 1.0, double tol = 1.0e-2);
   ~ThermalResistivity();

   void SetConductivityCoef(Coefficient & kCoef);

   double GetResistivity(const bool & ref = true, Vector * dR = NULL);

   void Visualize(ParGridFunction * dR = NULL);

   ParFiniteElementSpace * GetL2FESpace() { return L2FESpace_; }

private:
   void InitSecondaryObjects();

   void UpdateAndRebalance();

   void Solve(ParGridFunction * w = NULL);

   void CalcSensitivity(Vector & dF);

   double CalcResistivity(Vector * dR = NULL);

   double EstimateErrors();

   MPI_Comm * commPtr_;
   int        myid_;
   int        numProcs_;
   int        dim_;
   int        order_;
   int        n_;
   double     a_;
   double     tol_;
   ParMesh  * pmesh_;

   ParFiniteElementSpace * H1FESpace_;
   ParFiniteElementSpace * L2FESpace_;

   Coefficient           * kCoef_;
   Coefficient           * rkCoef_;
   Coefficient           * rCoef_;
   // Coefficient           * rkCoef1_;
   ParBilinearForm       * ak_;
   ParLinearForm         * gk_;
   // ParLinearForm         * gk1_;

   ParGridFunction * t_;
   ParGridFunction * rhs_;
   ParGridFunction * k_;
   ParGridFunction * w_;

   HypreParMatrix * A_;

   mutable HypreParVector * T_;
   mutable HypreParVector * RHS_;

   Vector errors_;

   Array<int> ess_bdr_;
   Array<int> ess_bdr2_;
   Array<int> ess_tdof_list_;
};

class KCoef : public Coefficient
{
public:
   KCoef(double r1 = 0.14, double r2 = 0.36, double r3 = 0.105)
      : r1_(r1),
        r2_(r2),
        r3_(r3)
   {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      double k0 = 0.025; /// air at 1 atm and 290 K
      double k1 = 237.0; /// aluminum at 290 K

      double x_data[3];
      Vector x(x_data, 3);

      T.Transform(ip, x);

      double y_data[3];
      Vector y(y_data,3); y = 0.0;
      for (int i=0; i<x.Size(); i++) { y[i] = x[i]; }

      if ( y.Norml2() <= r1_ ) { return k0; }
      if ( y.Norml2() <= r2_ ) { return k1; }
      if ( sqrt(y(1)*y(1)+y(2)*y(2)) <= r3_ ) { return k1; }
      if ( sqrt(y(2)*y(2)+y(0)*y(0)) <= r3_ ) { return k1; }
      if ( sqrt(y(0)*y(0)+y(1)*y(1)) <= r3_ ) { return k1; }

      return k0;
   }

private:
   double r1_;
   double r2_;
   double r3_;
};

static int prob_ = 0;

double kFunc(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // 2. Parse command-line options.
   int    dim = 2;
   int    order = 1;
   int    n = 8;
   double a = 1.0;
   double tol = 1.0e-4;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&dim, "-d", "--dimension",
                  "Space dimension");
   args.AddOption(&n, "-n", "--num-elements",
                  "Number of elements in one direction");
   args.AddOption(&a, "-a", "--cell-size",
                  "Length of unit cell");
   args.AddOption(&tol, "-t", "--tolerance",
                  "Relative Tolerance for R");
   args.AddOption(&prob_, "-p", "--problem",
                  "Used to select geometry");
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

   // FunctionCoefficient kCoef(kFunc);
   KCoef kCoef0;
   ThermalResistivity tr(comm, kCoef0, dim, order, n, a, tol);

   ParGridFunction dR(tr.GetL2FESpace());
   double R_lambda = tr.GetResistivity(true, &dR);
   if ( myid == 0 )
   {
      cout << "Specific Resistivity:  " << R_lambda << endl;
   }

   tr.Visualize(&dR);

   KCoef kCoef1(0.14, 0.36, 0.106);
   cout << 0 << endl;
   ParGridFunction k0(tr.GetL2FESpace());
   ParGridFunction k1(tr.GetL2FESpace());
   ParGridFunction dk(tr.GetL2FESpace());
   k0.ProjectCoefficient(kCoef0);
   k1.ProjectCoefficient(kCoef1);

   dk = k1; dk -= k0;
   // tr.Visualize(&dk);
   double deltaR = dk * dR;
   cout << "delta R: " << deltaR << endl;

   // cout << "k/dR:  " << k.Size() << "/" << dR.Size() << endl;
   // k += dR;
   // cout << 1 << endl;
   tr.SetConductivityCoef(kCoef1);
   cout << 2 << endl;
   double R_lambda1 = tr.GetResistivity(true);
   if ( myid == 0 )
   {
      cout << "Specific Resistivity:  " << R_lambda1 << endl;
      cout << "Delta R: " << R_lambda1 - R_lambda << endl;
      cout << "S Delta R: " << (R_lambda1 - R_lambda)/deltaR << endl;
   }

   tr.Visualize();

   MPI_Finalize();

   return 0;
}

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
         order = Tr.Elem1->OrderW() + 2*el.GetOrder();
         if (el.Space() == FunctionSpace::Pk)
         {
            order++;
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

         CalcAdjugate(Tr.Elem1->Jacobian(), invdfdx);

         if (dim == 1)
         {
            nor(0) = 2*eip.x - 1.0;
         }
         else
         {
            CalcOrtho(Tr.Face->Jacobian(), nor);
         }
         w = ip.weight / Tr.Elem1->Weight();
         if ( Q )
         {
            w *= Q->Eval(*Tr.Face, ip);
            //w *= Q->Eval(*Tr.Elem1, eip);
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
   Coefficient *c;
   Array<int> active_attr;

public:
   BdrRestrictedCoefficient(Coefficient &_c, Array<int> &attr)
   { c = &_c; attr.Copy(active_attr); }

   BdrRestrictedCoefficient(Array<int> &attr)
   { c = NULL; attr.Copy(active_attr); }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      if ( c )
      {
         return active_attr[T.Attribute-1] ? c->Eval(T, ip, GetTime()) : 0.0;
      }
      else
      {
         return active_attr[T.Attribute-1] ? 1.0 : 0.0;
      }
   }

};

ThermalResistivity::ThermalResistivity(MPI_Comm & comm, Coefficient & kCoef,
                                       int dim, int order, int n,
                                       double a, double tol)
   : commPtr_(&comm),
     myid_(0),
     numProcs_(1),
     dim_(dim),
     order_(order),
     n_(n),
     a_(a),
     tol_(tol),
     pmesh_(NULL),
     H1FESpace_(NULL),
     L2FESpace_(NULL),
     kCoef_(&kCoef),
     rkCoef_(NULL),
     rCoef_(NULL),
     // rkCoef1_(NULL),
     ak_(NULL),
     gk_(NULL),
     // gk1_(NULL),
     t_(NULL),
     rhs_(NULL),
     k_(NULL),
     w_(NULL),
     A_(NULL),
     T_(NULL),
     RHS_(NULL)
{
   MPI_Comm_rank(*commPtr_, &myid_);
   MPI_Comm_size(*commPtr_, &numProcs_);

   Mesh * mesh = NULL;
   switch (dim_)
   {
      case 1:
         mesh = new Mesh(n, 0.5 * a);
         break;
      case 2:
         mesh = new Mesh(n, n, Element::QUADRILATERAL, 1, 0.5 * a, 0.5 * a);
         break;
      case 3:
         mesh = new Mesh(n, n, n, Element::HEXAHEDRON, 1, 0.5 * a, 0.5 * a, 0.5 * a);
         break;
      default:
         MFEM_ASSERT(mesh != NULL, "Mesh dimension must be 2 or 3!");
   }
   mesh->EnsureNCMesh();
   pmesh_ = new ParMesh(*commPtr_, *mesh);
   delete mesh;

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
   L2FESpace_    = new L2_ParFESpace(pmesh_,order_-1,dim_);

   ak_ = new ParBilinearForm(H1FESpace_);
   ak_->AddDomainIntegrator(new DiffusionIntegrator(*kCoef_));

   rkCoef_ = new BdrRestrictedCoefficient(*kCoef_,ess_bdr2_);
   rCoef_ = new BdrRestrictedCoefficient(ess_bdr2_);
   gk_ = new ParLinearForm(H1FESpace_);
   gk_->AddBdrFaceIntegrator(new BdrGradIntegrator(*rkCoef_));

   // rkCoef1_ = new BdrRestrictedCoefficient(*kCoef_,ess_bdr_);
   // gk1_ = new ParLinearForm(H1FESpace_);
   // gk1_->AddBdrFaceIntegrator(new BdrGradIntegrator(*rkCoef1_));

   t_   = new ParGridFunction(H1FESpace_);
   rhs_ = new ParGridFunction(H1FESpace_);
   k_   = new ParGridFunction(H1FESpace_);
   w_   = new ParGridFunction(H1FESpace_);

   this->InitSecondaryObjects();
}

ThermalResistivity::~ThermalResistivity()
{
   delete T_;
   delete RHS_;
   delete A_;

   delete t_;
   delete rhs_;
   delete k_;
   delete w_;

   delete ak_;
   delete gk_;
   // delete gk1_;
   delete rkCoef_;
   delete rCoef_;
   // delete rkCoef1_;

   delete H1FESpace_;
   delete L2FESpace_;

   delete pmesh_;
}

void
ThermalResistivity::SetConductivityCoef(Coefficient & kCoef)
{
   kCoef_ = &kCoef;
   // cout << 0 << endl;
   delete ak_;
   ak_ = new ParBilinearForm(H1FESpace_);
   ak_->AddDomainIntegrator(new DiffusionIntegrator(*kCoef_));
   ak_->Assemble();
   ak_->Finalize();
   // cout << 1 << endl;
   delete A_;
   A_ = ak_->ParallelAssemble();
   // cout << 2 << endl;
   delete rkCoef_;
   rkCoef_ = new BdrRestrictedCoefficient(*kCoef_,ess_bdr2_);
   // cout << 3 << endl;
   delete gk_;
   cout << 3.1 << endl;
   gk_ = new ParLinearForm(H1FESpace_);
   // cout << 3.2 << endl;
   gk_->AddBdrFaceIntegrator(new BdrGradIntegrator(*rkCoef_));
   // cout << 3.3 << endl;
   gk_->Assemble();
   // cout << 4 << endl;

   // delete rkCoef1_;
   // rkCoef1_ = new BdrRestrictedCoefficient(*kCoef_,ess_bdr_);
   // delete gk1_;
   // cout << 4.1 << endl;
   // gk1_ = new ParLinearForm(H1FESpace_);
   // cout << 4.2 << endl;
   // gk1_->AddBdrFaceIntegrator(new BdrGradIntegrator(*rkCoef1_));
   // cout << 4.3 << endl;
   // gk1_->Assemble();
}

void
ThermalResistivity::UpdateAndRebalance()
{
   // cout << 0 << endl;
   delete T_;
   delete RHS_;
   delete A_;
   // cout << 1 << endl;
   H1FESpace_->Update();
   L2FESpace_->Update();
   // cout << 2 << endl;
   t_->Update();
   rhs_->Update();
   k_->Update();
   w_->Update();
   // cout << 3 << endl;
   if ( pmesh_->Nonconforming() && numProcs_ > 1 )
   {
      //  cout << 4 << endl;
      pmesh_->Rebalance();
      //  cout << 5 << endl;
      H1FESpace_->Update();
      L2FESpace_->Update();
      // cout << 6 << endl;
      t_->Update();
      rhs_->Update();
      k_->Update();
      w_->Update();
   }
   // cout << 7 << endl;
   gk_->Update();
   // gk1_->Update();
   ak_->Update();
   // cout << 8 << endl;
   H1FESpace_->UpdatesFinished();
   L2FESpace_->UpdatesFinished();
   // cout << 9 << endl;
   this->InitSecondaryObjects();
   // cout << 10 << endl;
}

void
ThermalResistivity::InitSecondaryObjects()
{
   // cout << "iso 0" << endl;
   H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_tdof_list_);
   // cout << "iso 1" << endl;
   ak_->Assemble();
   ak_->Finalize();
   A_ = ak_->ParallelAssemble();
   // cout << "iso 2" << endl;
   gk_->Assemble();
   // gk1_->Assemble();
   // cout << "iso 3" << endl;
   T_   = new HypreParVector(H1FESpace_);
   RHS_ = new HypreParVector(H1FESpace_);
   // cout << "iso 4" << endl;
}

void
ThermalResistivity::Solve(ParGridFunction * w)
{
   /// Set the Temperature to 0.0 on surface 1 and 1.0 pn surface 2
   ConstantCoefficient one(1.0);
   *t_ = 0.0;
   t_->ProjectBdrCoefficient(one, ess_bdr2_);

   *rhs_ = 0.0;

   ak_->FormLinearSystem(ess_tdof_list_, *t_, *rhs_, *A_, *T_, *RHS_);

   HypreBoomerAMG amg(*A_);
   amg.SetPrintLevel(0);
   HyprePCG pcg(*A_);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(200);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(amg);
   pcg.Mult(*RHS_, *T_);

   ak_->RecoverFEMSolution(*T_, *rhs_, *t_);

   if ( w != NULL )
   {
      /// Compute the solution to the adjoint problem
      // *w = 0.0;
      *w = *gk_;
      // *w *= -1.0;
      // *gk_ *= -1.0;
      // ConstantCoefficient wbc(-12.0);
      // w_->ProjectBdrCoefficient(wbc, ess_bdr2_);
      ak_->FormLinearSystem(ess_tdof_list_, *w, *gk_, *A_, *T_, *RHS_);
      pcg.Mult(*RHS_, *T_);
      ak_->RecoverFEMSolution(*T_, *gk_, *w);
      // *gk_ *= -1.0;
   }
}

double
ThermalResistivity::GetResistivity(const bool & ref, Vector * dR)
{
   // double tol = 1.0e-2;
   double R1 = this->CalcResistivity(dR);
   double R0 = 2.0 * R1;
   double err_R = 1.0;

   if ( myid_ == 0 )
   {
      cout << "R = " << R1 << endl;
   }

   double max_elem_error = 1.0e-4;
   double hysteresis = 0.25;
   int nc_limit = 3;

   /// Start by checking for possible derefinements
   if ( pmesh_->Nonconforming() && ref )
   {
      double threshold = hysteresis * max_elem_error;

      this->EstimateErrors();

      if (pmesh_->DerefineByError(errors_, threshold, nc_limit))
      {
         if (myid_ == 0)
         {
            cout << "\nDerefined elements." << endl;
         }
         // 12a. Update the space and the solution, rebalance the mesh.
         // UpdateAndRebalance(pmesh, fespace, x, a, b);
         this->UpdateAndRebalance();

         R0 = R1;
         R1 = this->CalcResistivity(dR);
         err_R = 2.0 * fabs( R1 - R0 ) / ( R1 + R0 );
      }
   }

   /// Now enter a refinement loop until the resistivity stops changing
   while ( err_R > tol_ && ref )
   {
      double tot_err = this->EstimateErrors();
      if ( myid_ == 0 )
      {
         cout << "Error:  " << tot_err << endl;
      }
      if (!pmesh_->RefineByError(errors_, max_elem_error, -1, nc_limit))
      {
         if (myid_ == 0)
         {
            cout << "\nReducing error tolerance" << endl;
         }

         max_elem_error /= 2.0;
      }
      else
      {
         if (myid_ == 0)
         {
            cout << "\nRefined elements." << endl;
         }

         this->UpdateAndRebalance();

         R0 = R1;
         R1 = this->CalcResistivity(dR);
         err_R = 2.0 * fabs( R1 - R0 ) / ( R1 + R0 );
         if ( myid_ == 0 )
         {
            cout << "R = " << R1 << ", rel change = " << err_R << ", tol = " << tol_ <<
                 endl;
         }
      }
   }

   return R1;
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
      // cout << i << ": " << 1 << endl;
      // cout << gVec.Size() << " " << tVec.Size() << endl;
      dF[l2Vdofs[0]] = gVec * tVec;
      // cout << i << ": " << 2<< endl;
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
      // cout << i << ": " << 0 << endl;
      dF[l2Vdofs[0]] -= wVec * stVec;

      dF[l2Vdofs[0]] /= L2FESpace_->GetElementTransformation(i)->Weight();
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
      *dR *= R / glb_flux;
   }

   return R;
}

double
ThermalResistivity::EstimateErrors()
{
   // Space for the discontinuous (original) flux
   DiffusionIntegrator flux_integrator;
   L2_FECollection flux_fec(order_, dim_);
   ParFiniteElementSpace flux_fes(pmesh_, &flux_fec, dim_);

   // Space for the smoothed (conforming) flux
   double norm_p = 1;

   FiniteElementCollection * smooth_flux_fec = NULL;
   if ( dim_ > 1 )
   {
      smooth_flux_fec = new RT_FECollection(order_-1, dim_);
   }
   else
   {
      smooth_flux_fec = new H1_FECollection(order_, dim_);
   }
   ParFiniteElementSpace smooth_flux_fes(pmesh_, smooth_flux_fec);

   // Another possible set of options for the smoothed flux space:
   // norm_p = 1;
   // H1_FECollection smooth_flux_fec(order, dim);
   // ParFiniteElementSpace smooth_flux_fes(&pmesh, &smooth_flux_fec, dim);
   double err = L2ZZErrorEstimator(flux_integrator, *t_,
                                   smooth_flux_fes, flux_fes, errors_, norm_p);

   delete smooth_flux_fec;
   return err;
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

double kFunc(const Vector &x)
{
   double k0 = 0.025; /// air at 1 atm and 290 K
   // double k1 = 1.0;   /// fiber reinforced plastic at 290 K
   double k1 = 237.0; /// aluminum at 290 K

   double y_data[3];
   Vector y(y_data,3); y = 0.0;
   for (int i=0; i<x.Size(); i++) { y[i] = x[i]; }

   switch ( prob_ )
   {
      case 1:
         if ( y(0) < 0.25 && y(1) < 0.25 && y(2) < 0.25 ) { return k1; }
         if ( y(0) > 0.25 && y(1) > 0.25 && y(2) < 0.25 ) { return k1; }
         if ( y(0) > 0.25 && y(1) < 0.25 && y(2) > 0.25 ) { return k1; }
         if ( y(0) < 0.25 && y(1) > 0.25 && y(2) > 0.25 ) { return k1; }
         break;
      case 3:
         // Spherical Shell and 3 Rods
      {
         double r1 = 0.14, r2 = 0.36, r3 = 0.105;
         if ( y.Norml2() <= r1 ) { return k0; }
         if ( y.Norml2() <= r2 ) { return k1; }
         if ( sqrt(y(1)*y(1)+y(2)*y(2)) <= r3 ) { return k1; }
         if ( sqrt(y(2)*y(2)+y(0)*y(0)) <= r3 ) { return k1; }
         if ( sqrt(y(0)*y(0)+y(1)*y(1)) <= r3 ) { return k1; }
      }
      break;
      case 4:
         // Spherical Shell and 4 Rods
      {
         double r1 = 0.14, r2 = 0.28, r3 = 0.1;
         if ( y.Norml2() <= r1 ) { return k0; }
         if ( y.Norml2() <= r2 ) { return k1; }
         {
            double rr = y(0)*y(0)+y(1)*y(1)+y(2)*y(2);
            if ( sqrt(rr-y(0)*y(1)-y(1)*y(2)-y(2)*y(0)) <= r3 ) { return k1; }
            if ( sqrt(rr+y(0)*y(1)+y(1)*y(2)-y(2)*y(0)) <= r3 ) { return k1; }
            if ( sqrt(rr+y(0)*y(1)-y(1)*y(2)+y(2)*y(0)) <= r3 ) { return k1; }
            if ( sqrt(rr-y(0)*y(1)+y(1)*y(2)+y(2)*y(0)) <= r3 ) { return k1; }
         }
      }
      break;
      case 6:
         // Spherical Shell and 6 Rods
      {
         double r1 = 0.12, r2 = 0.19, r3 = 0.08;
         if ( y.Norml2() <= r1 ) { return k0; }
         if ( y.Norml2() <= r2 ) { return k1; }
         {
            double rr = y(0)*y(0)+y(1)*y(1)+y(2)*y(2);
            if ( sqrt(rr+y(0)*y(0)-2.0*y(1)*y(2)) <= r3 ) { return k1; }
            if ( sqrt(rr+y(0)*y(0)+2.0*y(1)*y(2)) <= r3 ) { return k1; }
            if ( sqrt(rr+y(1)*y(1)-2.0*y(2)*y(0)) <= r3 ) { return k1; }
            if ( sqrt(rr+y(1)*y(1)+2.0*y(2)*y(0)) <= r3 ) { return k1; }
            if ( sqrt(rr+y(2)*y(2)-2.0*y(0)*y(1)) <= r3 ) { return k1; }
            if ( sqrt(rr+y(2)*y(2)+2.0*y(0)*y(1)) <= r3 ) { return k1; }
         }
      }
      break;
   }

   return 0.025;
}
