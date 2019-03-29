#include "mfem.hpp"
#include "../common/pfem_extras.hpp"
#include <fstream>
#include <iostream>

#include "thermal_resistivity_solver.hpp"

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;

ParMesh * CartMesh(MPI_Comm & comm, int dim, int n, double a);

class VFCoef : public Coefficient
{
public:
   VFCoef(double r1 = 0.14, double r2 = 0.36, double r3 = 0.105)
      : r1_(r1),
        r2_(r2),
        r3_(r3)
   {}

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      double x_data[3];
      Vector x(x_data, 3);

      T.Transform(ip, x);

      double y_data[3];
      Vector y(y_data,3); y = 0.0;
      for (int i=0; i<x.Size(); i++) { y[i] = x[i]; }

      if ( y.Norml2() <= r1_ ) { return 0.0; }
      if ( y.Norml2() <= r2_ ) { return 1.0; }
      if ( sqrt(y(1)*y(1)+y(2)*y(2)) <= r3_ ) { return 1.0; }
      if ( sqrt(y(2)*y(2)+y(0)*y(0)) <= r3_ ) { return 1.0; }
      if ( sqrt(y(0)*y(0)+y(1)*y(1)) <= r3_ ) { return 1.0; }

      return 0.0;
   }

private:
   double r1_;
   double r2_;
   double r3_;
};

class KCoef : public GridFunctionCoefficient
{
public:
   KCoef(GridFunction * vf, double k0, double k1)
      : GridFunctionCoefficient(vf),
        k0_(k0),
        k1_(k1)
   {}

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return k0_ + k1_ * this->GridFunctionCoefficient::Eval(T, ip);
   }

   double dk() { return k1_; }

private:

   double k0_;
   double k1_;
};

static int prob_ = 0;

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
   int    xi = 0;
   double dx = 0.01;
   bool visualization = true;

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
   args.AddOption(&prob_, "-p", "--problem",
                  "Used to select geometry");
   args.AddOption(&dx, "-dx", "--delta-x",
                  "");
   args.AddOption(&xi, "-xi", "--delta-x-index",
                  "");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
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

   // 3. Create a Cartesian mesh and allocate a volume fraction GridFunction
   ParMesh * pmesh = CartMesh(comm, dim, n, a);
   L2_ParFESpace L2FESpace(pmesh, 0, dim);
   ParGridFunction vf(&L2FESpace);
   VFCoef vfCoef;
   vf.ProjectCoefficient(vfCoef);

   // 4. Initialize the thermal conductivity from the volume fraction
   double k0 = 0.025; /// air at 1 atm and 290 K
   double k1 = 237.0; /// aluminum at 290 K

   KCoef kCoef(&vf, k0, k1);

   // GridFunctionCoefficient kCoef(&k);

   // 5. Setup the solver and compute the Resistivity
   ThermalResistivity tr(comm, *pmesh, L2FESpace, kCoef, dim, order, a);

   ParGridFunction dR(&L2FESpace);

   double R_lambda_0 = tr.GetResistivity(&dR);
   if ( myid == 0 )
   {
      cout << "Specific Resistivity:  " << R_lambda_0 << endl;
   }
   if ( visualization )
   {
      tr.Visualize(&dR);
   }

   // 6. Adjust the volume fraction and recompute the resisitivity
   if ( myid == 0 )
   {
      vf[xi] += dx;
   }
   tr.ConductivityChanged();

   double R_lambda_1 = tr.GetResistivity();

   if ( myid == 0 )
   {
      cout << "Specific Resistivity:  " << R_lambda_1 << endl;
      cout << "Expected dR:  " << dR[xi] * dx * kCoef.dk() << endl;
      cout << "Delta R:      " << R_lambda_1 - R_lambda_0 << endl;
      //cout << "Delta R / d x: " << (R_lambda_1 - R_lambda_0)/dx << endl;
   }

   // 7. Display fields
   if ( visualization )
   {
      tr.Visualize();
   }

   // 8. Exit
   MPI_Finalize();

   return 0;
}

ParMesh *
CartMesh(MPI_Comm & comm, int dim, int n, double a)
{
   Mesh * mesh = NULL;
   switch (dim)
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
   ParMesh * pmesh = new ParMesh(comm, *mesh);
   delete mesh;

   return pmesh;
}
