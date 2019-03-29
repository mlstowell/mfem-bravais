#include "mfem.hpp"
#include "../common/bravais.hpp"
#include <fstream>
#include <iostream>
#include <cerrno>      // errno

#include "meta_material_solver.hpp"

#ifndef _WIN32
#include <sys/stat.h>  // mkdir
#else
#include <direct.h>    // _mkdir
#define mkdir(dir, mode) _mkdir(dir)
#endif

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;
using namespace mfem::bravais;

// Volume Fraction Coefficient
static int prob_ = -1;
double vol_frac_coef(const Vector &);

int CreateDirectory(const string &dir_name, MPI_Comm & comm, int myid);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "./periodic-unit-cube.mesh";
   int lattice_type = 1;
   string lattice_label = "PC";
   int order = 1;
   int sr = 0, pr = 2;
   bool visualization = 1;
   bool visit = true;
   double a = 1.0;
   // double lambda = 2.07748e+9;
   // double mu = 0.729927e+9;
   // Gallium Arsenide at T=300K
   double lambda = 5.34e+11;
   double mu = 3.285e+11;

   OptionsParser args(argc, argv);
   args.AddOption(&lattice_type, "-bl", "--bravais-lattice",
                  "Bravais Lattice Type: "
                  " 1 - Primitive Cubic,"
                  " 2 - Body-Centered Cubic,"
                  " 3 - Face-Centered Cubic");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sr, "-sr", "--serial-refinement",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-pr", "--parallel-refinement",
                  "Number of parallel refinement levels.");
   args.AddOption(&prob_, "-p", "--problem-type",
                  "Problem Geometry.");
   args.AddOption(&a, "-a", "--lattice-size",
                  "Lattice Size");
   args.AddOption(&lambda, "-l", "--lambda",
                  "Lambda");
   args.AddOption(&mu, "-m", "--mu",
                  "Mu");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visualization",
                  "Enable or disable VisIt visualization.");
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

   BravaisLattice * bravais = NULL;

   switch (lattice_type)
   {
      case 1:
         // Primitive Cubic Lattice
         mesh_file = "./periodic-unit-cube.mesh";
         lattice_label = "PC";
         bravais = new CubicLattice(a);
         // nev = 30;
         break;
      case 2:
         // Body-Centered Cubic Lattice
         mesh_file = "./periodic-unit-truncated-octahedron.mesh";
         lattice_label = "BCC";
         bravais = new BodyCenteredCubicLattice(a);
         // nev = 54;
         break;
      case 3:
         // Face-Centered Cubic Lattice
         mesh_file = "./periodic-unit-rhombic-dodecahedron.mesh";
         lattice_label = "FCC";
         bravais = new FaceCenteredCubicLattice(a);
         // nev = 38;
         break;
      default:
         if (myid == 0)
         {
            cout << "Unsupported Lattice Type:  " << lattice_type << endl << flush;
         }
         MPI_Finalize();
         return 1;
         break;
   }

   ostringstream oss_prefix;
   oss_prefix << "Meta-Material-" << lattice_label;

   CreateDirectory(oss_prefix.str(),comm,myid);

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      if (myid == 0)
      {
         cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      }
      MPI_Finalize();
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   int euler = mesh->EulerNumber();
   if ( myid == 0 ) { cout << "Initial Euler Number:  " << euler << endl; }
   mesh->CheckElementOrientation(false);
   mesh->CheckBdrElementOrientation(false);

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
         int euler = mesh->EulerNumber();
         if ( myid == 0 )
         {
            cout << l+1 << ", Refined Euler Number:  " << euler << endl;
         }
         mesh->CheckElementOrientation(false);
         mesh->CheckBdrElementOrientation(false);
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = pr;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   L2_ParFESpace * L2FESpace    = new L2_ParFESpace(pmesh, 0,
                                                    pmesh->Dimension());

   int nElems = L2FESpace->GetVSize();
   cout << myid << ": nElems = " << nElems << endl;

   ParGridFunction * vf0 = new ParGridFunction(L2FESpace);
   // ParGridFunction * vf1 = new ParGridFunction(L2FESpace);

   FunctionCoefficient vfFunc(vol_frac_coef);
   vf0->ProjectCoefficient(vfFunc);
   /*
   vf1->ProjectCoefficient(vfFunc);

   double vf13 = (*vf0)[13];
   double dvf = 0.01 * (0.5 - vf13);
   (*vf1)[13] += dvf;
   */
   VisData vd("localhost", 19916, 1440, 900, 238, 238, 10, 45);

   if (visualization)
   {
      socketstream vf_sock;
      VisualizeField(vf_sock, *vf0, "Volume Fraction 0", vd);
      vd.IncrementWindow();
      // socketstream vf1_sock;
      // VisualizeField(vf1_sock, *vf1, "Volume Fraction 1", vd);
      // vd.IncrementWindow();
   }

   meta_material::Density density(*pmesh, bravais->GetUnitCellVolume(),
                                  0.0, 10.0);

   density.SetVolumeFraction(*vf0);

   vector<double> rho;
   density.GetHomogenizedProperties(rho);
   if ( myid == 0 )
   {
      ostringstream oss;
      oss << oss_prefix.str() << "/density.dat";
      ofstream ofs(oss.str().c_str());

      cout << "Effective Density:  ";
      for (unsigned int i=0; i<rho.size(); i++)
      {
         cout << rho[i]; ofs << rho[i];
         if ( i < rho.size()-1 ) { cout << ", "; ofs << "\t"; }
      }
      cout << endl; ofs << endl;
      ofs.close();
   }

   if (visualization)
   {
      density.InitializeGLVis(vd);
      density.DisplayToGLVis();
   }
   if ( visit )
   {
      density.WriteVisItFields(oss_prefix.str(), "Density");
   }

   meta_material::StiffnessTensor elasticity(*pmesh,
                                             bravais->GetUnitCellVolume(),
                                             0.0, 0.5,
                                             lambda, mu);

   elasticity.SetVolumeFraction(*vf0);

   vector<double> elas;
   elasticity.GetHomogenizedProperties(elas);
   if ( myid == 0 )
   {
      ostringstream oss;
      oss << oss_prefix.str() << "/stiffness_tensor.dat";
      ofstream ofs(oss.str().c_str());

      cout << "Effective Elasticity Tensor:  " << endl;
      int k = 0;
      for (unsigned int i=0; i<6; i++)
      {
         for (unsigned int j=0; j<i; j++)
         {
            cout << " -----------";
            ofs << elas[(11 - j) * j / 2 + i] << "\t";
         }
         for (unsigned int j=i; j<6; j++)
         {
            cout << " " << elas[k]; ofs << elas[k];
            if ( k < 20 ) { ofs << "\t"; }
            k++;
         }
         cout << endl; ofs << endl;
      }
      cout << endl;
      ofs.close();
   }
   if (visualization)
   {
      elasticity.InitializeGLVis(vd);
      elasticity.DisplayToGLVis();
   }
   if ( visit )
   {
      elasticity.WriteVisItFields(oss_prefix.str(), "StiffnessTensor");
   }
   /*
   vector<ParGridFunction> dRho;
   density.GetPropertySensitivities(dRho);

   vector<ParGridFunction> dElas;
   elasticity.GetPropertySensitivities(dElas);

   if (visualization)
   {
      socketstream drho_sock;
      VisualizeField(drho_sock, dRho[0], "dRho", vd);
      vd.IncrementWindow();

      socketstream delas_sock[21];
      for (int i=0; i<21; i++)
      {
         delas_sock[i].precision(8);
         ostringstream oss;
         oss << "dElas[" << i << "]";
         VisualizeField(delas_sock[i], dElas[i], oss.str().c_str(), vd);
         vd.IncrementWindow();
      }
   }

   density.SetVolumeFraction(*vf1);

   vector<double> new_rho;
   density.GetHomogenizedProperties(new_rho);

   cout << "Rho:  " << rho[0] << " -> " << new_rho[0]
        << ", " << dRho[0][13] * dvf
        << " vs " << new_rho[0] - rho[0] << endl;

   elasticity.SetVolumeFraction(*vf1);

   vector<double> new_elas;
   elasticity.GetHomogenizedProperties(new_elas);

   for (unsigned int i=0; i<new_elas.size(); i++)
   {
      cout << "Elas " << i << ":  " << elas[i] << " -> " << new_elas[i]
           << ", " << dElas[i][13] * dvf
           << " vs " << new_elas[i] - elas[i] << endl;
   }
   */
   delete vf0;
   // delete vf1;
   delete L2FESpace;
   delete pmesh;

   MPI_Finalize();

   if ( myid == 0 )
   {
      cout << "Exiting Main" << endl;
   }

   return 0;
}

int CreateDirectory(const string &dir_name, MPI_Comm & comm, int myid)
{
   int err;
#ifndef MFEM_USE_MPI
   err = mkdir(dir_name.c_str(), 0775);
   err = (err && (errno != EEXIST)) ? 1 : 0;
#else
   if (myid == 0)
   {
      err = mkdir(dir_name.c_str(), 0775);
      err = (err && (errno != EEXIST)) ? 1 : 0;
      MPI_Bcast(&err, 1, MPI_INT, 0, comm);
   }
   else
   {
      // Wait for rank 0 to create the directory
      MPI_Bcast(&err, 1, MPI_INT, 0, comm);
   }
#endif
   return err;
}

double
distToLine(double ox, double oy, double oz,
           double tx, double ty, double tz, const Vector & x)
{
   double xo_data[3];
   double xt_data[3];
   Vector xo(xo_data, 3);
   Vector xt(xt_data, 3);

   // xo = x - {ox,oy,oz}
   xo[0] = x[0] - ox;
   xo[1] = x[1] - oy;
   xo[2] = x[2] - oz;

   // xt = cross_product({tx,ty,tz}, xo)
   xt[0] = ty * xo[2] - tz * xo[1];
   xt[1] = tz * xo[0] - tx * xo[2];
   xt[2] = tx * xo[1] - ty * xo[0];

   return xt.Norml2();
}
double
vol_frac_coef(const Vector & x)
{
   switch ( prob_ )
   {
      case -1:
         // Uniform
         return 1.0;
         break;
      case 0:
         // Slab
         if ( fabs(x(0)) <= 0.25 ) { return 1.0; }
         break;
      case 1:
         // Cylinder
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= 0.5 ) { return 1.0; }
         break;
      case 2:
         // Sphere
         if ( x.Norml2() <= 0.5 ) { return 1.0; }
         break;
      case 3:
         // Sphere and 3 Rods
      {
         double r1 = 0.14, r2 = 0.36, r3 = 0.105;
         if ( x.Norml2() <= r1 ) { return 0.0; }
         if ( x.Norml2() <= r2 ) { return 1.0; }
         if ( sqrt(x(1)*x(1)+x(2)*x(2)) <= r3 ) { return 1.0; }
         if ( sqrt(x(2)*x(2)+x(0)*x(0)) <= r3 ) { return 1.0; }
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= r3 ) { return 1.0; }
      }
      break;
      case 4:
         // Sphere and 4 Rods
      {
         double r1 = 0.14, r2 = 0.28, r3 = 0.1;
         if ( x.Norml2() <= r1 ) { return 0.0; }
         if ( x.Norml2() <= r2 ) { return 1.0; }

         Vector y = x;
         y[0] -= 0.5; y[1] -= 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] -= 0.5; y[1] -= 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] -= 0.5; y[1] += 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] -= 0.5; y[1] += 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] -= 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] -= 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] += 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] += 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }

         double a = r3;
         double b = 1.0/sqrt(3.0);
         if ( distToLine(0.0, 0.0, 0.0, b, b, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0,-b, b, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0,-b,-b, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, b,-b, b, x) <= a ) { return 1.0; }
      }
      break;
      case 5:
         // Two spheres in a BCC configuration
         if ( x.Norml2() <= 0.3 )
         {
            return 1.0;
         }
         else
         {
            for (int i=0; i<8; i++)
            {
               int i1 = i%2;
               int i2 = (i/2)%2;
               int i4 = i/4;

               Vector u = x;
               u(0) -= i1?-0.5:0.5;
               u(1) -= i2?-0.5:0.5;
               u(2) -= i4?-0.5:0.5;

               if ( u.Norml2() <= 0.2 ) { return 1.0; }
            }
         }
         break;
      case 6:
         // Sphere and 6 Rods
      {
         double r1 = 0.12, r2 = 0.19, r3 = 0.08;
         if ( x.Norml2() <= r1 ) { return 0.0; }
         if ( x.Norml2() <= r2 ) { return 1.0; }

         Vector y = x;
         y[0] -= 0.5; y[1] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] -= 0.5; y[1] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }

         y = x; y[1] -= 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[1] -= 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[1] += 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[1] += 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }

         y = x; y[2] -= 0.5; y[0] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[2] -= 0.5; y[0] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[2] += 0.5; y[0] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[2] += 0.5; y[0] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }

         double a = r3;
         double b = 1.0/sqrt(2.0);
         if ( distToLine(0.0, 0.0, 0.0, b, b, 0, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, b,-b, 0, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, b, 0, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, b, 0,-b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, 0, b, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, 0, b,-b, x) <= a ) { return 1.0; }

      }
      break;
      case 7:
         if ( fabs(x(1)) + fabs(x(2)) < 0.25 ||
              fabs(x(0)) + fabs(x(2)) < 0.25 ||
              fabs(x(0)) + fabs(x(1) - 0.5) < 0.25 )
         {
            return 1.0;
         }
         break;
   }
   return 0.0;
}
