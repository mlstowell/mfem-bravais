#include "mfem.hpp"
#include "bravais.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;
using namespace mfem::bravais;

int main(int argc, char ** argv)
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
   double a = 1.0;

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

   BravaisSymmetryPoints * bravais = NULL;

   switch (lattice_type)
   {
      case 1:
         // Primitive Cubic Lattice
         mesh_file = "./periodic-unit-cube.mesh";
         lattice_label = "PC";
         bravais = new CubicSymmetryPoints(a);
         break;
      case 2:
         // Body-Centered Cubic Lattice
         mesh_file = "./periodic-unit-truncated-octahedron.mesh";
         lattice_label = "BCC";
         bravais = new BodyCenteredCubicSymmetryPoints(a);
         break;
      case 3:
         // Face-Centered Cubic Lattice
         mesh_file = "./periodic-unit-rhombic-dodecahedron.mesh";
         lattice_label = "FCC";
         bravais = new FaceCenteredCubicSymmetryPoints(a);
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

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   if (myid == 0)
   {
      cerr << "\nReading mesh file: " << mesh_file << endl;
   }
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

   cout << "Initial Euler Number:  " << mesh->EulerNumber() << endl;
   mesh->CheckElementOrientation(false);
   mesh->CheckBdrElementOrientation(false);

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
         cout << l+1 << ", Refined Euler Number:  " << mesh->EulerNumber() << endl;
         mesh->CheckElementOrientation(false);
         mesh->CheckBdrElementOrientation(false);
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(comm, *mesh);
   delete mesh;
   {
      int par_ref_levels = pr;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   cout << "Creating H1_FESpace" << endl;
   H1_ParFESpace * H1FESpace    = new H1_ParFESpace(pmesh, order,
                                                    pmesh->Dimension());
   ND_ParFESpace * HCurlFESpace = new ND_ParFESpace(pmesh, order,
                                                    pmesh->Dimension());
   RT_ParFESpace * HDivFESpace  = new RT_ParFESpace(pmesh, order,
                                                    pmesh->Dimension());
   L2_ParFESpace * L2FESpace    = new L2_ParFESpace(pmesh, order-1,
                                                    pmesh->Dimension());

   cout << "Creating H1FourierSeries" << endl;
   H1FourierSeries       fourier_h1(*bravais, *H1FESpace);
   HCurlFourierSeries fourier_hcurl(*bravais, *HCurlFESpace);
   HDivFourierSeries   fourier_hdiv(*bravais, *HDivFESpace);
   L2FourierSeries       fourier_l2(*bravais, *L2FESpace);

   cout << "Creating ParGridFunction" << endl;
   ParGridFunction x_h1(H1FESpace);
   ParGridFunction x_hcurl(HCurlFESpace);
   ParGridFunction x_hdiv(HDivFESpace);
   ParGridFunction x_l2(L2FESpace);

   HypreParVector  X_H1(H1FESpace);
   HypreParVector  X_HCurl(HCurlFESpace);
   HypreParVector  X_HDiv(HDivFESpace);
   HypreParVector  X_L2(L2FESpace);

   cout << "Creating RealModeCoefficient" << endl;
   RealModeCoefficient sCoef;
   sCoef.SetAmplitude(1.0);
   sCoef.SetModeIndices(-1,0,1);

   Vector v(3); v[0] = 1.0; v[1] = 2.0; v[2] = 3.0;
   VectorFunctionCoefficient vCoef(v, sCoef);

   vector<Vector> b;
   bravais->GetReciprocalLatticeVectors(b);
   sCoef.SetReciprocalLatticeVectors(b);

   cout << "Projecting Coefficient" << endl;
   x_h1.ProjectCoefficient(sCoef);
   x_hcurl.ProjectCoefficient(vCoef);
   x_hdiv.ProjectCoefficient(vCoef);
   x_l2.ProjectCoefficient(sCoef);

   cout << "Projecting onto Parallel vector" << endl;
   x_h1.ParallelProject(X_H1);
   x_hcurl.ParallelProject(X_HCurl);
   x_hdiv.ParallelProject(X_HDiv);
   x_l2.ParallelProject(X_L2);

   Vector ar_hcurl(3), ai_hcurl(3), ar_hdiv(3), ai_hdiv(3);

   cout << "Computing Fourier Coefficients" << endl;
   for (int i=-1; i<2; i++)
   {
      for (int j=-1; j<2; j++)
      {
         for (int k=-1; k<2; k++)
         {
            double ar_h1, ai_h1, ar_l2, ai_l2;
            fourier_h1.SetMode(i, j, k);
            fourier_hcurl.SetMode(i, j, k);
            fourier_hdiv.SetMode(i, j, k);
            fourier_l2.SetMode(i, j, k);
            fourier_h1.GetCoefficient(X_H1, ar_h1, ai_h1);
            fourier_hcurl.GetCoefficient(X_HCurl, ar_hcurl, ai_hcurl);
            fourier_hdiv.GetCoefficient(X_HDiv, ar_hdiv, ai_hdiv);
            fourier_l2.GetCoefficient(X_L2, ar_l2, ai_l2);

            if ( myid == 0 )
            {
               cout << "(" << i << "," << j << "," << k << "):  "
                    << ar_h1 << " + " << ai_h1 << " i, "
                    << ar_l2 << " + " << ai_l2 << " i" << endl;
               for (int l=0; l<3; l++)
               {
                  cout << "\t" << ar_hcurl[l] << " + " << ai_hcurl[l] << " i, "
                       << ar_hdiv[l] << " + " << ai_hdiv[l] << " i" << endl;
               }
            }
         }
      }
   }

   delete H1FESpace;
   delete HCurlFESpace;
   delete HDivFESpace;
   delete L2FESpace;
   delete bravais;
   delete pmesh;

   MPI_Finalize();

   cout << "Exiting Main" << endl;

   return 0;
}
