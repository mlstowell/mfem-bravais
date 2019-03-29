#include "mfem.hpp"
#include "bravais.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;
using namespace mfem::bravais;

class PhaseCoef: public Coefficient
{
public:
   PhaseCoef(Vector & kappa) { kappa_ = kappa; }

   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);

      T.Transform(ip, transip);

      return transip * kappa_;
   }

private:
   Vector kappa_;
};

int main(int argc, char ** argv)
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // 2. Parse command-line options.
   int lattice_type = 1;
   string lattice_label = "PC";
   int order = 1;
   int sr = 0, pr = 0, nr = 3;
   double a = -1.0, b = -1.0, c = -1.0;
   double alpha = -1.0, beta = -1.0, gamma = -1.0;
   double frac = 0.5;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&lattice_type, "-bl", "--bravais-lattice",
                  "Bravais Lattice Type: \n"
                  "  1 - Primitive Cubic (a),\n"
                  "  2 - Face-Centered Cubic (a),\n"
                  "  3 - Body-Centered Cubic (a),\n"
                  "  4 - Tetragonal (a, c),\n"
                  "  5 - Body-Centered Tetragonal (a, c),\n"
                  "  6 - Orthorhombic (a < b < c),\n"
                  "  7 - Face-Centered Orthorhombic (a < b < c),\n"
                  "  8 - Body-Centered Orthorhombic (a < b < c),\n"
                  "  9 - C-Centered Orthorhombic (a < b, c),\n"
                  " 10 - Hexagonal Prism (a, c),\n"
                  " 11 - Rhombohedral (a, 0 < alpha < pi),\n"
                  " 12 - Monoclinic (a, b <= c, 0 < alpha < pi/2),\n"
                  " 13 - C-Centered Monoclinic (a, b <= c, 0 < alpha < pi/2),\n"
                  " 14 - Triclinic (0 < alpha, beta, gamma < pi),\n"
                  " 15 - Oblique (a, b, 0 < gamma < pi),\n"
                  " 16 - Rectangular (a, b),\n"
                  " 17 - Centered Rectangular (a, b),\n"
                  " 18 - Hexagonal (a),\n"
                  " 19 - Square (a)"
                 );
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&a, "-a", "--a",
                  "");
   args.AddOption(&b, "-b", "--b",
                  "");
   args.AddOption(&c, "-c", "--c",
                  "");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "");
   args.AddOption(&beta, "-beta", "--beta",
                  "");
   args.AddOption(&gamma, "-gamma", "--gamma",
                  "");
   args.AddOption(&frac, "-f", "--frac",
                  "Fraction of inscribed circle radius for rods");
   args.AddOption(&sr, "-sr", "--serial-refinement",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-pr", "--parallel-refinement",
                  "Number of parallel refinement levels.");
   args.AddOption(&nr, "-nr", "--non-conforming-refinement",
                  "Number of non-conforming refinement levels.");
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

   BravaisLattice * bravais = NULL;

   switch (lattice_type)
   {
      case 1:
         // Cubic Lattice
         lattice_label = "CUB";
         if ( a <= 0.0 ) { a = 1.0; }
         bravais = new CubicLattice(a);
         break;
      case 2:
         // Face-Centered Cubic Lattice
         lattice_label = "FCC";
         if ( a <= 0.0 ) { a = 1.0; }
         bravais = new FaceCenteredCubicLattice(a);
         break;
      case 3:
         // Body-Centered Cubic Lattice
         lattice_label = "BCC";
         if ( a <= 0.0 ) { a = 1.0; }
         bravais = new BodyCenteredCubicLattice(a);
         break;
      case 4:
         // Tetragonal Lattice
         lattice_label = "TET";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( c <= 0.0 ) { c = 0.5; }
         bravais = new TetragonalLattice(a, c);
         break;
      case 5:
         // Body-centered Tetragonal Lattice
         lattice_label = "BCT";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( c <= 0.0 ) { c = 0.5; }
         bravais = new BodyCenteredTetragonalLattice(a, c);
         break;
      case 6:
         // Orthorhombic Lattice
         lattice_label = "ORC";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 0.8; }
         if ( c <= 0.0 ) { c = 1.0; }
         bravais = new OrthorhombicLattice(a, b, c);
         break;
      case 7:
         // Face-centered Orthorhombic Lattice
         lattice_label = "ORCF";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 0.8; }
         if ( c <= 0.0 ) { c = 1.0; }
         bravais = new FaceCenteredOrthorhombicLattice(a, b, c);
         break;
      case 8:
         // Body-centered Orthorhombic Lattice
         lattice_label = "ORCI";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 0.8; }
         if ( c <= 0.0 ) { c = 1.0; }
         bravais = new BodyCenteredOrthorhombicLattice(a, b, c);
         break;
      case 9:
         // C-Centered Orthorhombic Lattice
         lattice_label = "ORCC";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 1.0; }
         if ( c <= 0.0 ) { c = 1.0; }
         bravais = new BaseCenteredOrthorhombicLattice(a, b, c);
         break;
      case 10:
         // Hexagonal Prism Lattice
         lattice_label = "HEX";
         if ( a <= 0.0 ) { a = 1.0; }
         if ( c <= 0.0 ) { c = 1.0; }
         bravais = new HexagonalPrismLattice(a, c);
         break;
      case 11:
         // Rhombohedral Lattice
         lattice_label = "RHL";
         if (     a <= 0.0 ) { a = 1.0; }
         if ( alpha <= 0.0 ) { alpha = 0.25 * M_PI; }
         bravais = new RhombohedralLattice(a, alpha);
         break;
      case 12:
         // Monoclinic Lattice
         lattice_label = "MCL";
         if (     a <= 0.0 ) { a = 1.0; }
         if (     b <= 0.0 ) { b = 1.0; }
         if (     c <= 0.0 ) { c = 1.0; }
         if ( alpha <= 0.0 ) { alpha = 0.25 * M_PI; }
         bravais = new MonoclinicLattice(a, b, c, alpha);
         break;
      case 13:
         // C-centered Monoclinic Lattice
         lattice_label = "MCLC";
         if (     a <= 0.0 ) { a = 1.0; }
         if (     b <= 0.0 ) { b = 1.0; }
         if (     c <= 0.0 ) { c = 1.0; }
         if ( alpha <= 0.0 ) { alpha = 0.25 * M_PI; }
         bravais = new BaseCenteredMonoclinicLattice(a, b, c, alpha);
         break;
      case 14:
         // Triclinic Lattice
         lattice_label = "TRI";
         if (     a <= 0.0 ) { a = 1.0; }
         if (     b <= 0.0 ) { b = 1.0; }
         if (     c <= 0.0 ) { c = 1.0; }
         if ( alpha <= 0.0 ) { alpha = 0.25 * M_PI; }
         if (  beta <= 0.0 ) { beta = 0.25 * M_PI; }
         if ( gamma <= 0.0 ) { gamma = 0.25 * M_PI; }
         // bravais = new TriclinicLattice(a, b, c, alpha, beta, gamma);
         break;
      case 15:
         // Oblique Lattice
         lattice_label = "OBL";
         if ( a <= 0.0 ) { a = 1.0; }
         break;
      case 16:
         // Rectangular Lattice
         lattice_label = "RECT";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 1.0; }
         break;
      case 17:
         // Centered Rectangular Lattice
         lattice_label = "RECTI";
         if ( a <= 0.0 ) { a = 0.5; }
         if ( b <= 0.0 ) { b = 1.0; }
         break;
      case 18:
         // Hexagonal Lattice
         lattice_label = "HEX2D";
         if ( a <= 0.0 ) { a = 1.0; }
         bravais = new HexagonalLattice(a);
         break;
      case 19:
         // Primitive square Lattice
         lattice_label = "SQR";
         if ( a <= 0.0 ) { a = 1.0; }
         bravais = new SquareLattice(a);
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
   if ( myid == 0 )
   {
      cout << endl;
      cout << "Lattice type: " << lattice_label << endl;
   }

   vector<Vector> lat_vec;
   vector<Vector> rec_vec;
   bravais->GetLatticeVectors(lat_vec);
   bravais->GetReciprocalLatticeVectors(rec_vec);

   if ( myid == 0 )
   {
      cout << endl;
      cout << "Comparing lattice and reciprocal lattice vectors" << endl;
      DenseMatrix m((int)lat_vec.size());
      for (unsigned int i=0; i<lat_vec.size(); i++)
      {
         for (unsigned int j=0; j<lat_vec.size(); j++)
         {
            m(i,j) = lat_vec[i] * rec_vec[j];
         }
      }
      m.Print(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh;
   mesh = bravais->GetCoarseWignerSeitzMesh();
   ofstream ofs("auto.mesh");
   mesh->Print(ofs);
   int dim = mesh->Dimension();

   cout << "Initial Euler Number:  "
        << ((dim==3)?mesh->EulerNumber():mesh->EulerNumber2D()) << endl;
   mesh->CheckElementOrientation(false);
   mesh->CheckBdrElementOrientation(false);

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
         cout << l+1 << ", Refined Euler Number:  "
              << ((dim==3)?mesh->EulerNumber():mesh->EulerNumber2D()) << endl;
         mesh->CheckElementOrientation(false);
         mesh->CheckBdrElementOrientation(false);
      }
   }
   mesh->EnsureNCMesh();

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(comm, *mesh);
   delete mesh;
   /*
   {
      int par_ref_levels = pr;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   */
   ConstantCoefficient cOne(1.0);
   LatticeCoefficient  lCoef(*bravais,frac);
   H1_ParFESpace H1FESpace1(pmesh, 1, pmesh->Dimension());
   L2_ParFESpace L2FESpace0(pmesh, 0, pmesh->Dimension());

   ParGridFunction gfLat_h1(&H1FESpace1);
   ParGridFunction gfLat_l2(&L2FESpace0);

   for (int i=0; i<nr; i++)
   {
      GridFunctionCoefficient gfLatCoef(&gfLat_h1);
      gfLat_h1.ProjectCoefficient(lCoef);
      gfLat_l2.ProjectCoefficient(gfLatCoef);
      /*
      for (int k=0; k<ns; k++)
      {
      Vector kappa;
      bravais->GetSymmetryPoint(k, kappa);
      PhaseCoef phaseCoef(kappa);
      gfPhase[k]->ProjectCoefficient(phaseCoef);
           }
           */
      int num_el_to_refine = 0;
      Array<int> el_to_refine;

      for (int j=0; j< gfLat_l2.Size(); j++)
      {
         if ( 0.5 - fabs(gfLat_l2[j]-0.5) > 0.01 ) { num_el_to_refine++; }
      }
      el_to_refine.SetSize(num_el_to_refine);
      cout << "Found " << num_el_to_refine << " elements to refine"
           << " out of " << pmesh->GetNE() << endl;
      int e = 0;
      for (int j=0; j< gfLat_l2.Size(); j++)
      {
         if ( 0.5 - fabs(gfLat_l2[j]-0.5) > 0.01 )
         {
            el_to_refine[e] = j;
            e++;
         }
      }
      pmesh->GeneralRefinement(el_to_refine);

      H1FESpace1.Update();
      L2FESpace0.Update();
      gfLat_h1.Update();
      gfLat_l2.Update();
   }
   if ( nr > 0 )
   {
      gfLat_l2.ProjectCoefficient(lCoef);
   }
   else
   {
      gfLat_l2.ProjectCoefficient(cOne);
   }

   if ( myid == 0 ) { cout << "Creating L2_FESpace" << endl; }
   H1_ParFESpace H1FESpace(pmesh, order, pmesh->Dimension());
   L2_ParFESpace L2FESpace(pmesh, order, pmesh->Dimension());

   ParGridFunction gfLat(&L2FESpace);
   ParGridFunction gfOne(&L2FESpace);
   /*
   ParGridFunction * sinb_h1[3];
   ParGridFunction * sinb_l2[3];
   ParGridFunction * sinb_pr[3];
   for (int d=0; d<dim; d++)
   {
      sinb_h1[d] = new ParGridFunction(&H1FESpace);
      sinb_l2[d] = new ParGridFunction(&L2FESpace);
      sinb_pr[d] = new ParGridFunction(&L2FESpace);
   }
   */

   ParLinearForm lfOne(&L2FESpace);
   lfOne.AddDomainIntegrator(new DomainLFIntegrator(cOne));
   lfOne.Assemble();

   gfOne.ProjectCoefficient(cOne);

   if ( myid == 0 ) { cout << "Creating ImagModeCoefficient" << endl; }
   ImagModeCoefficient sCoef;
   sCoef.SetAmplitude(1.0);
   sCoef.SetReciprocalLatticeVectors(rec_vec);
   /*
   for (int d=0; d<dim; d++)
   {
      sCoef.SetModeIndices((d==0)?1:0, (d==1)?1:0, (d==2)?1:0);
      sinb_h1[d]->ProjectCoefficient(sCoef);
      sinb_l2[d]->ProjectCoefficient(sCoef);

      GridFunctionCoefficient gfCoef_h1(sinb_h1[d]);
      GridFunctionCoefficient gfCoef_l2(sinb_l2[d]);
      sinb_pr[d]->ProjectCoefficient(gfCoef_h1);
      double err = sinb_pr[d]->ComputeL2Error(gfCoef_l2);

      cout << "Error in H1 vs. L2 projection: " << err << endl;
   }
   */
   double  vol = bravais->GetUnitCellVolume();
   double avol = lfOne * gfOne;

   if ( myid == 0 )
   {
      cout << "Volume:  " << avol << " (should be " << vol << ")" << endl;
   }

   int ns = bravais->GetNumberSymmetryPoints();
   ParGridFunction ** gfPhase = new ParGridFunction*[ns];
   // socketstream ** phaseSocks = new socketstream*[ns];
   for (int k=0; k<ns; k++)
   {
      gfPhase[k] = new ParGridFunction(&L2FESpace);
      //phaseSocks[k] = new socketstream(vishost, visport);
   }

   for (int k=0; k<ns; k++)
   {
      Vector kappa;
      bravais->GetSymmetryPoint(k, kappa);
      PhaseCoef phaseCoef(kappa);
      gfPhase[k]->ProjectCoefficient(phaseCoef);
      cout << bravais->GetSymmetryPointLabel(k)
           << " Kappa: (" << kappa[0] << "," << kappa[1] << "," << kappa[2]
           << ") ";
      cout << kappa.Norml2() << " " << gfPhase[k]->Normlinf()
           << " " << gfPhase[k]->Normlinf()/kappa.Norml2()  << endl;
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      /*
      for (int d=0; d<dim; d++)
      {
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << *sinb_l2[d]
        << "window_title 'L2'" << "keys am" << flush;
      }
      */
      {
         cout << "Displaying lattice" << endl;
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << gfLat_l2
                  << "window_title 'Lattice'" << "keys amppp" << flush;
         /*
           cout << "Displaying phase plots" << endl;
         for (int k=0; k<ns; k++)
         {
           *phaseSocks[k] << "parallel " << num_procs << " " << myid << "\n";
           phaseSocks[k]->precision(8);
           *phaseSocks[k] << "solution\n" << *pmesh << *gfPhase[k]
                       << "window_title 'Phase'" << "keys amppp" << flush;
         }
         */
      }

      /*
      for (int d=0; d<dim; d++)
      {
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << *sinb_h1[d]
        << "window_title 'H1'" << "keys a" << flush;
      }
      */
   }

   ostringstream vname;
   vname << "Test_Bravais_" << lattice_label;
   VisItDataCollection visit_dc(vname.str().c_str(), pmesh);
   visit_dc.RegisterField("truss", &gfLat_l2);
   for (int k=0; k<ns; k++)
   {
      visit_dc.RegisterField(bravais->GetSymmetryPointLabel(k),gfPhase[k]);
   }
   visit_dc.Save();

   // for (int d=0; d<dim; d++) { delete sinb_h1[d]; }
   // for (int d=0; d<dim; d++) { delete sinb_l2[d]; }
}
