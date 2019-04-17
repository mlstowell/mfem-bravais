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
//
//      ----------------------------------------------------------------
//      Display Bravais Miniapp:  Visualize Bravais lattice meshes
//      ----------------------------------------------------------------
//
// This miniapp visualizes various types of finite element basis functions on a
// single mesh element in 1D, 2D and 3D. The order and the type of finite
// element space can be changed, and the mesh element is either the reference
// one, or a simple transformation of it. Dynamic creation and interaction with
// multiple GLVis windows is demonstrated.
//
// Compile with: make display-basis
//
// Sample runs:  display-basis
//               display_basis -e 2 -b 3 -o 3
//               display-basis -e 5 -b 1 -o 1

#include "mfem.hpp"
#include "../lib/bravais.hpp"
#include <vector>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::bravais;
using namespace mfem::miniapps;
/*
string   elemTypeStr(const Element::Type & eType);
inline bool elemIs1D(const Element::Type & eType);
inline bool elemIs2D(const Element::Type & eType);
inline bool elemIs3D(const Element::Type & eType);

string   latTypeStr(const BRAVAIS_LATTICE_TYPE & blType);
inline bool latIs1D(const BRAVAIS_LATTICE_TYPE & blType);
inline bool latIs2D(const BRAVAIS_LATTICE_TYPE & blType);
inline bool latIs3D(const BRAVAIS_LATTICE_TYPE & blType);

string   basisTypeStr(char bType);
inline bool basisIs1D(char bType);
inline bool basisIs2D(char bType);
inline bool basisIs3D(char bType);

string mapTypeStr(int mType);
*/
/*
int update_basis(vector<socketstream*> & sock, const VisWinLayout & vwl,
                 Element::Type e, char bType, int bOrder, int mType,
                 Deformation::DefType dType, const DeformationData & defData,
                 bool visualization);
*/
int main(int argc, char *argv[])
{
   // Parse command-line options.
   int lt = (int)PRIMITIVE_SQUARE;
   double a = 1.0;
   double b = 1.0;
   double c = 1.0;
   double alpha = NAN;
   double beta  = NAN;
   double gamma = NAN;

   bool fdMesh = true;

   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&lt, "-lt", "--lattice-type",
                  BravaisLatticeFactory::GetLatticeOptionStr().c_str());
   args.AddOption(&a, "-a", "--a",
                  "Lattice spacing a");
   args.AddOption(&b, "-b", "--b",
                  "Lattice spacing b");
   args.AddOption(&c, "-c", "--c",
                  "Lattice spacing c");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "Angle alpha in degrees");
   args.AddOption(&beta, "-beta", "--beta",
                  "Angle beta in degrees");
   args.AddOption(&gamma, "-gamma", "--gamma",
                  "Angle gamma in degrees");
   args.AddOption(&fdMesh, "-fd", "--fund-domain", "-no-fd",
                  "--no-fund-domain",
                  "Enable or disable meshes derived from "
                  "the fundamental domain.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   {
      args.PrintOptions(cout);
   }
   alpha = (isnan(alpha)) ? 0.0 : (M_PI * alpha / 180.0);
   beta  = (isnan(beta )) ? 0.0 : (M_PI * beta  / 180.0);
   gamma = (isnan(gamma)) ? 0.0 : (M_PI * gamma / 180.0);

   BRAVAIS_LATTICE_TYPE blType = (BRAVAIS_LATTICE_TYPE)lt;

   BravaisLatticeFactory fact;

   BravaisLattice * lat = NULL;
   Mesh * mesh = NULL;

   /*
   socketstream sock;

   if (visualization)
   {
      // GLVis server to visualize to
      char vishost[] = "localhost";
      int  visport   = 19916;
      sock.open(vishost, visport);
      sock.precision(8);
   }
   */
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sock(vishost, visport);
   sock.precision(8);

   // Collect user input
   bool print_char = true;
   while (true)
   {
      delete lat; lat = fact.GetLattice(blType, a, b, c, alpha, beta, gamma);
      if (lat)
      {
         if (fact.Is1D(blType))
         {
            dynamic_cast<BravaisLattice1D*>(lat)->GetAxialLength(a);
         }
         else if (fact.Is2D(blType))
         {
            BravaisLattice2D * lat2d = dynamic_cast<BravaisLattice2D*>(lat);
            lat2d->GetAxialLengths(a, b);
            lat2d->GetInteraxialAngle(gamma);
         }
         else if (fact.Is3D(blType))
         {
            BravaisLattice3D * lat3d = dynamic_cast<BravaisLattice3D*>(lat);
            lat3d->GetAxialLengths(a, b, c);
            lat3d->GetInteraxialAngles(alpha, beta, gamma);
         }

         delete mesh; mesh = lat->GetWignerSeitzMesh(fdMesh);
         if (mesh)
         {
            if (visualization)
            {
               sock << "keys q\n" << flush;
               sock.open(vishost, visport);
               sock << "mesh\n" << *mesh << flush;
               //mesh->Print(sock);
            }
         }
         else
         {
            cerr << "Mesh construction failed.  Try again" << endl;
            if (!sock.is_open()) { sock.open(vishost, visport); }
         }
      }
      else
      {
         cerr << "Lattice construction failed.  Try again" << endl;
         if (!sock.is_open()) { sock.open(vishost, visport); }
      }
      if (print_char)
      {
         cout << endl;
         cout << "Lattice Type:     " << fact.GetLatticeName(blType)
              << " (" << lat->GetLatticeTypeLabel() << ")" << endl;
         cout << "Parameters:       " << lat->GetParameterBoundStr() << endl;
         if (fact.Is1D(blType))
         {
            cout << "Lattice Spacing:  " << a;
         }
         else if (fact.Is2D(blType))
         {
            cout << "Lattice Spacings: " << a << " " << b << endl;
            cout << "Lattice Angle:    " << 180.0 * gamma / M_PI << endl;
         }
         else if (fact.Is3D(blType))
         {
            cout << "Lattice Spacings: " << a << " " << b << " " << c << endl;
            cout << "Lattice Angles:   "
                 << " " << 180.0 * alpha / M_PI
                 << " " << 180.0 * beta / M_PI
                 << " " << 180.0 * gamma / M_PI
                 << endl;
         }
         // cout << "Basis function order:  " << bOrder << endl;
         // cout << "Map Type:              " << mapTypeStr(mType) << endl;
      }
      if (!visualization) { break; }

      print_char = false;
      cout << endl;
      cout << "What would you like to do?\n"
           "q) Quit\n"
           "c) Close Window and Quit\n"
           "l) Change Lattice Type\n"
           "s) Change Sizes\n"
           "a) Change Angles\n"
           "f) Toggle fundamental domain meshes\n";
      cout << "--> " << flush;
      char mk;
      cin >> mk;

      if (mk == 'q')
      {
         break;
      }
      if (mk == 'c')
      {
         sock << "keys q";
         break;
      }
      if (mk == 'l')
      {
         lt = 0;
         cout << fact.GetLatticeOptionStr();
         cout << "enter new lattice type --> " << flush;
         cin >> lt;
         if ( lt <= 0 || lt > (int)PRIMITIVE_TRICLINIC )
         {
            cout << "invalid lattice type \"" << lt << "\"." << endl;
         }
         else
         {
            blType = (BRAVAIS_LATTICE_TYPE)lt;
            print_char = true;
         }
      }
      if (mk == 's')
      {
         if (fact.Is1D(blType))
         {
            double a_tmp = -1.0;
            cout << "Lattice Spacing: " << a << endl;
            cout << "enter new lattice spacing --> " << flush;
            cin >> a_tmp;
            if (a_tmp > 0.0)
            {
               a = a_tmp;
               print_char = true;
            }
            else
            {
               cerr << "invalid lattice spacing \"" << a_tmp << "\"." << endl;
            }
         }
         else if (fact.Is2D(blType))
         {
            double a_tmp = -1.0;
            double b_tmp = -1.0;
            cout << "Lattice Spacings: " << a << " " << b << endl;
            cout << "enter new lattice spacings --> " << flush;
            cin >> a_tmp;
            if (a_tmp > 0.0)
            {
               a = a_tmp;
               print_char = true;
            }
            else
            {
               cerr << "invalid lattice spacing \"" << a_tmp << "\"." << endl;
            }
            cin >> b_tmp;
            if (b_tmp > 0.0)
            {
               b = b_tmp;
               print_char = true;
            }
            else
            {
               cerr << "invalid lattice spacing \"" << b_tmp << "\"." << endl;
            }
         }
         else if (fact.Is3D(blType))
         {
            double a_tmp = -1.0;
            double b_tmp = -1.0;
            double c_tmp = -1.0;
            cout << "Lattice Spacings: " << a << " " << b << " " << c << endl;
            cout << "enter new lattice spacings --> " << flush;
            cin >> a_tmp;
            if (a_tmp > 0.0)
            {
               a = a_tmp;
               print_char = true;
            }
            else
            {
               cerr << "invalid lattice spacing \"" << a_tmp << "\"." << endl;
            }
            cin >> b_tmp;
            if (b_tmp > 0.0)
            {
               b = b_tmp;
               print_char = true;
            }
            else
            {
               cerr << "invalid lattice spacing \"" << b_tmp << "\"." << endl;
            }
            cin >> c_tmp;
            if (c_tmp > 0.0)
            {
               c = c_tmp;
               print_char = true;
            }
            else
            {
               cerr << "invalid lattice spacing \"" << c_tmp << "\"." << endl;
            }
         }
      }
      if (mk == 'a')
      {
         if (fact.Is2D(blType))
         {
            double gamma_tmp = NAN;
            cout << "Lattice Angle: " << 180.0 * gamma / M_PI << endl;
            cout << "enter new lattice angle, 0 < gamma < 90 (in degrees) --> "
                 << flush;
            cin >> gamma_tmp;
            gamma = M_PI * gamma_tmp / 180.0;
            print_char = true;
         }
         else if (fact.Is3D(blType))
         {
            double alpha_tmp = -1.0;
            double beta_tmp  = -1.0;
            double gamma_tmp = -1.0;
            cout << "Lattice Angles: " << 180.0 * alpha / M_PI
                 << " " << 180.0 * beta / M_PI
                 << " " << 180.0 * gamma / M_PI << endl;
            cout << "enter new lattice angles --> " << flush;
            cin >> alpha_tmp;
            cin >> beta_tmp;
            cin >> gamma_tmp;
            alpha = M_PI * alpha_tmp / 180.0;
            beta  = M_PI * beta_tmp  / 180.0;
            gamma = M_PI * gamma_tmp / 180.0;
            print_char = true;
         }
      }
      if (mk == 'f')
      {
         fdMesh = !fdMesh;
      }
      /*
      if (mk == 'b')
      {
         char bChar = 0;
         cout << "valid basis types:\n";
         cout << "h) H1 Finite Element\n";
         cout << "p) H1 Positive Finite Element\n";
         if ( elemIs2D(eType) || elemIs3D(eType) )
         {
            cout << "n) Nedelec Finite Element\n";
            cout << "r) Raviart-Thomas Finite Element\n";
         }
         cout << "l) L2 Finite Element\n";
         if ( elemIs1D(eType) || elemIs2D(eType) )
         {
            cout << "c) Crouzeix-Raviart Finite Element\n";
         }
         cout << "f) Fixed Order Continuous Finite Element\n";
         if ( elemIs2D(eType) )
         {
            cout << "g) Gauss Discontinuous Finite Element\n";
         }
         cout << "enter new basis type --> " << flush;
         cin >> bChar;
         if ( bChar == 'h' || bChar == 'p' || bChar == 'l' || bChar == 'f' ||
              ((bChar == 'n' || bChar == 'r') &&
               (elemIs2D(eType) || elemIs3D(eType))) ||
              (bChar == 'c' && (elemIs1D(eType) || elemIs2D(eType))) ||
              (bChar == 'g' && elemIs2D(eType)))
         {
            bType = bChar;
            if ( bType == 'h' )
            {
               mType = FiniteElement::VALUE;
            }
            else if ( bType == 'p' )
            {
               mType = FiniteElement::VALUE;
            }
            else if ( bType == 'n' )
            {
               mType = FiniteElement::H_CURL;
            }
            else if ( bType == 'r' )
            {
               mType = FiniteElement::H_DIV;
            }
            else if ( bType == 'l' )
            {
               if ( mType != FiniteElement::VALUE &&
                    mType != FiniteElement::INTEGRAL )
               {
                  mType = FiniteElement::VALUE;
               }
            }
            else if ( bType == 'c' )
            {
               bOrder = 1;
               mType  = FiniteElement::VALUE;
            }
            else if ( bType == 'f' )
            {
               if ( bOrder < 1 || bOrder > 3)
               {
                  bOrder = 1;
               }
               mType  = FiniteElement::VALUE;
            }
            else if ( bType == 'g' )
            {
               if ( bOrder < 1 || bOrder > 2)
               {
                  bOrder = 1;
               }
               mType  = FiniteElement::VALUE;
            }
            print_char = true;
         }
         else
         {
            cout << "invalid basis type \"" << bChar << "\"." << endl;
         }
      }
      */
      /*
      if (mk == 'm' && bType == 'l')
      {
         int mInt = 0;
         cout << "valid map types:\n"
              "0) VALUE\n"
              "1) INTEGRAL\n";
         cout << "enter new map type --> " << flush;
         cin >> mInt;
         if (mInt >=0 && mInt <= 1)
         {
            mType = mInt;
            print_char = true;
         }
         else
         {
            cout << "invalid map type \"" << mInt << "\"." << endl;
         }
      }
      */
      /*
      if (mk == 'o')
      {
         int oInt = 1;
         int oMin = ( bType == 'h' || bType == 'p' || bType == 'n' ||
                      bType == 'f' || bType == 'g')?1:0;
         int oMax = -1;
         switch (bType)
         {
            case 'g':
               oMax = 2;
               break;
            case 'f':
               oMax = 3;
               break;
            default:
               oMax = -1;
         }
         cout << "basis function order must be >= " << oMin;
         if ( oMax >= 0 )
         {
            cout << " and <= " << oMax;
         }
         cout << endl;
         cout << "enter new basis function order --> " << flush;
         cin >> oInt;
         if ( oInt >= oMin && oInt <= (oMax>=0)?oMax:oInt )
         {
            bOrder = oInt;
            print_char = true;
         }
         else
         {
            cout << "invalid basis order \"" << oInt << "\"." << endl;
         }
      }
      */
      /*
      if (mk == 't')
      {
         cout << "transformation options:\n";
         cout << "r) reset to reference element\n";
         cout << "u) uniform scaling\n";
         if ( elemIs2D(eType) || elemIs3D(eType) )
         {
            cout << "c) compression\n";
            cout << "s) shear\n";
         }
         cout << "enter transformation type --> " << flush;
         char tk;
         cin >> tk;
         if (tk == 'r')
         {
            dType = Deformation::INVALID;
         }
         else if (tk == 'u')
         {
            cout << "enter scaling constant --> " << flush;
            cin >> defData.uniformScale;
            if ( defData.uniformScale > 0.0 )
            {
               dType = Deformation::UNIFORM;
            }
         }
         else if (tk == 'c' && !elemIs1D(eType))
         {
            int dim = elemIs2D(eType)?2:3;
            cout << "enter compression factor --> " << flush;
            cin >> defData.squeezeFactor;
            cout << "enter compression axis (0-" << dim-1 << ") --> " << flush;
            cin >> defData.squeezeAxis;

            if ( defData.squeezeFactor > 0.0 &&
                 (defData.squeezeAxis >= 0 && defData.squeezeAxis < dim))
            {
               dType = Deformation::SQUEEZE;
            }
         }
         else if (tk == 's' && !elemIs1D(eType))
         {
            int dim = elemIs2D(eType)?2:3;
            cout << "enter shear vector (components separated by spaces) --> "
                 << flush;
            defData.shearVec.SetSize(dim);
            for (int i=0; i<dim; i++)
            {
               cin >> defData.shearVec[i];
            }
            cout << "enter shear axis (0-" << dim-1 << ") --> " << flush;
            cin >> defData.shearAxis;

            if ( defData.shearAxis >= 0 && defData.shearAxis < dim )
            {
               dType = Deformation::SHEAR;
            }
         }
      }
      */
   }

   // Cleanup
   delete mesh;
   delete lat;

   // Exit
   return 0;
}

string elemTypeStr(const Element::Type & eType)
{
   switch (eType)
   {
      case Element::POINT:
         return "POINT";
         break;
      case Element::SEGMENT:
         return "SEGMENT";
         break;
      case Element::TRIANGLE:
         return "TRIANGLE";
         break;
      case Element::QUADRILATERAL:
         return "QUADRILATERAL";
         break;
      case Element::TETRAHEDRON:
         return "TETRAHEDRON";
         break;
      case Element::HEXAHEDRON:
         return "HEXAHEDRON";
         break;
      default:
         return "INVALID";
         break;
   };
}

string
basisTypeStr(char bType)
{
   switch (bType)
   {
      case 'h':
         return "Continuous (H1)";
         break;
      case 'p':
         return "Continuous Positive (H1)";
         break;
      case 'n':
         return "Nedelec";
         break;
      case 'r':
         return "Raviart-Thomas";
         break;
      case 'l':
         return "Discontinuous (L2)";
         break;
      case 'f':
         return "Fixed Order Continuous";
         break;
      case 'g':
         return "Gaussian Discontinuous";
         break;
      case 'c':
         return "Crouzeix-Raviart";
         break;
      default:
         return "INVALID";
         break;
   };
}

string
mapTypeStr(int mType)
{
   switch (mType)
   {
      case FiniteElement::VALUE:
         return "VALUE";
         break;
      case FiniteElement::H_CURL:
         return "H_CURL";
         break;
      case FiniteElement::H_DIV:
         return "H_DIV";
         break;
      case FiniteElement::INTEGRAL:
         return "INTEGRAL";
         break;
      default:
         return "INVALID";
         break;
   }
}
/*
int
update_basis(vector<socketstream*> & sock,  const VisWinLayout & vwl,
             Element::Type e, char bType, int bOrder, int mType,
             Deformation::DefType dType, const DeformationData & defData,
             bool visualization)
{
   bool vec = false;

   Mesh *mesh;
   ElementMeshStream imesh(e);
   if (!imesh)
   {
      {
         cerr << "\nProblem with meshstream object\n" << endl;
      }
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   int dim = mesh->Dimension();

   if ( dType != Deformation::INVALID )
   {
      Deformation defCoef(dim, dType, defData);
      mesh->Transform(defCoef);
   }

   FiniteElementCollection * FEC = NULL;
   switch (bType)
   {
      case 'h':
         FEC = new H1_FECollection(bOrder, dim);
         vec = false;
         break;
      case 'p':
         FEC = new H1Pos_FECollection(bOrder, dim);
         vec = false;
         break;
      case 'n':
         FEC = new ND_FECollection(bOrder, dim);
         vec = true;
         break;
      case 'r':
         FEC = new RT_FECollection(bOrder-1, dim);
         vec = true;
         break;
      case 'l':
         FEC = new L2_FECollection(bOrder, dim, BasisType::GaussLegendre,
                                   mType);
         vec = false;
         break;
      case 'c':
         FEC = new CrouzeixRaviartFECollection();
         break;
      case 'f':
         if ( bOrder == 1 )
         {
            FEC = new LinearFECollection();
         }
         else if ( bOrder == 2 )
         {
            FEC = new QuadraticFECollection();
         }
         else if ( bOrder == 3 )
         {
            FEC = new CubicFECollection();
         }
         break;
      case 'g':
         if ( bOrder == 1 )
         {
            FEC = new GaussLinearDiscont2DFECollection();
         }
         else if ( bOrder == 2 )
         {
            FEC = new GaussQuadraticDiscont2DFECollection();
         }
         break;
   }
   if ( FEC == NULL)
   {
      delete mesh;
      return 1;
   }

   FiniteElementSpace FESpace(mesh, FEC);

   int ndof = FESpace.GetVSize();

   Array<int> vdofs;
   FESpace.GetElementVDofs(0,vdofs);

   char vishost[] = "localhost";
   int  visport   = 19916;

   int offx = vwl.w+10, offy = vwl.h+45; // window offsets

   for (unsigned int i=0; i<sock.size(); i++)
   {
      *sock[i] << "keys q";
      delete sock[i];
   }

   sock.resize(ndof);
   for (int i=0; i<ndof; i++)
   {
      sock[i] = new socketstream; sock[i]->precision(8);
   }

   GridFunction ** x = new GridFunction*[ndof];
   for (int i=0; i<ndof; i++)
   {
      x[i]  = new GridFunction(&FESpace);
      *x[i] = 0.0;
      if ( vdofs[i] < 0 )
      {
         (*x[i])(-1-vdofs[i]) = -1.0;
      }
      else
      {
         (*x[i])(vdofs[i]) = 1.0;
      }
   }

   int ref = 0;
   int exOrder = 0;
   if ( bType == 'n' ) { exOrder++; }
   if ( bType == 'r' ) { exOrder += 2; }
   while ( 1<<ref < bOrder + exOrder || ref == 0 )
   {
      mesh->UniformRefinement();
      FESpace.Update();

      for (int i=0; i<ndof; i++)
      {
         x[i]->Update();
      }
      ref++;
   }

   for (int i=0; i<ndof; i++)
   {
      ostringstream oss;
      oss << "DoF " << i + 1;
      if (visualization)
      {
         VisualizeField(*sock[i], vishost, visport, *x[i], oss.str().c_str(),
                        (i % vwl.nx) * offx, ((i / vwl.nx) % vwl.ny) * offy,
                        vwl.w, vwl.h,
                        vec);
      }
   }

   for (int i=0; i<ndof; i++)
   {
      delete x[i];
   }
   delete [] x;

   delete FEC;
   delete mesh;

   return 0;
}
*/
/*
static string lattice_types =
"1D Bravais Lattices (1 types)\n
\t 1 - Primitive Segment,\n
2D Bravais Lattices (5 types)\n
\t 2 - Primitive Square,\n
\t 3 - Primitive Hexagonal,\n
\t 4 - Primitive Rectangular,\n
\t 5 - Centered Rectangular,\n
\t 6 - Primitive Oblique,\n
3D Bravais Lattices (14 types)\n
\t 7 - Primitive Cubic,\n
\t 8 - Face-Centered Cubic,\n
\t 9 - Body-Centered Cubic,\n
\t10 - Primitive Tetragonal,\n
\t11 - Body-Centered Tetragonal,\n
\t12 - Primitive Orthorhombic,\n
\t13 - Face-Centered Orthorhombic,\n
\t14 - Body-Centered Orthorhombic,\n
\t15 - Base-Centered Orthorhombic,\n
\t16 - Primitive Hexagonal Prism,\n
\t17 - Primitive Rhombohedral,\n
\t18 - Primitive Monoclinic,\n
\t19 - Base-Centered Monoclinic,\n
\t20 - Primitive Triclinic\n";
*/
