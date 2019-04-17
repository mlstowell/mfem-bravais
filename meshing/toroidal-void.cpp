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

#include "mfem.hpp"
#include "../lib/bravais.hpp"
#include <fstream>
#include <limits>
#include <cstdlib>

using namespace std;
using namespace mfem;
using namespace mfem::bravais;
using namespace mfem::miniapps;

int       toint(int d, double v);
double todouble(int d, int v);

int main (int argc, char *argv[])
{
   int lt = (int)BODY_CENTERED_CUBIC;
   double a = 1.0;
   double b = 1.0;
   double c = 1.0;
   double alpha = NAN;
   double beta  = NAN;
   double gamma = NAN;
   int sr = 0;

   double r0 = 0.25;
   double r1 = 1.0;
   double r2 = 10.0;

   bool fdMesh = false;
   
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
   args.AddOption(&sr, "-sr", "--serial-refinement",
                  "Number of serial refinement levels.");
   args.AddOption(&r0, "-r0", "--minor-radius",
                  "Minor radius of torus.");
   args.AddOption(&r1, "-r1", "--major-radius",
                  "Major radius of torus.");
   args.AddOption(&r2, "-r2", "--sphere-radius",
                  "Radius of sphere.");
   args.AddOption(&fdMesh, "-fd", "--fund-domain", "-no-fd",
                  "--no-fund-domain",
                  "Enable or disable meshes derived from "
		  "the fundamental domain.");
   args.Parse();
   if (!args.Good())
   {
      if (!args.Help())
      {
         args.PrintError(cout);
         cout << endl;
      }
      cout << "Create a periodic mesh from a serial mesh:\n"
           << "   periodic-mesh -m <mesh_file>\n"
           << "All Options:\n";
      args.PrintHelp(cout);
      return 1;
   }
   args.PrintOptions(cout);

   BRAVAIS_LATTICE_TYPE blType = (BRAVAIS_LATTICE_TYPE)lt;

   BravaisLatticeFactory fact;

   BravaisLattice * lat = fact.GetLattice(blType, a, b, c, alpha, beta, gamma);

   Mesh * mesh_cell = lat->GetWignerSeitzMesh(fdMesh);

   int nvert = mesh_cell->GetNV();
   int nelem = mesh_cell->GetNE();
   cout << "Unit cell consists of " << nvert << " vertices and "
	<< nelem << " elements." << endl;
   
   Vector cell_vert;
   mesh_cell->GetVertices(cell_vert);
   
   vector<Vector> aVec;
   lat->GetLatticeVectors(aVec);

   double nrm = aVec[0].Norml2();
   int nr = (int)ceil(sqrt(3.0) * r2 / nrm);

   Array<int> vert;
   Vector vc(3); 
   Vector vr(3);

   // Dry run to count cells
   int icell = 0;
   for (int i=-nr; i<=nr; i++)
   {
      for (int j=-nr; j<=nr; j++)
      {
         for (int k=-nr; k<=nr; k++)
         {
	   if ((i==0 && j==0 && k==1) ||
	       (i==0 && j==0 && k==-1) ||
	       (i==1 && j==1 && k==0) ||
	       (i==-1 && j==-1 && k==0) ||
	       (i==1 && j==1 && k==1) ||
	       (i==-1 && j==-1 && k==-1)
	       )
	   {
	     continue;
	   }
	   vc = 0.0;
	   vc.Add(i, aVec[0]);
	   vc.Add(j, aVec[1]);
	   vc.Add(k, aVec[2]);
	   if (vc.Norml2() <= r2)
	   {
	     icell++;
	   }
	 }
      }
   }
   cout << "Found " << icell << " unit cells inside our sphere" << endl;

   int ncells = icell;
   
   cout << "Initial mesh should contain " << nvert * ncells
	<< " vertices and " << nelem * ncells << " elements." << endl; 

   Mesh mesh(3, nvert * ncells, nelem * ncells);

   icell = 0;
   for (int i=-nr; i<=nr; i++)
   {
      for (int j=-nr; j<=nr; j++)
      {
         for (int k=-nr; k<=nr; k++)
         {
	   if ((i==0 && j==0 && k==1) ||
	       (i==0 && j==0 && k==-1) ||
	       (i==1 && j==1 && k==0) ||
	       (i==-1 && j==-1 && k==0) ||
	       (i==1 && j==1 && k==1) ||
	       (i==-1 && j==-1 && k==-1))
	     {
	       continue;
	     }
	   vc = 0.0;
	   vc.Add(i, aVec[0]);
	   vc.Add(j, aVec[1]);
	   vc.Add(k, aVec[2]);
	   if (vc.Norml2() <= r2)
	   {
	     for (int p=0; p<nvert; p++)
	     {
	       vr = vc;
	       vr[0] += cell_vert[0 * nvert + p];
	       vr[1] += cell_vert[1 * nvert + p];
	       vr[2] += cell_vert[2 * nvert + p];
	       mesh.AddVertex(vr);
	     }
	     for (int q=0; q<nelem; q++)
	     {
	       mesh_cell->GetElementVertices(q, vert);
	       for (int s=0; s<vert.Size(); s++)
		 {
		   vert[s] += icell * nvert;
		 }
	       if (vert.Size() == 4)
	       {
		 mesh.AddTet(vert);
	       }
	       else if (vert.Size() == 5)
	       {
		 mesh.AddPyramid(vert);
	       }
	       else if (vert.Size() == 6)
	       {
		 mesh.AddWedge(vert);
	       }
	       else if (vert.Size() == 8)
	       {
		 mesh.AddHex(vert);
	       }
	     }
	     icell++;
	   }
	 }
      }
   }

   MergeMeshNodes(&mesh);

   cout << "After merging nodes mesh contains " << mesh.GetNV() << " vertices and "
	<< mesh.GetNE() << " elements." << endl;
   
   mesh.GenerateBoundaryElements();
   mesh.FinalizeMesh(true);

   cout << "Finalized mesh contains " << mesh.GetNV() << " vertices and "
	<< mesh.GetNE() << " elements." << endl;
   
   // mesh.RemoveUnusedVertices();
   
   // cout << "After removing unused nodes mesh contains " << mesh.GetNV() << " vertices and "
   //	<< mesh.GetNE() << " elements." << endl;
   
   cout << "Euler Number of Final Mesh:    " << mesh.EulerNumber() << endl;

   for (int i=0; i<mesh.GetNBE(); i++)
   {
     Element * be = mesh.GetBdrElement(i);
     be->GetVertices(vert);
     int nv = vert.Size();

     vc = 0.0;
     
     for (int j=0; j<vert.Size(); j++)
       {
	 Vector vv(mesh.GetVertex(vert[j]), 3);
	 vc.Add(1.0 / nv, vv);
       }
     if (vc.Norml2() < 0.5 * r2)
       {
	 be->SetAttribute(2);
       }
   }
      
   ofstream ofs("toroidal-void.mesh");
   mesh.Print(ofs);
   ofs.close();

   
   delete lat;
   delete mesh_cell;
}

int toint(int d, double v)
{
   return (int)copysign(round(fabs(v)*pow(10.0,d)),v);
}

double todouble(int d, int v)
{
   return pow(10.0,-d)*v;
}
