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

int       toint(int d, double v);
double todouble(int d, int v);

int main (int argc, char *argv[])
{
   const char *mesh_file = "../../data/beam-hex.mesh";
   double a = 1.0;
   double c = 1.0;
   int lattice_type = 1;
   int sr = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to visualize.");
   args.AddOption(&lattice_type, "-bl", "--bravais-lattice",
                  "Bravais Lattice Type: "
                  " 1 - Primitive Cubic,"
                  " 2 - Body-Centered Cubic,"
                  " 3 - Face-Centered Cubic,"
                  " 4 - Hexagonal Prism");
   args.AddOption(&sr, "-sr", "--serial-refinement",
                  "Number of serial refinement levels.");
   args.AddOption(&a, "-a", "--a",
                  "");
   args.AddOption(&c, "-c", "--c",
                  "");
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

   Mesh *mesh;
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      cerr << "can not open mesh file: " << mesh_file << endl;
      return 2;
   }
   mesh = new Mesh(imesh, 1, 1);
   imesh.close();

   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // int dim  = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   cout << "Euler Number of Initial Mesh:  " << mesh->EulerNumber() << endl;

   // int na = -1;
   // vector<Vector> lat_vecs;
   vector<Vector> trans_vecs;

   BravaisLattice * bravais = NULL;

   switch (lattice_type)
   {
      case 1:
      {
         bravais = new CubicLattice(a);
         /*
         bravais->GetLatticeVectors(lat_vecs);
              na = 3;
              trans_vecs.resize(na);
              for (int i=0; i<na; i++)
              {
                 trans_vecs[i].SetSize(sdim);
              }
              trans_vecs[0][0] =  1.0; trans_vecs[0][1] =  0.0; trans_vecs[0][2] =  0.0;
              trans_vecs[1][0] =  0.0; trans_vecs[1][1] =  1.0; trans_vecs[1][2] =  0.0;
              trans_vecs[2][0] =  0.0; trans_vecs[2][1] =  0.0; trans_vecs[2][2] =  1.0;
         */
      }
      break;
      case 2:
      {
         bravais = new BodyCenteredCubicLattice(a);
         /*
         bravais->GetLatticeVectors(lat_vecs);
              na = 7;
              trans_vecs.resize(na);
              for (int i=0; i<na; i++)
              {
                 trans_vecs[i].SetSize(sdim);
              }
              trans_vecs[0][0] =  1.0; trans_vecs[0][1] =  0.0; trans_vecs[0][2] =  0.0;
              trans_vecs[1][0] =  0.0; trans_vecs[1][1] =  1.0; trans_vecs[1][2] =  0.0;
              trans_vecs[2][0] =  0.0; trans_vecs[2][1] =  0.0; trans_vecs[2][2] =  1.0;
              trans_vecs[3][0] =  0.5; trans_vecs[3][1] =  0.5; trans_vecs[3][2] =  0.5;
              trans_vecs[4][0] = -0.5; trans_vecs[4][1] =  0.5; trans_vecs[4][2] =  0.5;
              trans_vecs[5][0] = -0.5; trans_vecs[5][1] = -0.5; trans_vecs[5][2] =  0.5;
              trans_vecs[6][0] =  0.5; trans_vecs[6][1] = -0.5; trans_vecs[6][2] =  0.5;
         */
      }
      break;
      case 3:
      {
         bravais = new FaceCenteredCubicLattice(a);
         /*
         bravais->GetLatticeVectors(lat_vecs);
              na = 6;
              trans_vecs.resize(na);
              for (int i=0; i<na; i++)
              {
                 trans_vecs[i].SetSize(sdim);
              }
              trans_vecs[0][0] =  0.5; trans_vecs[0][1] =  0.5; trans_vecs[0][2] =  0.0;
              trans_vecs[1][0] =  0.5; trans_vecs[1][1] = -0.5; trans_vecs[1][2] =  0.0;
              trans_vecs[2][0] =  0.5; trans_vecs[2][1] =  0.0; trans_vecs[2][2] =  0.5;
              trans_vecs[3][0] =  0.0; trans_vecs[3][1] =  0.5; trans_vecs[3][2] =  0.5;
              trans_vecs[4][0] = -0.5; trans_vecs[4][1] =  0.0; trans_vecs[4][2] =  0.5;
              trans_vecs[5][0] =  0.0; trans_vecs[5][1] = -0.5; trans_vecs[5][2] =  0.5;
         */
      }
      break;
      case 4:
      {
         bravais = new HexagonalPrismLattice(a, c);
         /*
         bravais->GetLatticeVectors(lat_vecs);
              na = 4;
              trans_vecs.resize(na);
              for (int i=0; i<na; i++)
              {
                 trans_vecs[i].SetSize(sdim);
              }
              trans_vecs[0][0] =  1.0; trans_vecs[0][1] =  0.0; trans_vecs[0][2] =  0.0;
              trans_vecs[1][0] =  0.5; trans_vecs[1][1] =  0.8660254037844386; trans_vecs[1][2] =  0.0;
              trans_vecs[2][0] =  0.5; trans_vecs[2][1] = -0.8660254037844386; trans_vecs[2][2] =  0.0;
              trans_vecs[3][0] =  0.0; trans_vecs[3][1] =  0.0; trans_vecs[3][2] =  1.0;
         */
      }
      break;
   };
   bravais->GetTranslationVectors(trans_vecs);
   // na = (int)trans_vecs.size();

   // ofstream ofs1("mesh-r1.mesh");
   // mesh->Print(ofs1);
   // ofs1.close();

   map<int,map<int,map<int,int> > > c2v;
   set<int> v;

   int d = 5;
   Vector xMax(3), xMin(3);
   xMax = xMin = 0.0;

   for (int be=0; be<mesh->GetNBE(); be++)
   {
      Array<int> dofs;
      mesh->GetBdrElementVertices(be,dofs);
      // cout << be << ":  ";
      // dofs.Print(cout,4);
      for (int i=0; i<dofs.Size(); i++)
      {
         double * coords = mesh->GetVertex(dofs[i]);
         for (int j=0; j<sdim; j++)
         {
            // cout << " " << coords[j];
            xMax[j] = max(xMax[j],coords[j]);
            xMin[j] = min(xMin[j],coords[j]);
         }
         c2v[toint(d,coords[0])][toint(d,coords[1])][toint(d,coords[2])] =
            dofs[i];
         v.insert(dofs[i]);
         // cout << endl;
      }
   }
   cout << "Number of Boundary Vertices:  " << v.size() << endl;

   cout << "xMin: ";
   xMin.Print(cout,3);
   cout << "xMax: ";
   xMax.Print(cout,3);


   /*
   for (int be=0; be<mesh->GetNBE(); be++)
   {
     Array<int> dofs;
     mesh->GetBdrElementVertices(be,dofs);
     for (int i=0; i<dofs.Size(); i++)
     {
       double * coords = mesh->GetVertex(dofs[i]);
       for (int j=0; j<sdim; j++)
       {
       }
     }
   }
   */
   // double tol = 1.0e-2;

   set<int> xc;
   set<int> yc;
   set<int> zc;

   map<int,map<int,map<int,int> > >::iterator mxyz;
   map<int,map<int,int> >::iterator myz;
   map<int,int>::iterator mz;

   {
      int c = 0;
      for (mxyz=c2v.begin(); mxyz!=c2v.end(); mxyz++)
      {
         for (myz=mxyz->second.begin(); myz!=mxyz->second.end(); myz++)
         {
            for (mz=myz->second.begin(); mz!=myz->second.end(); mz++)
            {
               cout << mz->second << ":  "
                    << mxyz->first << " " << myz->first << " " << mz->first
                    << endl;
               c++;
            }
         }
      }
   }

   // vector<map<int,int> > v2v(6);
   map<int,int>        slaves;
   map<int,set<int> > masters;

   set<int>::iterator sit;
   for (sit=v.begin(); sit!=v.end(); sit++) { masters[*sit]; }

   Vector at(3);

   for (unsigned int i=0; i<trans_vecs.size(); i++)
   {
      int c = 0;
      cout << "trans_vecs = ";
      trans_vecs[i].Print(cout,3);
      for (mxyz=c2v.begin(); mxyz!=c2v.end(); mxyz++)
      {
         // double x = todouble(d,mxyz->first) + trans_vecs[i][0];
         // int xi = mxyz->first + toint(d,trans_vecs[i][0]);
         // cout << "xi " << xi << endl;
         /*if ( x > xMax[0] + tol * (xMax[0] - xMin[0]) ||
         x < xMin[0] - tol * (xMax[0] - xMin[0]) )
         {
         continue;
         }
         else*/// if ( c2v.find(toint(d,x)) == c2v.end() )
         /*if ( c2v.find(xi) == c2v.end() )
               {
                  continue;
               }*/

         for (myz=mxyz->second.begin(); myz!=mxyz->second.end(); myz++)
         {
            // double y = todouble(d,myz->first) + trans_vecs[i][1];
            // int yi = myz->first + toint(d,trans_vecs[i][1]);
            // cout << "yi " << yi << endl;
            /*if ( y > xMax[1] + tol * (xMax[1] - xMin[1]) ||
                 y < xMin[1] - tol * (xMax[1] - xMin[1]) )
            {
              continue;
            }
            else*/// if ( mxyz->second.find(toint(d,y)) == mxyz->second.end() )
            /*if ( c2v[xi].find(yi) == c2v[xi].end() )
                  {
                     continue;
                }*/
            for (mz=myz->second.begin(); mz!=myz->second.end(); mz++)
            {
               // double z = todouble(d,mz->first) + trans_vecs[i][2];
               // int zi = mz->first + toint(d,trans_vecs[i][2]);
               // cout << "zi " << zi << endl;
               /*if ( z > xMax[2] + tol * (xMax[2] - xMin[2]) ||
               z < xMin[2] - tol * (xMax[2] - xMin[2]) )
               {
                 continue;
               }
               else*/// if ( myz->second.find(toint(d,z)) == myz->second.end() )
               /*if ( c2v[xi][yi].find(zi) == c2v[xi][yi].end() )
                     {
                        continue;
                     }
               */
               double * coords = mesh->GetVertex(mz->second);

               at = trans_vecs[i];
               at[0] += coords[0];
               at[1] += coords[1];
               at[2] += coords[2];

               int xi = toint(d,at[0]);
               int yi = toint(d,at[1]);
               int zi = toint(d,at[2]);

               if (c2v.find(xi) == c2v.end()) { continue; }
               if (c2v[xi].find(yi) == c2v[xi].end()) { continue; }
               if (c2v[xi][yi].find(zi) == c2v[xi][yi].end()) { continue; }

               // cout << mz->second << ":  "
               //    << mxyz->first << " " << myz->first << " " << mz->first
               //    << " -> " << c2v[xi][yi][zi];
               //v2v[i][c2v[xi][yi][zi]] = mz->second;
               int master = mz->second;
               int slave  = c2v[xi][yi][zi];

               bool mInM = masters.find(master) != masters.end();
               bool sInM = masters.find(slave)  != masters.end();

               // cout << "\t" << mInM << "\t" << sInM << endl;

               if ( mInM && sInM )
               {
                  //cout << "\t" << master << " is master of " << masters[master].size() << " vertices" << endl;
                  //cout << "\t" << slave << " is master of " << masters[slave].size() << " vertices" << endl;
                  // Both vertices are currently masters
                  //   Demote "slave" to be a slave of master
                  masters[master].insert(slave);
                  slaves[slave] = master;
                  for (sit=masters[slave].begin();
                       sit!=masters[slave].end(); sit++)
                  {
                     masters[master].insert(*sit);
                     slaves[*sit] = master;
                  }
                  masters.erase(slave);
                  //cout << "\t" << master << " is now master of " << masters[master].size() << " vertices" << endl;
               }
               else if ( mInM && !sInM )
               {
                  //cout << "\t" << master << " is master of " << masters[master].size() << " vertices" << endl;
                  //cout << "\t" << slave << " is a slave of " << slaves[slave] << " along with " << masters[slaves[slave]].size()-1 << " other vertices" << endl;
                  // "master" is already a master and "slave" is already a slave
                  // Make "master" and its slaves slaves of "slave"'s master
                  if ( master != slaves[slave] )
                  {
                     masters[slaves[slave]].insert(master);
                     slaves[master] = slaves[slave];
                     for (sit=masters[master].begin();
                          sit!=masters[master].end(); sit++)
                     {
                        masters[slaves[slave]].insert(*sit);
                        slaves[*sit] = slaves[slave];
                     }
                     masters.erase(master);
                  }
               }
               else if ( !mInM && sInM )
               {
                  // cout << "\t" << master << " is a slave of " << slaves[master] << " along with " << masters[slaves[master]].size()-1 << " other vertices" << endl;
                  // cout << "\t" << slave << " is master of " << masters[slave].size() << " vertices" << endl;
                  // "master" is currently a slave and
                  // "slave" is currently a master
                  // Make "slave" and its slaves slaves of "master"'s master
                  if ( slave != slaves[master] )
                  {
                     masters[slaves[master]].insert(slave);
                     slaves[slave] = slaves[master];
                     for (sit=masters[slave].begin();
                          sit!=masters[slave].end(); sit++)
                     {
                        masters[slaves[master]].insert(*sit);
                        slaves[*sit] = slaves[master];
                     }
                     masters.erase(slave);
                  }
               }
               else
               {
                  // cout << "\t" << master << " is a slave of " << slaves[master] << " along with " << masters[slaves[master]].size()-1 << " other vertices" << endl;
                  // cout << "\t" << slave << " is a slave of " << slaves[slave] << " along with " << masters[slaves[slave]].size()-1 << " other vertices" << endl;
                  // Both vertices are currently slaves
                  // Make "slave" and its fellow slaves slaves
                  // of "master"'s master
                  if ( slaves[master] != slaves[slave] )
                  {
                     masters[slaves[master]].insert(slaves[slave]);
                     slaves[slaves[slave]] = slaves[master];
                  }
                  for (sit=masters[slaves[slave]].begin();
                       sit!=masters[slaves[slave]].end(); sit++)
                  {
                     masters[slaves[master]].insert(*sit);
                     slaves[*sit] = slaves[master];
                  }
                  if ( slaves[master] != slaves[slave] )
                  {
                     masters.erase(slaves[slave]);
                  }
               }
               c++;
            }
         }
      }
      cout << "Found " << c << " possible node";
      if ( c != 1 ) { cout << "s"; }
      cout <<" to project." << endl;
   }
   cout << "Number of Master Vertices:  " << masters.size() << endl;
   cout << "Number of Slave Vertices:   " << slaves.size() << endl;

   Array<int> v2v(mesh->GetNV());

   for (int i=0; i<v2v.Size(); i++)
   {
      v2v[i] = i;
   }

   map<int,int>::iterator mit;
   for (mit=slaves.begin(); mit!=slaves.end(); mit++)
   {
      v2v[mit->first] = mit->second;
   }

   map<int,set<int> >::iterator msit;
   for (msit=masters.begin(); msit!=masters.end(); msit++)
   {
      /*
       double * coords = mesh->GetVertex(msit->first);
       cout << msit->first << ":  " << coords[0] << " " << coords[1] << " " <<
            coords[2] << endl;
      */
      // cout << msit->first << ":";
      // for (sit=msit->second.begin(); sit!=msit->second.end(); sit++)
      // cout << " " << *sit;
      // cout << endl;
   }

   Mesh *per_mesh = new Mesh(*mesh, true);

   per_mesh->SetCurvature(1,true);

   // renumber elements
   for (int i = 0; i < per_mesh->GetNE(); i++)
   {
      Element *el = per_mesh->GetElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   // renumber boundary elements
   for (int i = 0; i < per_mesh->GetNBE(); i++)
   {
      Element *el = per_mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }

   per_mesh->RemoveUnusedVertices();
   // per_mesh->RemoveInternalBoundaries();

   cout << "Euler Number of Final Mesh:    " << per_mesh->EulerNumber() << endl;

   ofstream ofs("periodic-mesh.mesh");
   per_mesh->Print(ofs);
   ofs.close();

   delete per_mesh;
   delete mesh;
}

int toint(int d, double v)
{
   return (int)copysign(round(fabs(v)*pow(10.0,d)),v);
}

double todouble(int d, int v)
{
   return pow(10.0,-d)*v;
}
