#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cassert>

using namespace std;
using namespace mfem;

void trans(Mesh * mesh, double a, double b, double c, double alpha);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "unit-truncated-octahedron-new.mesh";
   double a = 0.675, b = 0.875, c = 1.0, alpha = 0.33*M_PI;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&a, "-a", "--a",
                  "");
   args.AddOption(&b, "-b", "--b",
                  "");
   args.AddOption(&c, "-c", "--c",
                  "");
   args.AddOption(&alpha, "-alpha", "--alpha",
                  "");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if ( c > sqrt(2.0) * a )
   {
      mesh_file = "unit-elongated-dodecahedron.mesh";
   }

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   assert( dim == 3 );

   trans(mesh, a, b, c, alpha);

   ofstream mesh_ofs("transformed.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }

   delete mesh;
}

void trans_trunc_dod(Mesh * mesh, double a, double b, double c, double alpha);
void trans_elong_dod(Mesh * mesh, double a, double b, double c, double alpha);

void trans(Mesh * mesh, double a, double b, double c, double alpha)
{
   if ( c < sqrt(2.0) * a )
   {
      trans_trunc_dod(mesh, a, b, c, alpha);
   }
   else
   {
      trans_elong_dod(mesh, a, b, c, alpha);
   }
}

void trans_trunc_dod(Mesh * mesh, double a, double b, double c, double alpha)
{
   // assert( c < sqrt(2.0) * a );
   assert(mesh->GetNV() == 38);

   double * v = NULL;
   double ca = cos(alpha);
   double sa = sin(alpha);

   // X = -a/2
   v = mesh->GetVertex(0);
   v[0] = -0.5*a;            v[1] =  0.25*c*c/a;       v[2] =  0.0;
   v = mesh->GetVertex(1);
   v[0] = -0.5*a;            v[1] =  0.0;              v[2] =  0.25*c;
   v = mesh->GetVertex(2);
   v[0] = -0.5*a;            v[1] = -0.25*c*c/a;       v[2] =  0.0;
   v = mesh->GetVertex(3);
   v[0] = -0.5*a;            v[1] =  0.0;              v[2] = -0.25*c;

   // X = +a/2
   v = mesh->GetVertex(4);
   v[0] =  0.5*a;            v[1] =  0.25*c*c/a;       v[2] =  0.0;
   v = mesh->GetVertex(5);
   v[0] =  0.5*a;            v[1] =  0.0;              v[2] =  0.25*c;
   v = mesh->GetVertex(6);
   v[0] =  0.5*a;            v[1] = -0.25*c*c/a;       v[2] =  0.0;
   v = mesh->GetVertex(7);
   v[0] =  0.5*a;            v[1] =  0.0;              v[2] = -0.25*c;

   // Y = -a/2
   v = mesh->GetVertex(8);
   v[0] =  0.25*c*c/a;       v[1] = -0.5*a;            v[2] =  0.0;
   v = mesh->GetVertex(9);
   v[0] =  0.0;              v[1] = -0.5*a;            v[2] =  0.25*c;
   v = mesh->GetVertex(10);
   v[0] = -0.25*c*c/a;       v[1] = -0.5*a;            v[2] =  0.0;
   v = mesh->GetVertex(11);
   v[0] =  0.0;              v[1] = -0.5*a;            v[2] = -0.25*c;

   // Y = +a/2
   v = mesh->GetVertex(12);
   v[0] =  0.25*c*c/a;       v[1] =  0.5*a;            v[2] =  0.0;
   v = mesh->GetVertex(13);
   v[0] =  0.0;              v[1] =  0.5*a;            v[2] =  0.25*c;
   v = mesh->GetVertex(14);
   v[0] = -0.25*c*c/a;       v[1] =  0.5*a;            v[2] =  0.0;
   v = mesh->GetVertex(15);
   v[0] =  0.0;              v[1] =  0.5*a;            v[2] = -0.25*c;

   // Base
   v = mesh->GetVertex(16);
   v[0] =  0.5*a-0.25*c*c/a; v[1] =  0.0;              v[2] = -0.5*c;
   v = mesh->GetVertex(17);
   v[0] =  0.0;              v[1] =  0.5*a-0.25*c*c/a; v[2] = -0.5*c;
   v = mesh->GetVertex(18);
   v[0] = -0.5*a+0.25*c*c/a; v[1] =  0.0;              v[2] = -0.5*c;
   v = mesh->GetVertex(19);
   v[0] =  0.0;              v[1] = -0.5*a+0.25*c*c/a; v[2] = -0.5*c;

   // Top
   v = mesh->GetVertex(20);
   v[0] =  0.5*a-0.25*c*c/a; v[1] =  0.0;              v[2] =  0.5*c;
   v = mesh->GetVertex(21);
   v[0] =  0.0;              v[1] =  0.5*a-0.25*c*c/a; v[2] =  0.5*c;
   v = mesh->GetVertex(22);
   v[0] = -0.5*a+0.25*c*c/a; v[1] =  0.0;              v[2] =  0.5*c;
   v = mesh->GetVertex(23);
   v[0] =  0.0;              v[1] = -0.5*a+0.25*c*c/a; v[2] =  0.5*c;

   // Hexagon Centers in -X half space
   v = mesh->GetVertex(24);
   v[0] = -0.5*a+0.25*c*c/a; v[1] = -0.25*c*c/a;       v[2] = -0.25*c;
   v = mesh->GetVertex(25);
   v[0] = -0.25*c*c/a;       v[1] = -0.5*a+0.25*c*c/a; v[2] =  0.25*c;
   v = mesh->GetVertex(26);
   v[0] = -0.5*a+0.25*c*c/a; v[1] =  0.25*c*c/a;       v[2] = -0.25*c;
   v = mesh->GetVertex(27);
   v[0] = -0.25*c*c/a;       v[1] =  0.5*a-0.25*c*c/a; v[2] =  0.25*c;

   // Hexagon Centers in +X half space
   v = mesh->GetVertex(28);
   v[0] =  0.5*a-0.25*c*c/a; v[1] = -0.25*c*c/a;       v[2] = -0.25*c;
   v = mesh->GetVertex(29);
   v[0] =  0.25*c*c/a;       v[1] = -0.5*a+0.25*c*c/a; v[2] =  0.25*c;
   v = mesh->GetVertex(30);
   v[0] =  0.5*a-0.25*c*c/a; v[1] =  0.25*c*c/a;       v[2] = -0.25*c;
   v = mesh->GetVertex(31);
   v[0] =  0.25*c*c/a;       v[1] =  0.5*a-0.25*c*c/a; v[2] =  0.25*c;

   // Interior Nodes
   v = mesh->GetVertex(32);
   v[0] = -0.5*a+0.25*c*c/a; v[1] =  0.0;              v[2] =  0.0;
   v = mesh->GetVertex(33);
   v[0] =  0.5*a-0.25*c*c/a; v[1] =  0.0;              v[2] =  0.0;
   v = mesh->GetVertex(34);
   v[0] =  0.0;              v[1] = -0.5*a+0.25*c*c/a; v[2] =  0.0;
   v = mesh->GetVertex(35);
   v[0] =  0.0;              v[1] =  0.5*a-0.25*c*c/a; v[2] =  0.0;
   v = mesh->GetVertex(36);
   v[0] =  0.0;              v[1] = -0.5*c*c/a+0.5*a;  v[2] = -0.25*c;
   v = mesh->GetVertex(37);
   v[0] =  0.5*c*c/a-0.5*a;  v[1] =  0.0;              v[2] =  0.25*c;

}

void trans_elong_dod(Mesh * mesh, double a, double b, double c, double alpha)
{
   assert( c > sqrt(2.0) * a );
   assert(mesh->GetNV() == 24);

   double * v = NULL;

   // Bottom
   v = mesh->GetVertex(0);
   v[0] =  0.0;              v[1] =  0.0;              v[2] = -0.5*a*a/c-0.25*c;

   // 1st Ring
   v = mesh->GetVertex(1);
   v[0] =  0.0;              v[1] = -0.5*a;            v[2] = -0.25*c;
   v = mesh->GetVertex(2);
   v[0] =  0.0;              v[1] =  0.5*a;            v[2] = -0.25*c;
   v = mesh->GetVertex(3);
   v[0] = -0.5*a;            v[1] =  0.0;              v[2] = -0.25*c;
   v = mesh->GetVertex(4);
   v[0] =  0.5*a;            v[1] =  0.0;              v[2] = -0.25*c;

   // 2nd Ring
   v = mesh->GetVertex(5);
   v[0] = -0.5*a;            v[1] = -0.5*a;            v[2] = -0.25*c+0.5*a*a/c;
   v = mesh->GetVertex(6);
   v[0] = -0.5*a;            v[1] =  0.5*a;            v[2] = -0.25*c+0.5*a*a/c;
   v = mesh->GetVertex(7);
   v[0] =  0.5*a;            v[1] = -0.5*a;            v[2] = -0.25*c+0.5*a*a/c;
   v = mesh->GetVertex(8);
   v[0] =  0.5*a;            v[1] =  0.5*a;            v[2] = -0.25*c+0.5*a*a/c;

   // 1st Interior Point
   v = mesh->GetVertex(9);
   v[0] =  0.0;              v[1] =  0.0;              v[2] = -0.25*c+0.5*a*a/c;

   // Hexagon Centers
   v = mesh->GetVertex(10);
   v[0] = -0.5*a;            v[1] =  0.0;              v[2] = -0.25*c+a*a/c;
   v = mesh->GetVertex(11);
   v[0] =  0.5*a;            v[1] =  0.0;              v[2] = -0.25*c+a*a/c;
   v = mesh->GetVertex(12);
   v[0] =  0.0;              v[1] = -0.5*a;            v[2] =  0.25*c-a*a/c;
   v = mesh->GetVertex(13);
   v[0] =  0.0;              v[1] =  0.5*a;            v[2] =  0.25*c-a*a/c;

   // 2nd Interior Point
   v = mesh->GetVertex(14);
   v[0] =  0.0;              v[1] =  0.0;              v[2] =  0.25*c-0.5*a*a/c;

   // 3rd Ring
   v = mesh->GetVertex(15);
   v[0] = -0.5*a;            v[1] = -0.5*a;            v[2] =  0.25*c-0.5*a*a/c;
   v = mesh->GetVertex(16);
   v[0] = -0.5*a;            v[1] =  0.5*a;            v[2] =  0.25*c-0.5*a*a/c;
   v = mesh->GetVertex(17);
   v[0] =  0.5*a;            v[1] = -0.5*a;            v[2] =  0.25*c-0.5*a*a/c;
   v = mesh->GetVertex(18);
   v[0] =  0.5*a;            v[1] =  0.5*a;            v[2] =  0.25*c-0.5*a*a/c;

   // 4th Ring
   v = mesh->GetVertex(19);
   v[0] =  0.0;              v[1] = -0.5*a;            v[2] =  0.25*c;
   v = mesh->GetVertex(20);
   v[0] =  0.0;              v[1] =  0.5*a;            v[2] =  0.25*c;
   v = mesh->GetVertex(21);
   v[0] = -0.5*a;            v[1] =  0.0;              v[2] =  0.25*c;
   v = mesh->GetVertex(22);
   v[0] =  0.5*a;            v[1] =  0.0;              v[2] =  0.25*c;

   // Top
   v = mesh->GetVertex(23);
   v[0] =  0.0;              v[1] =  0.0;              v[2] =  0.5*a*a/c+0.25*c;
}
