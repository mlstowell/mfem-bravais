#include "mfem.hpp"
#include "../common/pfem_extras.hpp"
#include "../common/bravais.hpp"
#include<fstream>
#include<iostream>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;
using namespace mfem::bravais;

void
PrintLocalMatrix(BilinearForm & bf,
                 FiniteElementSpace & fes,
                 const string & file);

void
PrintLocalMatrix(DiscreteLinearOperator & dlo,
                 FiniteElementSpace & fesD,
                 FiniteElementSpace & fesR,
                 const string & file);

void
PrintDoFMapping(FiniteElementSpace & fes,
                const string & file);

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
   const char *sym_pt_lbl = "R";
   int order = 1;
   int sr = 0, pr = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sr, "-sr", "--serial-refinement",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-pr", "--parallel-refinement",
                  "Number of parallel refinement levels.");
   args.AddOption(&sym_pt_lbl, "-pt", "--symmetry-point",
                  "Cubic symmetry point: Gamma, X, M, R.");
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

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
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

   int dim = pmesh->Dimension();
   ParFiniteElementSpace * H1FESpace    = new H1_ParFESpace(pmesh,order,dim);
   ParFiniteElementSpace * HCurlFESpace = new ND_ParFESpace(pmesh,order,dim);
   ParFiniteElementSpace * HDivFESpace  = new RT_ParFESpace(pmesh,order,dim);

   PrintDoFMapping(   *H1FESpace,"h1_dofs.vec");
   PrintDoFMapping(*HCurlFESpace,"hcurl_dofs.vec");
   PrintDoFMapping( *HDivFESpace,"hdiv_dofs.vec");

   BravaisLattice * bravais = new CubicLattice();
   map<string,int> sym_pt_idx;
   sym_pt_idx["Gamma"] = 0;
   sym_pt_idx["X"]     = 1;
   sym_pt_idx["M"]     = 2;
   sym_pt_idx["R"]     = 3;

   // The eigenvalue problem being solved is a 2x2 block system given by:
   //
   // |  A1  DKZ ||Er|          |M1   0 ||Er|
   // |-DKZ   A1 ||Ei| = lambda | 0  M1 ||Ei|
   //
   // M1 is a standard H(curl) mass matrix
   // A1 is similar to an H(curl) stiffness matrix
   // DKZ computes Curl(kappa x E) - kappa x Curl(E)
   // Er and Ei are the real and imaginary parts of the Electric field

   Vector kappa(3);
   bravais->GetSymmetryPoint(sym_pt_idx[sym_pt_lbl],kappa);
   kappa *= M_PI;

   ParBilinearForm m1(HCurlFESpace);
   m1.AddDomainIntegrator(new VectorFEMassIntegrator);
   m1.Assemble();
   m1.Finalize();
   HypreParMatrix * M1 = m1.ParallelAssemble();
   M1->Print("M1.mat");
   PrintLocalMatrix(m1,*HCurlFESpace,"m1_loc.mat");

   ParBilinearForm m2(HCurlFESpace);
   m2.AddDomainIntegrator(new VectorFEMassIntegrator);
   m2.Assemble();
   m2.Finalize();
   HypreParMatrix * M2 = m2.ParallelAssemble();
   M2->Print("M2.mat");
   PrintLocalMatrix(m2,*HDivFESpace,"m2_loc.mat");

   ParDiscreteGradOperator t01(H1FESpace, HCurlFESpace);
   t01.Assemble();
   t01.Finalize();
   HypreParMatrix * T01 = t01.ParallelAssemble();
   T01->Print("T01.mat");
   PrintLocalMatrix(t01,*H1FESpace,*HCurlFESpace,"t01_loc.mat");

   ParDiscreteCurlOperator t12(HCurlFESpace, HDivFESpace);
   t12.Assemble();
   t12.Finalize();
   HypreParMatrix * T12 = t12.ParallelAssemble();
   T12->Print("T12.mat");
   PrintLocalMatrix(t12,*HCurlFESpace,*HDivFESpace,"t12_loc.mat");

   ParDiscreteVectorProductOperator z01(H1FESpace, HCurlFESpace,kappa);
   z01.Assemble();
   z01.Finalize();
   HypreParMatrix * Z01 = z01.ParallelAssemble();
   Z01->Print("Z01.mat");
   PrintLocalMatrix(z01,*H1FESpace,*HCurlFESpace,"z01_loc.mat");

   ParDiscreteVectorCrossProductOperator z12(HCurlFESpace,
                                             HDivFESpace,kappa);
   z12.Assemble();
   z12.Finalize();
   HypreParMatrix * Z12 = z12.ParallelAssemble();
   Z12->Print("Z12.mat");
   PrintLocalMatrix(z12,*HCurlFESpace,*HDivFESpace,"z12_loc.mat");

   HypreParMatrix * CMC = RAP(M2, T12);
   HypreParMatrix * ZMZ = RAP(M2, Z12);
   HypreParMatrix * CMZ = RAP(T12, M2, Z12);
   HypreParMatrix * ZMC = RAP(Z12, M2, T12);
   *ZMC *= -1.0;
   HypreParMatrix * DKZ = ParAdd(CMZ,ZMC);
   DKZ->Print("DKZ.mat");

   HypreParMatrix *  A1 = ParAdd(CMC,ZMZ);
   A1->Print("A1.mat");

   delete CMC;
   delete CMZ;
   delete ZMC;
   delete ZMZ;
   delete DKZ;
   delete A1;

   HypreParMatrix * GMG  = RAP(M1,T01);
   HypreParMatrix * ZMZ0 = RAP(M1,Z01);
   HypreParMatrix * GMZ  = RAP(T01,M1,Z01);
   HypreParMatrix * ZMG  = RAP(Z01,M1,T01);
   *ZMG *= -1.0;
   HypreParMatrix * DKZ0 = ParAdd(GMZ,ZMG);
   DKZ0->Print("DKZ0.mat");

   HypreParMatrix *  A0 = ParAdd(GMG,ZMZ0);
   A0->Print("A0.mat");
   /*
   cout << "abc 0" << endl;
   ParBilinearForm abc1(HCurlFESpace);
   cout << "abc 1" << endl;
   abc1.AddBoundaryIntegrator(new VectorFEMassIntegrator);
   cout << "abc 2" << endl;
   abc1.Assemble();
   cout << "abc 3" << endl;
   abc1.Finalize();
   cout << "abc 4" << endl;
   HypreParMatrix * ABC1 = abc1.ParallelAssemble();
   cout << "abc 5" << endl;
   ABC1->Print("ABC1.mat");
   cout << "abc 6" << endl;
   PrintLocalMatrix(abc1,*HCurlFESpace,"abc1_loc.mat");
   cout << "abc 7" << endl;
   delete ABC1;
   cout << "abc 8" << endl;
   */
   delete GMG;
   delete GMZ;
   delete ZMG;
   delete ZMZ0;
   delete DKZ0;
   delete A0;

   delete M1;
   delete M2;

   delete T01;
   delete T12;
   delete Z01;
   delete Z12;

   delete H1FESpace;
   delete HCurlFESpace;
   delete HDivFESpace;
   delete pmesh;
}

void
PrintLocalMatrix(BilinearForm & bf, FiniteElementSpace & fes,
                 const string & file)
{
   DenseMatrix mat;
   (*bf.GetDBFI())[0]->AssembleElementMatrix(*fes.GetFE(0),
                                             *fes.GetElementTransformation(0),
                                             mat);
   ofstream ofs(file.c_str());
   mat.Print(ofs,6);
   ofs.close();
}

void
PrintLocalMatrix(DiscreteLinearOperator & dlo,
                 FiniteElementSpace & fesD,
                 FiniteElementSpace & fesR,
                 const string & file)
{
   DenseMatrix mat;
   (*dlo.GetDI())[0]->AssembleElementMatrix2(*fesD.GetFE(0),
                                             *fesR.GetFE(0),
                                             *fesD.GetElementTransformation(0),
                                             mat);
   ofstream ofs(file.c_str());
   mat.Print(ofs,6);
   ofs.close();

}

void
PrintDoFMapping(FiniteElementSpace & fes,
                const string & file)
{
   Array<int> vdofs;

   ofstream ofs(file.c_str());
   ofs << fes.GetNE() << " " << fes.GetNDofs() << endl;

   for (int i=0; i<fes.GetNE(); i++)
   {
      fes.GetElementVDofs(i,vdofs);
      ofs << vdofs.Size();
      for (int j=0; j<vdofs.Size(); j++)
      {
         ofs << " " << vdofs[j];
      }
      ofs << endl;
   }

   ofs.close();
}
