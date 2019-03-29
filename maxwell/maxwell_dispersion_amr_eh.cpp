#include "mfem.hpp"
#include "maxwell_bloch_amr_eh.hpp"
#include "../common/bravais.hpp"
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include <cerrno>      // errno
#ifndef _WIN32
#include <sys/stat.h>  // mkdir
#else
#include <direct.h>    // _mkdir
#define mkdir(dir, mode) _mkdir(dir)
#endif

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;
using namespace mfem::bloch;
using namespace mfem::bravais;

// Material Coefficients
static int prob_ = -1;
double mass_coef(const Vector &);
double stiffness_coef(const Vector &);

int CreateDirectory(const string &dir_name, MPI_Comm & comm, int myid);

void PrintPhaseShifts(const vector<Vector> & lattice_vecs,
                      const Vector & kappa);

int  CountVectors(int lattice_type);

void CreateInitialVectors(int lattice_type,
                          const BravaisLattice & bravais,
                          const Vector & kappa,
                          ParFiniteElementSpace & fespace,
                          int & nev,
                          vector<HypreParVector*> & init_vecs);

void WriteDispersionData(int myid, ostream & os, int c,
                         const string & label, vector<double> & eigenvalues);

void BuildInitialMesh(Mesh & emesh, BravaisLattice & bravais, int nr, double h);

void RescaleEigMesh(Mesh & emesh,
                    double & scaleFactor, double maxEig, double height,
                    bool firstEigs);

void VisualizeMesh(MPI_Comm & comm,  int myid, int num_procs,
                   const string & title, Mesh & emesh, socketstream & sock);

class FourierVectorCoefficient
{
public:
   FourierVectorCoefficient();

   void SetN(int n0, int n1, int n2) { n_[0] = n0; n_[1] = n1; n_[2] = n2; }
   void SetA(const Vector & Ar, const Vector & Ai) { Ar_ = Ar; Ai_ = Ai; }

   const vector<int> & GetN()  const { return n_; }
   const Vector      & GetAr() const { return Ar_; }
   const Vector      & GetAi() const { return Ai_; }

   double GetEnergy() const { return Ar_ * Ar_ + Ai_ * Ai_; }

   complex<double> operator*(const FourierVectorCoefficient & v) const;

   void Print(ostream & os);

private:
   vector<int> n_;
   Vector Ar_;
   Vector Ai_;
};

class FourierVectorCoefficients
{
public:
   FourierVectorCoefficients();
   ~FourierVectorCoefficients();

   void   SetOmega( double omega) { omega_ = omega; }
   double GetOmega() const        { return omega_; }
   double GetEnergy() const;
   double GetEnergy(int tier) const;

   void AddCoefficient(int n0, int n1, int n2,
                       const Vector & Ar,
                       const Vector & Ai);

   void GetCoefficient(int n0, int n1, int n2,
                       Vector & Ar,
                       Vector & Ai) const;

   const FourierVectorCoefficient * GetCoefficient(int n0,int n1,int n2) const;


   complex<double> operator*(const FourierVectorCoefficients & v) const;

   void Print(ostream & os);

private:
   double omega_;

   map<int,set<FourierVectorCoefficient*> > coefs_;
};

void ComputeFourierCoefficients(int myid, ostream & ofs_coef, int nmax,
                                HCurlFourierSeries & fourier,
                                MaxwellBlochWaveEquationAMR & eq,
                                ParFiniteElementSpace & HCurlFESpace,
                                const vector<double> & eigenvalues,
                                map<int,FourierVectorCoefficients> & mfc);

void CompareFourierCoefficients(
   map<int, map<int, FourierVectorCoefficients> > & mfc);

void IdentifyDegeneracies(const vector<double> & eigenvalues,
                          double zero_tol, double rel_tol,
                          vector<set<int> > & degen);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // 2. Parse command-line options.
   int bl_type = 1;
   string lattice_label = "";
   int order = 1;
   int sr = 0, pr = 0;
   int ar = 0;
   int logging = 0;
   // bool midpoints = true;
   bool visualization = true;
   bool visit = true;
   // int nev = 0;
   // int np = 1;
   double a = -1.0, b = -1.0, c = -1.0;
   double alpha = -1.0, beta = -1.0, gamma = -1.0;
   double lcf = -1.0;
   double etol = 1.0e-2;

   OptionsParser args(argc, argv);
   args.AddOption(&bl_type, "-bl", "--bravais-lattice",
                  "Bravais Lattice Type: \n"
                  "            1 - Primitive Cubic\n"
                  "            2 - Body-Centered Cubic\n"
                  "            3 - Face-Centered Cubic\n"
                  "            4 - Primitive Tetragonal\n"
                  "            5 - Body-Centered Tetragonal\n"
                  "            6 - Primitive Orthorhombic\n"
                  "            7 - Face-Centered Orthorhombic\n"
                  "            8 - Body-Centered Orthorhombic\n"
                  "            9 - Base-Centered Orthorhombic\n"
                  "           10 - Primitive Hexagonal\n"
                  "           11 - Primitive Rhombohedral\n"
                  "           12 - Primitive Monoclinic\n"
                  "           13 - Base-Centered Monoclinic\n"
                  "           14 - Triclinic"
                 );
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sr, "-sr", "--serial-refinement",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-pr", "--parallel-refinement",
                  "Number of parallel refinement levels.");
   args.AddOption(&ar, "-ar", "--adaptive-refinement",
                  "Number of adaptive refinement levels.");
   args.AddOption(&prob_, "-p", "--problem-type",
                  "Problem Geometry.");
   args.AddOption(&a, "-a", "--lattice-a",
                  "Lattice spacing a");
   args.AddOption(&b, "-b", "--lattice-b",
                  "Lattice spacing b");
   args.AddOption(&c, "-c", "--lattice-c",
                  "Lattice spacing c");
   args.AddOption(&alpha, "-alpha", "--lattice-alpha",
                  "Lattice angle alpha");
   args.AddOption(&beta, "-beta", "--lattice-beta",
                  "Lattice angle beta");
   args.AddOption(&gamma, "-gamma", "--lattice-gamma",
                  "Lattice angle gamma");
   args.AddOption(&lcf, "-lcf", "--lattice-coef-frac",
                  "Fraction of inscribed circle radius for rods");
   // args.AddOption(&midpoints, "-mp", "--mid-points", "-no-mid-points",
   //               "--no-mid-points",
   //               "Enable or disable mid-point calculations.");
   // args.AddOption(&np, "-np", "--num-points",
   //               "Number of intermediate points between symmetry points.");
   args.AddOption(&logging, "-l", "--logging",
                  "Output message level.");
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
   /*
   if ( midpoints && np%2 == 0 )
   {
      np++;
   }
   */
   if (myid == 0)
   {
      cout << "Creating symmetry points for lattice " << bl_type << endl;
   }

   BRAVAIS_LATTICE_TYPE lattice_type = (BRAVAIS_LATTICE_TYPE)(bl_type + 5);
   BravaisLattice * bravais = BravaisLatticeFactory(lattice_type,
                                                    a, b, c,
                                                    alpha, beta, gamma,
                                                    logging);
   lattice_label = bravais->GetLatticeTypeLabel();

   Mesh * mesh = bravais->GetCoarseWignerSeitzMesh();

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
   mesh->EnsureNCMesh();

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   /*
   ParMesh *pmesh = new ParMesh(comm, *mesh);
   delete mesh;
   {
      int par_ref_levels = pr;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   */

   map<string,MaxwellBlochWaveEquationAMR *> eq;
   // HCurlFourierSeries fourier_hcurl(*bravais, *HCurlFESpace);
   /*
   HYPRE_Int size = eq->GetHCurlFESpace()->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of complex unknowns: " << size << endl;
   }
   */
   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets
   // int offy = Wh+45; // window offsets

   /*
   ND_ParFESpace * HCurlFESpace = new ND_ParFESpace(pmesh, order,
                                                    pmesh->Dimension());
   L2_ParFESpace * L2FESpace    = new L2_ParFESpace(pmesh, 0,
                                                    pmesh->Dimension());

   int nElems = L2FESpace->GetVSize();
   cout << myid << ": nElems = " << nElems << endl;

   ParGridFunction * m = new ParGridFunction(L2FESpace);
   ParGridFunction * k = new ParGridFunction(L2FESpace);

   FunctionCoefficient mFunc(mass_coef);
   FunctionCoefficient kFunc(stiffness_coef);

   m->ProjectCoefficient(mFunc);
   k->ProjectCoefficient(kFunc);

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets
   // int offy = Wh+45; // window offsets
   if (visualization)
   {
      socketstream m_sock, k_sock;

      VisualizeField(m_sock, vishost, visport, *m,
                     "Mass Coefficient", Wx, Wy, Ww, Wh);
      VisualizeField(k_sock, vishost, visport, *k,
                     "Stiffness Coefficient", Wx, Wy+offy, Ww, Wh);
   }

   GridFunctionCoefficient mCoef(m);
   GridFunctionCoefficient kCoef(k);
   */

   LatticeCoefficient  epsLatCoef(*bravais, lcf, 1.0, 10.0);
   ConstantCoefficient muLatCoef(1.0);
   // PowerCoefficient mInvLatCoef(mLatCoef, -1);

   FunctionCoefficient mFunc(mass_coef);
   FunctionCoefficient aFunc(stiffness_coef);

   // eq->SetLatticeSize(a);
   // eq->SetNumEigs(nev);
   // eq->SetMassCoef(mFunc);
   // eq->SetStiffnessCoef(aFunc);

   // DenseMatrix dispersion(num_beta,nev);

   ostringstream oss_prefix;
   oss_prefix << "Maxwell-Dispersion-AMR-EH-";
   oss_prefix << lattice_label;

   CreateDirectory(oss_prefix.str(),comm,myid);

   ostringstream oss_disp, oss_coef;
   oss_disp << oss_prefix.str() << "/disp.dat";
   oss_coef << oss_prefix.str() << "/coef.dat";

   ofstream ofs_disp, ofs_coef;

   if ( myid == 0 )
   {
      ofs_disp.open(oss_disp.str().c_str());
      ofs_coef.open(oss_coef.str().c_str());
   }

   // HypreParVector ** init_vecs = new HypreParVector*[nev];
   // for (int i=0; i<nev; i++) init_vecs[i] = NULL;
   // vector<HypreParVector*> init_vecs;

   Vector kappa(3); kappa = 0.0;
   vector<Vector> lattice_vecs;
   bravais->GetLatticeVectors(lattice_vecs);
   /*
   CreateInitialVectors(bl_type, *bravais, kappa,
                        *eq->GetHCurlFESpace(),
                        nev, init_vecs);

   */
   // Prepare specialized mesh for visualization
   int nr     = CountVectors(bl_type);
   int nseg   = 0;
   int nnode  = 0;
   int nelem  = 0;
   int nbelem = 0;
   for (unsigned int p=0; p<bravais->GetNumberPaths(); p++)
   {
      nseg   += bravais->GetNumberPathSegments(p);
      nnode  += 2 * (2 * bravais->GetNumberPathSegments(p) + 1);
      nelem  += 2 * bravais->GetNumberPathSegments(p);
      nbelem += 4 * bravais->GetNumberPathSegments(p);
      nbelem += 2;
   }
   double gr = 0.5 + sqrt(1.25);
   double w  = nseg;
   double h  = w / gr;
   double sf = 1.0;

   cout << "nseg " << nseg << endl;

   Mesh * emesh = new Mesh(2, nr * nnode, nr * nelem,
                           0 * nr * nbelem, 3);

   BuildInitialMesh(*emesh, *bravais, nr, h);

   // char vishost[] = "localhost";
   // int  visport   = 19916;
   socketstream a_sock, m_sock;
   socketstream e_sock(vishost, visport);
   // a_sock.precision(8);
   // m_sock.precision(8);
   e_sock.precision(8);

   if ( visualization )
   {
      VisualizeMesh(comm, myid, num_procs, "Dispersion", *emesh, e_sock);
      e_sock << "window_geometry "
             << Wx+offx << " " << Wy << " "
             << 2*Wh << " " << (int)floor(Wh * gr) << "\n"
             << "keys\n maa\n" << "axis_labels 'b' ' ' 'omega'\n"<< flush;
   }

   cout << "NNode/NElem " << nr*nnode << " " << nr * nelem << endl;

   int ci = 0;
   int vtx = 0;
   string label = "";

   map<string,vector<double> > sp_eigs;
   vector<double> eigenvalues;
   DenseMatrix iMv;
   ofstream ofs_iMv("iMv.dat");

   ostringstream oss_title;

   map<string,int> c_by_label;
   map<int, vector<set<int> > > degen;
   // map<int, map<int, FourierVectorCoefficients> > mfc;

   set<int> modes;
   for (int i=0; i<8; i++) { modes.insert(i); }

   // for (unsigned int p=0; p<bravais->GetNumberPaths(); p++)
   for (unsigned int p=0; p<1; p++)
   {
      cout << "Starting path" << endl;
      int e0, e1;
      int i0 = 0;
      bravais->GetPathSegmentEndPointIndices(p, 0, e0, e1);
      bravais->GetSymmetryPoint(e0, kappa);
      label = bravais->GetSymmetryPointLabel(e0);
      cout << "Symmetry point: " << label << endl;
      if ( p > 0 ) { oss_title << "|"; }
      oss_title << label;
      if ( label == "Gamma" ) { i0 = 2; }
      //if ( label == "Gamma" ) { continue; }
      if ( false)
      {
         if ( eq.find(label) == eq.end() )
         {
            eq[label] =
               new MaxwellBlochWaveEquationAMR(comm, *mesh, order, ar, logging);
            if ( lcf > 0.0 )
            {
               eq[label]->SetEpsilonCoef(epsLatCoef);
               eq[label]->SetMuCoef(muLatCoef);
            }
            else
            {
               eq[label]->SetEpsilonCoef(mFunc);
               eq[label]->SetMuCoef(aFunc);
            }
         }
         if ( sp_eigs.find(label) == sp_eigs.end() )
         {
            /*
            cout << "Creating initial vectors" << endl;
                  CreateInitialVectors(bl_type, *bravais, kappa,
                                       *eq[label]->GetHCurlFESpace(),
                                       nev, init_vecs);
            */
            cout << "Starting eigenmode computation" << endl;
            eq[label]->GetEigenvalues(kappa, sp_eigs[label]);

            eq[label]->DisplayToGLVis(a_sock, m_sock, vishost, visport,
                                      Wx, Wy, Ww, Wh, offx, offy);

            IdentifyDegeneracies(sp_eigs[label], 1.0e-4, 1.0e-4, degen[ci]);
            /*
                 ComputeFourierCoefficients(myid, ofs_coef, 1,
                                            fourier_hcurl, *eq,
                                            *HCurlFESpace,
                                            sp_eigs[label],
                                            mfc[ci]);
            */
            if (label != "Gamma" )
            {
               eq[label]->GetInnerProducts(iMv);
               ofs_iMv << endl << label << endl;
               iMv.Print(ofs_iMv);
               ofs_iMv << endl;
            }
         }
         cout << "Writing data to disp.dat" << endl;
         WriteDispersionData(myid,ofs_disp,ci,label,sp_eigs[label]);
         ci++;

         RescaleEigMesh(*emesh, sf, sp_eigs[label][sp_eigs[label].size() - 1],
                        h, true);
      }
      /*
      int ilast = sp_eigs[label].size() - 1;
      if ( p == 0 && sqrt(fabs(sp_eigs[label][ilast])) < h )
      {
      sf = h / sqrt(fabs(sp_eigs[label][ilast]));
           }
           else if ( sf * sqrt(fabs(sp_eigs[label][ilast])) > h )
           {
       double prev_sf = sf;
       sf = h / sqrt(fabs(sp_eigs[label][ilast]));

       for (int i=0; i<nr*nnode; i++)
       {
          double * vc = emesh->GetVertex(i);
          vc[2] *= sf / prev_sf;
       }
           }
           */
      for (unsigned int i=i0; i<sp_eigs[label].size(); i++)
      {
         double * vc0 = emesh->GetVertex(vtx);
         double * vc1 = emesh->GetVertex(vtx+1);

         vc0[2] = sf * sqrt(fabs(sp_eigs[label][i]));
         vc1[2] = sf * sqrt(fabs(sp_eigs[label][i]));

         vtx += 2;
      }

      if ( visualization )
      {
         VisualizeMesh(comm, myid, num_procs, oss_title.str(), *emesh, e_sock);
      }

      // for (unsigned int s=0; s<bravais->GetNumberPathSegments(p); s++)
      for (unsigned int s=0; s<1; s++)
      {
         bravais->GetIntermediatePoint(p,s,kappa);
         label = bravais->GetIntermediatePointLabel(p,s);
         cout << "Intermediate point: " << label << endl;
         oss_title << "-" << label;

         if ( eq.find(label) == eq.end() )
         {
            eq[label] =
               new MaxwellBlochWaveEquationAMR(comm, *mesh, order, ar, logging);
            if ( lcf > 0.0 )
            {
               eq[label]->SetEpsilonCoef(epsLatCoef);
               eq[label]->SetMuCoef(muLatCoef);
            }
            else
            {
               eq[label]->SetEpsilonCoef(mFunc);
               eq[label]->SetMuCoef(aFunc);
            }
         }
         /*
              cout << "Creating initial vectors" << endl;
              CreateInitialVectors(bl_type, *bravais, kappa,
                                   *eq[label]->GetHCurlFESpace(),
                                   nev, init_vecs);
         */
         cout << "Starting eigenmode computation" << endl;
         eq[label]->GetEigenvalues(kappa, modes, etol,
                                   eigenvalues);
         eq[label]->DisplayToGLVis(a_sock, m_sock, vishost, visport,
                                   Wx, Wy, Ww, Wh, offx, offy);

         eq[label]->WriteVisitFields(oss_prefix.str(),label);

         cout << "Writing data to disp.dat" << endl;
         WriteDispersionData(myid,ofs_disp,ci,label,eigenvalues);
         IdentifyDegeneracies(eigenvalues, 1.0e-4, 1.0e-4, degen[ci]);
         /*
              ComputeFourierCoefficients(myid, ofs_coef, 1,
                                         fourier_hcurl, *eq,
                                         *HCurlFESpace,
                                         eigenvalues,
                                         mfc[ci]);
         */
         eq[label]->GetInnerProducts(iMv);
         ofs_iMv << endl << label << endl;
         iMv.Print(ofs_iMv);
         ofs_iMv << endl;

         ci++;

         RescaleEigMesh(*emesh, sf, eigenvalues[eigenvalues.size() - 1],
                        h, false);
         /*
         ilast = eigenvalues.size() - 1;
         if ( sf * sqrt(fabs(eigenvalues[ilast])) > h )
         {
           double prev_sf = sf;
           sf = h / sqrt(fabs(eigenvalues[ilast]));

           for (int i=0; i<nr*nnode; i++)
           {
             double * vc = emesh->GetVertex(i);
             vc[2] *= sf / prev_sf;
           }
         }
         */
         for (unsigned int i=0; i<eigenvalues.size(); i++)
         {
            double * vc0 = emesh->GetVertex(vtx);
            double * vc1 = emesh->GetVertex(vtx+1);

            vc0[2] = sf * sqrt(fabs(eigenvalues[i]));
            vc1[2] = sf * sqrt(fabs(eigenvalues[i]));

            vtx += 2;
         }

         if ( visualization )
         {
            VisualizeMesh(comm, myid, num_procs, oss_title.str(),
                          *emesh, e_sock);
         }
         if (false)
         {
            bravais->GetPathSegmentEndPointIndices(p, s, e0, e1);
            bravais->GetSymmetryPoint(e1, kappa);
            label = bravais->GetSymmetryPointLabel(e1);
            cout << "Symmetry point: " << label << endl;
            oss_title << "-" << label;
            if ( eq.find(label) == eq.end() )
            {
               eq[label] =
                  new MaxwellBlochWaveEquationAMR(comm, *mesh, order, ar, logging);
               if ( lcf > 0.0 )
               {
                  eq[label]->SetEpsilonCoef(epsLatCoef);
                  eq[label]->SetMuCoef(muLatCoef);
               }
               else
               {
                  eq[label]->SetEpsilonCoef(mFunc);
                  eq[label]->SetMuCoef(aFunc);
               }
            }
            if ( sp_eigs.find(label) == sp_eigs.end() )
            {
               /*
                cout << "Creating initial vectors" << endl;
                     CreateInitialVectors(bl_type, *bravais, kappa,
                                          *eq[label]->GetHCurlFESpace(),
                                          nev, init_vecs);
               */
               cout << "Starting eigenmode computation" << endl;
               eq[label]->GetEigenvalues(kappa, modes, etol,
                                         sp_eigs[label]);
               eq[label]->DisplayToGLVis(a_sock, m_sock, vishost, visport,
                                         Wx, Wy, Ww, Wh, offx, offy);
               IdentifyDegeneracies(sp_eigs[label], 1.0e-4, 1.0e-4, degen[ci]);
               /*
                    ComputeFourierCoefficients(myid, ofs_coef, 1,
                                               fourier_hcurl, *eq,
                                               *HCurlFESpace,
                                               sp_eigs[label],
                                               mfc[ci]);
               */
               if (label != "Gamma" )
               {
                  eq[label]->GetInnerProducts(iMv);
                  ofs_iMv << endl << label << endl;
                  iMv.Print(ofs_iMv);
                  ofs_iMv << endl;
               }
            }
            cout << "Writing data to disp.dat" << endl;
            WriteDispersionData(myid,ofs_disp,ci,label,sp_eigs[label]);
            ci++;

            if ( label == "Gamma" ) { i0 = 2; }
            else { i0 = 0; }
            RescaleEigMesh(*emesh, sf, sp_eigs[label][sp_eigs[label].size() - 1],
                           h, false);
            /*
            ilast = sp_eigs[label].size() - 1;
            if ( sf * sqrt(fabs(sp_eigs[label][ilast])) > h )
            {
               double prev_sf = sf;
               sf = h / sqrt(fabs(sp_eigs[label][ilast]));

               for (int i=0; i<nr*nnode; i++)
               {
                  double * vc = emesh->GetVertex(i);
                  vc[2] *= sf / prev_sf;
               }
            }
            */
            for (unsigned int i=i0; i<sp_eigs[label].size(); i++)
            {
               double * vc0 = emesh->GetVertex(vtx);
               double * vc1 = emesh->GetVertex(vtx+1);

               vc0[2] = sf * sqrt(fabs(sp_eigs[label][i]));
               vc1[2] = sf * sqrt(fabs(sp_eigs[label][i]));

               vtx += 2;
            }

            if ( visualization )
            {
               VisualizeMesh(comm, myid, num_procs, oss_title.str(),
                             *emesh, e_sock);
            }
         }
      }
      cout << "End of path" << endl;
   }
   ofs_iMv.close();

   {
      if ( myid == 0 )
      {
         ofstream ofs_emesh("eigs.mesh");
         emesh->Print(ofs_emesh);
      }
   }

   /*

   // set<string> sp;
   map<string,vector<double> > sp_eigs;
   map<string,int> c_by_label;
   map<int, vector<set<int> > > degen;
   // map<int, map<int, FourierVectorCoefficients> > mfc;

   for (unsigned int p=0; p<bravais->GetNumberPaths(); p++)
      // for (unsigned int p=1; p<bravais->GetNumberPaths(); p++)
   {
      int e0 = -1, e1 = -1;
      string label = "";
      Vector kappa0(3), kappa1(3);
      vector<double> eigenvalues;

      for (unsigned int s=0; s<bravais->GetNumberPathSegments(p); s++)
      {
         bravais->GetPathSegmentEndPointIndices(p,s,e0,e1);
         bravais->GetSymmetryPoint(e0,kappa0);
         bravais->GetSymmetryPoint(e1,kappa1);

         for (int i=0; i<=np; i++)
         {
            add(double(np+1-i)/(np+1),kappa0,double(i)/(np+1),kappa1,kappa);

            if ( i == 0 )
            {
               label = bravais->GetSymmetryPointLabel(e0);
            }
            else if ( np % 2 == 1 && i == (np + 1) / 2 )
            {
               label = bravais->GetIntermediatePointLabel(p,s);
            }
            else
            {
               label = "-";
            }

            if ( ( i == 0 && sp_eigs.find(label) == sp_eigs.end() ) || i != 0 )
            {
               if ( myid == 0 )
               {
                  if ( label != "-" )
                  {
                     cout << "Computing modes for symmetry point \""
                          << label << "\"." << endl;
                     PrintPhaseShifts(lattice_vecs, kappa);
           cout << endl;
                  }
                  else
                  {
                     cout << "Computing modes for the point: " << endl;
                     PrintPhaseShifts(lattice_vecs, kappa);
                     cout << "between" << endl;
                     PrintPhaseShifts(lattice_vecs, kappa0);
                     cout << "and" << endl;
                     PrintPhaseShifts(lattice_vecs, kappa1);
           cout << endl;
                  }
               }

          if ( p != 0 || s != 0 )
       CreateInitialVectors(bl_type, *bravais, kappa,
                  *eq->GetHCurlFESpace(),
                  nev, init_vecs);

               eq->GetEigenvalues(nev, kappa, init_vecs, eigenvalues);

               if ( i == 0 )
               {
                  sp_eigs[label].resize(eigenvalues.size());
                  for (unsigned int j=0; j<eigenvalues.size(); j++)
                  {
                     sp_eigs[label][j] = eigenvalues[j];
                  }
               }

               if ( visit )
               {
                  if ( i == 0 || ( midpoints && i == ( np + 1 ) / 2 ) )
                  {
                     eq->WriteVisitFields(oss_prefix.str(),label);
                  }
               }

               WriteDispersionData(myid,ofs_disp,ci,label,eigenvalues);

               IdentifyDegeneracies(eigenvalues, 1.0e-4, 1.0e-4, degen[ci]);
               ci++;
            }
            else
            {
               WriteDispersionData(myid,ofs_disp,ci,label,sp_eigs[label]);
               ci++;
            }
         }
      }

      label = bravais->GetSymmetryPointLabel(e1);

      if ( sp_eigs.find(label) == sp_eigs.end() )
      {
         if ( myid == 0 )
         {
            cout << "Computing modes for symmetry point \""
                 << label << "\"." << endl;
            PrintPhaseShifts(lattice_vecs, kappa1);
         }

         CreateInitialVectors(bl_type, *bravais, kappa1,
                              *eq->GetHCurlFESpace(),
                              nev, init_vecs);
         eq->GetEigenvalues(nev, kappa1, init_vecs, eigenvalues);

         sp_eigs[label].resize(eigenvalues.size());
         for (unsigned int i=0; i<eigenvalues.size(); i++)
         {
            sp_eigs[label][i] = eigenvalues[i];
         }

         WriteDispersionData(myid,ofs_disp,ci,label,eigenvalues);
         ci++;

         if ( visit ) { eq->WriteVisitFields(oss_prefix.str(),label); }
      }
      else
      {
         WriteDispersionData(myid,ofs_disp,ci,label,sp_eigs[label]);
         ci++;
      }
   }
   ofs_disp.close();
   */
   /*
   map<int, map<int,FourierVectorCoefficients> >::iterator mmit;
   map<int,FourierVectorCoefficients>::iterator mit;
   set<int>::iterator sit;
   for (mmit=mfc.begin(); mmit!=mfc.end(); mmit++)
   {
      ofs_coef << "Kappa index:  " << mmit->first << endl;
      ofs_coef << "Degeneracies: " << endl;
      for (unsigned int i=0; i<degen[mmit->first].size(); i++)
      {
         for (sit=degen[mmit->first][i].begin(); sit!=degen[mmit->first][i].end(); sit++)
         {
            ofs_coef << "\t" << *sit;
         }
         ofs_coef << endl;
      }
      for (mit=mmit->second.begin(); mit!=mmit->second.end(); mit++)
      {
         ofs_coef << "Eigenvalue Index:  " << mit->first << endl;
         mit->second.Print(ofs_coef);
      }
      ofs_coef << endl;
   }
   ofs_coef.close();
   */
   // for (int i=0; i<nev; i++) { delete init_vecs[i]; }

   // delete HCurlFESpace;
   // delete L2FESpace;
   delete bravais;

   map<string,MaxwellBlochWaveEquationAMR *>::iterator eqit;
   for (eqit=eq.begin(); eqit!=eq.end(); eqit++) { delete eqit->second; }
   delete mesh;

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

void
PrintPhaseShifts(const vector<Vector> & lattice_vecs,
                 const Vector & kappa)
{
   for (unsigned int i=0; i<lattice_vecs.size(); i++)
   {
      cout << lattice_vecs[i] * kappa * 180.0 / M_PI
           << "\tdegrees in direction ";
      lattice_vecs[i].Print(cout);
   }
}

int
CountVectors(int lattice_type)
{
   int n = 0;

   switch (lattice_type)
   {
      case 1:
      {
         n = 7;
      }
      break;
      case 2:
      {
         n = 13;
      }
      break;
      case 3:
      {
         n = 15;
      }
      break;
      default:
         cout << "Unsupported lattice type:  " << lattice_type << endl;
   };
   return 4 * n;
}

void
CreateInitialVectors(int lattice_type,
                     const BravaisLattice & bravais,
                     const Vector & kappa,
                     ParFiniteElementSpace & HCurlFESpace,
                     int & nev,
                     vector<HypreParVector*> & init_vecs)
{
   // Initialize MPI variables
   MPI_Comm comm = HCurlFESpace.GetComm();
   int numProcs  = HCurlFESpace.GetNRanks();

   RealModeCoefficient cosCoef;
   ImagModeCoefficient sinCoef;

   vector<Vector> b;
   bravais.GetReciprocalLatticeVectors(b);
   cosCoef.SetReciprocalLatticeVectors(b);
   sinCoef.SetReciprocalLatticeVectors(b);

   Vector v(3); v = 0.0;
   VectorFunctionCoefficient E0CosCoef(v,cosCoef);
   VectorFunctionCoefficient E0SinCoef(v,sinCoef);

   ParGridFunction Er(&HCurlFESpace);
   ParGridFunction Ei(&HCurlFESpace);
   ParGridFunction Hr(&HCurlFESpace);
   ParGridFunction Hi(&HCurlFESpace);

   Array<int> bOffsets(5);
   bOffsets[0] = 0;
   bOffsets[1] = HCurlFESpace.TrueVSize();
   bOffsets[2] = HCurlFESpace.TrueVSize();
   bOffsets[3] = HCurlFESpace.TrueVSize();
   bOffsets[4] = HCurlFESpace.TrueVSize();
   bOffsets.PartialSum();

   int locSize = 4*HCurlFESpace.TrueVSize();
   int glbSize = 0;

   HYPRE_Int * part = NULL;

   if (HYPRE_AssumedPartitionCheck())
   {
      part = new HYPRE_Int[2];

      MPI_Scan(&locSize, &part[1], 1, HYPRE_MPI_INT, MPI_SUM, comm);

      part[0] = part[1] - locSize;

      MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm);
   }
   else
   {
      part = new HYPRE_Int[numProcs+1];

      MPI_Allgather(&locSize, 1, MPI_INT,
                    &part[1], 1, HYPRE_MPI_INT, comm);

      part[0] = 0;
      for (int i=0; i<numProcs; i++)
      {
         part[i+1] += part[i];
      }

      glbSize = part[numProcs];
   }


   BlockVector EHa(NULL,bOffsets);
   BlockVector EHb(NULL,bOffsets);

   Vector k(3);
   Array2D<int> n;

   switch (lattice_type)
   {
      case 1:
      {
         n.SetSize(7,3);
         n[ 0][0] =  0; n[ 0][1] =  0; n[ 0][2] =  0;
         n[ 1][0] =  1; n[ 1][1] =  0; n[ 1][2] =  0;
         n[ 2][0] = -1; n[ 2][1] =  0; n[ 2][2] =  0;
         n[ 3][0] =  0; n[ 3][1] =  1; n[ 3][2] =  0;
         n[ 4][0] =  0; n[ 4][1] = -1; n[ 4][2] =  0;
         n[ 5][0] =  0; n[ 5][1] =  0; n[ 5][2] =  1;
         n[ 6][0] =  0; n[ 6][1] =  0; n[ 6][2] = -1;
      }
      break;
      case 2:
      {
         n.SetSize(13,3);
         n[ 0][0] =  0; n[ 0][1] =  0; n[ 0][2] =  0;

         n[ 1][0] =  0; n[ 1][1] =  1; n[ 1][2] =  1;
         n[ 2][0] =  0; n[ 2][1] =  1; n[ 2][2] = -1;
         n[ 3][0] =  0; n[ 3][1] = -1; n[ 3][2] =  1;
         n[ 4][0] =  0; n[ 4][1] = -1; n[ 4][2] = -1;

         n[ 5][0] =  1; n[ 5][1] =  0; n[ 5][2] =  1;
         n[ 6][0] =  1; n[ 6][1] =  0; n[ 6][2] = -1;
         n[ 7][0] = -1; n[ 7][1] =  0; n[ 7][2] =  1;
         n[ 8][0] = -1; n[ 8][1] =  0; n[ 8][2] = -1;

         n[ 9][0] =  1; n[ 9][1] =  1; n[ 9][2] =  0;
         n[10][0] =  1; n[10][1] = -1; n[10][2] =  0;
         n[11][0] = -1; n[11][1] =  1; n[11][2] =  0;
         n[12][0] = -1; n[12][1] = -1; n[12][2] =  0;
      }
      break;
      case 3:
      {
         // n.SetSize(9,3);
         n.SetSize(15,3);
         n[ 0][0] =  0; n[ 0][1] =  0; n[ 0][2] =  0;

         n[ 1][0] =  1; n[ 1][1] =  1; n[ 1][2] =  1;
         n[ 2][0] =  1; n[ 2][1] =  1; n[ 2][2] = -1;
         n[ 3][0] =  1; n[ 3][1] = -1; n[ 3][2] =  1;
         n[ 4][0] =  1; n[ 4][1] = -1; n[ 4][2] = -1;
         n[ 5][0] = -1; n[ 5][1] =  1; n[ 5][2] =  1;
         n[ 6][0] = -1; n[ 6][1] =  1; n[ 6][2] = -1;
         n[ 7][0] = -1; n[ 7][1] = -1; n[ 7][2] =  1;
         n[ 8][0] = -1; n[ 8][1] = -1; n[ 8][2] = -1;

         n[ 9][0] =  1; n[ 9][1] =  0; n[ 9][2] =  0;
         n[10][0] = -1; n[10][1] =  0; n[10][2] =  0;
         n[11][0] =  0; n[11][1] =  1; n[11][2] =  0;
         n[12][0] =  0; n[12][1] = -1; n[12][2] =  0;
         n[13][0] =  0; n[13][1] =  0; n[13][2] =  1;
         n[14][0] =  0; n[14][1] =  0; n[14][2] = -1;
      }
      break;
      default:
         cout << "Unsupported lattice type:  " << lattice_type << endl;
   };

   vector<int> e2n;
   vector<Vector> E0(3*n.NumRows());

   DenseMatrix Pkp(3);

   Vector eval(3);
   DenseMatrix evect(3);

   int c = 0;
   for (int i=0; i<n.NumRows(); i++)
   {
      k = kappa;
      for (int j=0; j<3; j++)
      {
         k.Add((double)n[i][j],b[j]);
      }

      // We need to find vectors orthogonal to k
      double normk = k.Norml2();

      // k should have a length of order 1
      if ( normk < 1e-2 )
      {
         // k is zero so pick three independent vectors
         E0[c].SetSize(3); E0[c][0] = 1.0; E0[c][1] = 0.0; E0[c][2] = 0.0; c++;
         E0[c].SetSize(3); E0[c][0] = 0.0; E0[c][1] = 1.0; E0[c][2] = 0.0; c++;
         E0[c].SetSize(3); E0[c][0] = 0.0; E0[c][1] = 0.0; E0[c][2] = 1.0; c++;
         e2n.push_back(i);
         e2n.push_back(i);
         e2n.push_back(i);
      }
      else
      {
         // Find two vectors orthogonal to k
         for (int j=0; j<3; j++)
         {
            Pkp(j,j) = 1.0;
            Pkp(j,(j+1)%3) = 0.0;
            Pkp(j,(j+2)%3) = 0.0;
            for (int l=0; l<3; l++)
            {
               Pkp(j,l) -= k[j]*k[l]/(normk*normk);
            }
         }
         Pkp.Eigensystem(eval,evect);

         // cout << "k vec "; k.Print(cout,3); cout << endl;
         // cout << "Evals "; eval.Print(cout,3); cout << endl;
         // evect.Print(cout,3);

         E0[c+0].SetSize(3);
         E0[c+1].SetSize(3);
         for (int j=0; j<3; j++)
         {
            E0[c+0][j] = evect(1,j);
            E0[c+1][j] = evect(2,j);
         }
         c+= 2;

         e2n.push_back(i);
         e2n.push_back(i);
      }
   }
   E0.resize(c);
   /*
   for (unsigned int i=0; i<E0.size(); i++)
   {
     cout << i << ":  ("
    << n[e2n[i]][0] << "," << n[e2n[i]][1] << "," << n[e2n[i]][2]
    << ") ";
     E0[i].Print(cout,3);
     cout << endl;
   }
   */
   /*
   Array2D<int> n(nev/2,3);
   vector<Vector> E0(nev/2);
   for (unsigned int i=0; i<E0.size(); i++)
   {
    E0[i].SetSize(3);
    E0[i] = 0.0;
   }

   switch (lattice_type) {
   case 1:
    {
      // First the 6 non-oscillatory modes (reused for real and imaginary)
      E0[ 0][0] = 1.0; n[ 0][0] =  0; n[ 0][1] =  0; n[ 0][2] =  0;
      E0[ 1][1] = 1.0; n[ 1][0] =  0; n[ 1][1] =  0; n[ 1][2] =  0;
      E0[ 2][2] = 1.0; n[ 2][0] =  0; n[ 2][1] =  0; n[ 2][2] =  0;
    }
    break;
   case 2:
    {
      // First the 6 non-oscillatory modes (reused for real and imaginary)
      E0[ 0][0] = 1.0; n[ 0][0] =  0; n[ 0][1] =  0; n[ 0][2] =  0;
      E0[ 1][1] = 1.0; n[ 1][0] =  0; n[ 1][1] =  0; n[ 1][2] =  0;
      E0[ 2][2] = 1.0; n[ 2][0] =  0; n[ 2][1] =  0; n[ 2][2] =  0;
    }
    break;
   case 3:
    {
      // First the 6 non-oscillatory modes (reused for real and imaginary)
      E0[ 0][0] = 1.0; n[ 0][0] =  0; n[ 0][1] =  0; n[ 0][2] =  0;
      E0[ 1][1] = 1.0; n[ 1][0] =  0; n[ 1][1] =  0; n[ 1][2] =  0;
      E0[ 2][2] = 1.0; n[ 2][0] =  0; n[ 2][1] =  0; n[ 2][2] =  0;
    }
    break;
   };

   for (unsigned int i=0; i<E0.size(); i++)
   {
    if ( init_vecs[2*i+0] == NULL )
      init_vecs[2*i+0] = new HypreParVector(comm,glbSize,part);
    if ( init_vecs[2*i+1] == NULL )
      init_vecs[2*i+1] = new HypreParVector(comm,glbSize,part);

    Ea.SetData(init_vecs[2*i+0]->GetData());
    Eb.SetData(init_vecs[2*i+1]->GetData());

    cosCoef.SetModeNumbers(n[i][0],n[i][1],n[i][2]);
    sinCoef.SetModeNumbers(n[i][0],n[i][1],n[i][2]);

    E0CosCoef.SetConstantVector(E0[i]);
    E0SinCoef.SetConstantVector(E0[i]);

    // Real Amplitude
    cosCoef.SetAmplitude( 1.0);
    sinCoef.SetAmplitude( 1.0);

    Er.ProjectCoefficient(E0CosCoef);
    Ei.ProjectCoefficient(E0SinCoef);

    Er.ParallelProject(Ea.GetBlock(0));
    Ei.ParallelProject(Ea.GetBlock(1));

    // Imaginary Amplitude
    cosCoef.SetAmplitude( 1.0);
    sinCoef.SetAmplitude(-1.0);

    Er.ProjectCoefficient(E0SinCoef);
    Ei.ProjectCoefficient(E0CosCoef);

    Er.ParallelProject(Eb.GetBlock(0));
    Ei.ParallelProject(Eb.GetBlock(1));
   }
   */
   // cout << "nev " << nev << ", size of E0 " << E0.size() << endl;
   nev = 2*E0.size();
   if ( (int)init_vecs.size() > nev )
   {
      for (unsigned int i=nev; i<init_vecs.size(); i++)
      {
         delete init_vecs[i];
      }
      init_vecs.resize(nev);
   }
   else
   {
      int nev0 = init_vecs.size();
      init_vecs.resize(nev);

      for (int i=nev0; i<nev; i++)
      {
         init_vecs[i] = new HypreParVector(comm,glbSize,part);
      }
   }

   for (unsigned int i=0; i<E0.size(); i++)
   {
      // Ea.SetData(init_vecs[2*i+0]->GetData());
      // Eb.SetData(init_vecs[2*i+1]->GetData());

      cosCoef.SetModeIndices(n[e2n[i]][0],n[e2n[i]][1],n[e2n[i]][2]);
      sinCoef.SetModeIndices(n[e2n[i]][0],n[e2n[i]][1],n[e2n[i]][2]);

      E0CosCoef.SetConstantVector(E0[i]);
      E0SinCoef.SetConstantVector(E0[i]);

      // Real Amplitude
      cosCoef.SetAmplitude( 1.0);
      sinCoef.SetAmplitude( 1.0);

      Er.ProjectCoefficient(E0CosCoef);
      Ei.ProjectCoefficient(E0SinCoef);

      // Er.ParallelProject(Ea.GetBlock(0));
      // Ei.ParallelProject(Ea.GetBlock(1));
      // Hr.ParallelProject(Ea.GetBlock(2));
      // Hi.ParallelProject(Ea.GetBlock(3));

      // Imaginary Amplitude
      cosCoef.SetAmplitude( 1.0);
      sinCoef.SetAmplitude(-1.0);

      Er.ProjectCoefficient(E0SinCoef);
      Ei.ProjectCoefficient(E0CosCoef);

      // Er.ParallelProject(Eb.GetBlock(0));
      // Ei.ParallelProject(Eb.GetBlock(1));
      // Hr.ParallelProject(Eb.GetBlock(2));
      // Hi.ParallelProject(Eb.GetBlock(3));
   }

}

void
WriteDispersionData(int myid,ostream & os, int c,
                    const string & label, vector<double> & eigenvalues)
{
   if ( myid == 0 )
   {
      os << c << "\t" << label;

      int i0 = 0;
      if ( label == "Gamma" ) { i0 = 2; }

      for (unsigned int i=i0; i<eigenvalues.size(); i++)
      {
         if ( eigenvalues[i] > 0.0 )
         {
            os << "\t" << sqrt(eigenvalues[i]);
         }
         else if ( eigenvalues[i] > -1.0e-6 )
         {
            os << "\t" << 0.0;
         }
         else
         {
            os << "\t" << -1.0;
         }
      }
      os << endl << flush;
   }
}

FourierVectorCoefficient::FourierVectorCoefficient()
{
   n_.resize(3); Ar_.SetSize(3); Ai_.SetSize(3);
}

complex<double>
FourierVectorCoefficient::operator*(const FourierVectorCoefficient & v) const
{
   complex<double> a = 0.0;

   const vector<int> & n0 = this->GetN();
   const vector<int> & n1 = v.GetN();

   if ( n0[0] == n1[0] && n0[1] == n1[1] && n0[2] == n1[2]  )
   {
      const Vector & Ar0 = this->GetAr();
      const Vector & Ai0 = this->GetAi();

      const Vector & Ar1 = v.GetAi();
      const Vector & Ai1 = v.GetAi();

      // a.real() += Ar0 * Ar1 + Ai0 * Ai1;
      // a.imag() += Ar0 * Ai1 - Ai0 * Ar1;
      a += complex<double>(Ar0 * Ar1 + Ai0 * Ai1, Ar0 * Ai1 - Ai0 * Ar1);
   }

   return a;
}

void
FourierVectorCoefficient::Print(ostream & os)
{
   os << "n = (" << n_[0] << "," << n_[1] << "," << n_[2] <<  "), "
      << "Ar = (" << Ar_[0] << "," << Ar_[1] << "," << Ar_[2] << "), "
      << "Ai = (" << Ai_[0] << "," << Ai_[1] << "," << Ai_[2] << ")";
}

FourierVectorCoefficients::FourierVectorCoefficients()
   : omega_(NAN)
{
}

FourierVectorCoefficients::~FourierVectorCoefficients()
{
   map<int,set<FourierVectorCoefficient*> >::iterator mit;
   set<FourierVectorCoefficient*>::iterator sit;
   for (mit=coefs_.begin(); mit!=coefs_.end(); mit++)
   {
      for (sit=mit->second.begin(); sit!=mit->second.end(); sit++)
      {
         delete *sit;
      }
   }
}

double
FourierVectorCoefficients::GetEnergy() const
{
   double e = 0.0;
   map<int,set<FourierVectorCoefficient*> >::const_iterator mit;
   for (mit=coefs_.begin(); mit!=coefs_.end(); mit++)
   {
      e += this->GetEnergy(mit->first);
   }
   return e;
}

double
FourierVectorCoefficients::GetEnergy(int tier) const
{
   double e = 0.0;

   if ( coefs_.find(tier) != coefs_.end() )
   {
      const set<FourierVectorCoefficient*> tierCoefs = coefs_.find(tier)->second;
      set<FourierVectorCoefficient*>::const_iterator sit;
      for (sit=tierCoefs.begin(); sit!=tierCoefs.end(); sit++)
      {
         e += (*sit)->GetEnergy();
      }
   }
   return e;
}

void
FourierVectorCoefficients::AddCoefficient(int n0, int n1, int n2,
                                          const Vector & Ar,
                                          const Vector & Ai)
{
   int tier = n0 * n0 + n1 * n1 + n2 * n2;

   FourierVectorCoefficient * fc = new FourierVectorCoefficient;
   fc->SetN(n0,n1,n2);
   fc->SetA(Ar, Ai);

   coefs_[tier].insert(fc);
}

void
FourierVectorCoefficients::GetCoefficient(int n0, int n1, int n2,
                                          Vector & Ar,
                                          Vector & Ai) const
{
   Ar.SetSize(3); Ai.SetSize(3);

   const FourierVectorCoefficient * fvc = this->GetCoefficient(n0, n1, n2);

   if ( fvc != NULL )
   {
      Ar = fvc->GetAr();
      Ai = fvc->GetAi();
   }
   else
   {
      Ar = 0.0; Ai = 0.0;
   }
}

const FourierVectorCoefficient *
FourierVectorCoefficients::GetCoefficient(int n0, int n1, int n2) const
{
   int tier = n0 * n0 + n1 * n1 + n2 * n2;

   map<int,set<FourierVectorCoefficient*> >::const_iterator mit;

   mit = coefs_.find(tier);
   if ( mit == coefs_.end() )
   {
      return NULL;
   }

   set<FourierVectorCoefficient*>::const_iterator sit;

   for (sit=mit->second.begin(); sit!=mit->second.end(); sit++)
   {
      const vector<int> & n = (*sit)->GetN();
      if ( n[0] == n0 && n[1] == n1 && n2 == n[2] )
      {
         return *sit;
      }
   }

   return NULL;
}

complex<double>
FourierVectorCoefficients::operator*(const FourierVectorCoefficients & v) const
{
   complex<double> a = 0.0;

   map<int,set<FourierVectorCoefficient*> >::const_iterator mit;
   set<FourierVectorCoefficient*>::const_iterator sit;
   for (mit=coefs_.begin(); mit!=coefs_.end(); mit++)
   {
      for (sit=mit->second.begin(); sit!=mit->second.end(); sit++)
      {
         const vector<int> & n = (*sit)->GetN();

         const FourierVectorCoefficient * fvc0 = *sit;
         const FourierVectorCoefficient * fvc1 = v.GetCoefficient(n[0],n[1],n[2]);

         if ( fvc1 != NULL )
         {
            a += (*fvc0) * (*fvc1);
         }
      }
   }

   return a;
}

void
FourierVectorCoefficients::Print(ostream & os)
{
   map<int,set<FourierVectorCoefficient*> >::iterator mit;
   set<FourierVectorCoefficient*>::iterator sit;
   os << "Omega:  " << omega_ << endl;

   double en = this->GetEnergy();
   os << "Total Energy:     " << en << endl;
   for (mit=coefs_.begin(); mit!=coefs_.end(); mit++)
   {
      os << "Energy Fraction:  " << this->GetEnergy(mit->first)/en << endl;
      for (sit=mit->second.begin(); sit!=mit->second.end(); sit++)
      {
         (*sit)->Print(os); os << endl;
      }
      os << endl;
   }
}

void ComputeFourierCoefficients(int myid, ostream & ofs_coef, int nmax,
                                HCurlFourierSeries & fourier,
                                MaxwellBlochWaveEquationAMR & eq,
                                ParFiniteElementSpace & HCurlFESpace,
                                const vector<double> & eigenvalues,
                                map<int,FourierVectorCoefficients> & mfc)
{
   cout << "Computing Fourier coefficients" << endl;

   HypreParVector  Er(HCurlFESpace.GetComm(),
                      HCurlFESpace.GlobalTrueVSize(),
                      NULL,
                      HCurlFESpace.GetTrueDofOffsets());

   HypreParVector  Ei(HCurlFESpace.GetComm(),
                      HCurlFESpace.GlobalTrueVSize(),
                      NULL,
                      HCurlFESpace.GetTrueDofOffsets());

   double tol = 1e-6;
   Vector Err(3), Eri(3), Eir(3), Eii(3), Ar(3), Ai(3);

   for (int i=0; i<=nmax; i++)
   {
      for (int j=(i==0)?0:-nmax; j<=nmax; j++)
      {
         for (int k=(i==0&&j==0)?0:-nmax; k<=nmax; k++)
         {
            fourier.SetMode(i, j, k);
            for (unsigned int l=0; l<eigenvalues.size(); l++)
            {
               eq.GetEigenvectorE(l, Er, Ei);

               fourier.GetCoefficient(Er, Err, Eri);
               fourier.GetCoefficient(Ei, Eir, Eii);

               if ( myid == 0 )
               {
                  double nrmErr = Err.Normlinf();
                  double nrmEri = Eri.Normlinf();
                  double nrmEir = Eir.Normlinf();
                  double nrmEii = Eii.Normlinf();

                  if ( nrmErr > tol || nrmEri > tol || nrmEir > tol || nrmEii > tol )
                  {
                     if ( mfc.find(l) == mfc.end() )
                     {
                        mfc[l].SetOmega(sqrt(eigenvalues[l]));
                     }
                     Ar = Err; Ar -= Eii;
                     Ai = Eir; Ai += Eri;
                     mfc[l].AddCoefficient(i,j,k,Ar,Ai);

                     if ( i != 0 || j != 0 || k != 0 )
                     {
                        Ar = Err; Ar += Eii;
                        Ai = Eir; Ai -= Eri;
                        mfc[l].AddCoefficient(-i,-j,-k,Ar,Ai);
                     }
                  }
               }
            }
         }
      }
   }
   cout << "Done computing Fourier coefficients" << endl << flush;
}

void
CompareFourierCoefficients(
   map<int, map<int, FourierVectorCoefficients> > & mfc)
{
   ofstream ofs("P.mat");

   map<int, map<int, FourierVectorCoefficients> >::iterator mmit0;
   map<int, map<int, FourierVectorCoefficients> >::iterator mmit1;
   for ( mmit0=mfc.begin(), mmit1=mfc.begin(), mmit1++;
         mmit1!=mfc.end();
         mmit0++, mmit1++)
   {
      int w0 = mmit0->second.rbegin()->first;
      int w1 = mmit1->second.rbegin()->first;
      DenseMatrix P(w0+1,w1+1);

      for (int i=0; i<=w0; i++)
      {
         for (int j=0; j<=w1; j++)
         {
            complex<double> a = 0.0;
            if ( mmit0->second.find(i) != mmit0->second.end() &&
                 mmit1->second.find(j) != mmit1->second.end() )
            {
               a = mmit0->second[i] * mmit1->second[j];
               P(i,j) = std::abs(a);
            }
            else
            {
               P(i,j)= 0.0;
            }
         }
      }
      ofs << mmit1->first << " -> " << mmit0->first << endl;
      P.Print(ofs);
      ofs << endl;
   }
   ofs.close();
}

void IdentifyDegeneracies(const vector<double> & eigenvalues,
                          double zero_tol, double rel_tol,
                          vector<set<int> > & degen)
{
   cout << "Identifying degeneracies" << endl;

   // Assume no degeneracies
   degen.resize(eigenvalues.size());

   // No eigenvalues means no degeneracies
   if ( eigenvalues.size() == 0 )
   {
      return;
   }

   // Place the first eigenvalue in the first set
   int nd = 0;
   degen[nd].insert(0);

   // Switch to select between zero_tol and rel_tol
   bool zeroes = eigenvalues[0] < zero_tol;

   // Scan the eigenvalues
   for (unsigned int i=1; i<eigenvalues.size(); i++)
   {
      if ( zeroes )
      {
         // The previous eigenvalue was a zero
         if ( eigenvalues[i] > zero_tol )
         {
            // This eigenvalue is not a zero
            nd++;
            zeroes = false;
         }
         // Place this eigenvalue in the appropriate grouping
         degen[nd].insert(i);
      }
      else
      {
         // The previous eigenvalue was non-zero
         if ( fabs( eigenvalues[i] - eigenvalues[i-1] ) >
              ( eigenvalues[i] + eigenvalues[i-1] ) * 0.5 * rel_tol )
         {
            // This eigenvalue belongs to a new grouping
            nd++;
         }
         // Place this eigenvalue in the appropriate grouping
         degen[nd].insert(i);
      }
   }

   // Adjust size down to the number of degeneracies identified
   degen.resize( nd + 1 );

   cout << "Done identifying degeneracies" << endl << flush;
}

void BuildInitialMesh(Mesh & emesh, BravaisLattice & bravais, int nr, double h)
{
   int    v[4];
   Vector x0(3); x0 = 0.0;
   Vector x1(3); x1 = 0.0;

   double d = 0.25;

   x0[2] = 0.0;
   x1[1] = -d;
   x1[2] =  h;

   int vi = 0;
   int vtx = 0;
   int elm = 0;

   for (unsigned int p=0; p<bravais.GetNumberPaths(); p++)
   {
      for (int i=0; i<nr; i++)
      {
         emesh.AddVertex(x0);
         emesh.AddVertex(x1);
         vtx += 2;
      }
      vi++;

      for (unsigned int s=0; s<bravais.GetNumberPathSegments(p); s++)
      {
         x0[0] += 0.5;
         x1[0] += 0.5;

         for (int i=0; i<nr; i++)
         {
            emesh.AddVertex(x0);
            emesh.AddVertex(x1);
            vtx += 2;

            v[0] = 2*i + 2*nr*(vi-1);
            v[1] = 2*i + 2*nr*vi;
            v[2] = 2*i + 2*nr*vi+1;
            v[3] = 2*i + 2*nr*(vi-1) + 1;

            emesh.AddQuad(v);
            elm++;

         }
         vi++;

         x0[0] += 0.5;
         x1[0] += 0.5;
         for (int i=0; i<nr; i++)
         {
            emesh.AddVertex(x0);
            emesh.AddVertex(x1);
            vtx += 2;

            v[0] = 2*i + 2*nr*(vi-1);
            v[1] = 2*i + 2*nr*vi;
            v[2] = 2*i + 2*nr*vi+1;
            v[3] = 2*i + 2*nr*(vi-1) + 1;

            emesh.AddQuad(v);
            elm++;
         }
         vi++;
      }
   }
   emesh.FinalizeQuadMesh(1);
}

void RescaleEigMesh(Mesh & emesh,
                    double & scaleFactor, double maxEig, double height,
                    bool firstEigs)
{
   cout << "Checking if rescaling is necessary" << endl;
   if ( firstEigs && sqrt(fabs(maxEig)) < height )
   {
      cout << "Scaling upwards" << endl;
      scaleFactor = height / sqrt(fabs(maxEig));
   }
   else if ( scaleFactor * sqrt(fabs(maxEig)) > height )
   {
      cout << "Scaling downwards" << endl;
      double prev_sf = scaleFactor;
      scaleFactor = height / sqrt(fabs(maxEig));

      for (int i=0; i<emesh.GetNV(); i++)
      {
         double * vc = emesh.GetVertex(i);
         vc[2] *= scaleFactor / prev_sf;
      }
   }
   cout << "Leaving rescaling" << endl << flush;
}

void VisualizeMesh(MPI_Comm & comm, int myid, int num_procs,
                   const string & title, Mesh & emesh, socketstream & sock)
{
   cout << "Building ParMesh" << endl;

   int * epart = new int[emesh.GetNE()];
   for (int i=0; i<emesh.GetNE(); i++) { epart[i] = 0; }

   ParMesh pemesh(comm, emesh, epart);
   delete epart;

   L2_FECollection fec(0, 2);
   ParFiniteElementSpace fespace(&pemesh, &fec);
   ParGridFunction color(&fespace);
   color = 0.0;

   cout << "Sending data to socket" << endl;
   sock << "parallel " << num_procs << " " << myid << "\n"
        << "solution\n" << pemesh << color
        << "window_title '" << title << "'\n"
        << flush;
   cout << "Data sent to socket" << endl << flush;
}

double mass_coef(const Vector & x)
{
   double eps1 = 10.0;
   double eps2 = 100.0;

   switch ( prob_ )
   {
      case 0:
         // Slab
         if ( fabs(x(0)) <= 0.5 ) { return eps1; }
         break;
      case 1:
         // Cylinder
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= 0.5 ) { return eps1; }
         break;
      case 2:
         // Sphere
         if ( x.Norml2() <= 0.25 ) { return eps1; }
         break;
      case 3:
         // Spherical Shell and 3 Rods
      {
         double r1 = 0.14, r2 = 0.36, r3 = 0.105;
         double eps0 = 1.0, eps3 = 12.96;
         if ( x.Norml2() <= r1 ) { return eps0; }
         if ( x.Norml2() <= r2 ) { return eps3; }
         if ( sqrt(x(1)*x(1)+x(2)*x(2)) <= r3 ) { return eps3; }
         if ( sqrt(x(2)*x(2)+x(0)*x(0)) <= r3 ) { return eps3; }
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= r3 ) { return eps3; }
      }
      break;
      case 4:
         // Spherical Shell and 4 Rods
      {
         double r1 = 0.14, r2 = 0.28, r3 = 0.1;
         double eps0 = 1.0, eps3 = 12.96;
         if ( x.Norml2() <= r1 ) { return eps0; }
         if ( x.Norml2() <= r2 ) { return eps3; }
         {
            double rr = x(0)*x(0)+x(1)*x(1)+x(2)*x(2);
            if ( sqrt(rr-x(0)*x(1)-x(1)*x(2)-x(2)*x(0)) <= r3 ) { return eps3; }
            if ( sqrt(rr+x(0)*x(1)+x(1)*x(2)-x(2)*x(0)) <= r3 ) { return eps3; }
            if ( sqrt(rr+x(0)*x(1)-x(1)*x(2)+x(2)*x(0)) <= r3 ) { return eps3; }
            if ( sqrt(rr-x(0)*x(1)+x(1)*x(2)+x(2)*x(0)) <= r3 ) { return eps3; }
         }
      }
      break;
      case 5:
         // Two spheres in a BCC configuration
         if ( x.Norml2() <= 0.99 ) { return eps2; }
         {
            for (int i=0; i<8; i++)
            {
               int i1 = i%2;
               int i2 = (i/2)%2;
               int i4 = i/4;

               Vector u = x;
               u(0) -= i1?-1.0:1.0;
               u(1) -= i2?-1.0:1.0;
               u(2) -= i4?-1.0:1.0;

               if ( u.Norml2() <= 0.74 ) { return eps2; }
            }
         }
      case 6:
         // Spherical Shell and 6 Rods
      {
         double r1 = 0.12, r2 = 0.19, r3 = 0.08;
         double eps0 = 1.0, eps3 = 12.96;
         if ( x.Norml2() <= r1 ) { return eps0; }
         if ( x.Norml2() <= r2 ) { return eps3; }
         {
            double rr = x(0)*x(0)+x(1)*x(1)+x(2)*x(2);
            if ( sqrt(rr+x(0)*x(0)-2.0*x(1)*x(2)) <= r3 ) { return eps3; }
            if ( sqrt(rr+x(0)*x(0)+2.0*x(1)*x(2)) <= r3 ) { return eps3; }
            if ( sqrt(rr+x(1)*x(1)-2.0*x(2)*x(0)) <= r3 ) { return eps3; }
            if ( sqrt(rr+x(1)*x(1)+2.0*x(2)*x(0)) <= r3 ) { return eps3; }
            if ( sqrt(rr+x(2)*x(2)-2.0*x(0)*x(1)) <= r3 ) { return eps3; }
            if ( sqrt(rr+x(2)*x(2)+2.0*x(0)*x(1)) <= r3 ) { return eps3; }
         }
      }
      break;
      case 7:
         // Doubly Periodic array of square, air holes of infinite depth
         // From Mias/Webb/Ferrari Paper
         if ( fabs(x(0)) <= 0.25 && fabs(x(1)) <= 0.25 )
         {
            return 1.0;
         }
         return 13.0;
         break;
      case 8:
         if ( fabs(x(0)) < 0.1 || fabs(x(1)) < 0.1 || fabs(x(2)) < 0.1 )
         {
            return eps2;
         }
         break;
      default:
         return 1.0;
   }
   return 1.0;
}

double stiffness_coef(const Vector &x)
{
   return 1.0;
}

double phi(const Vector & x)
{
   return cos(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(2));
}

void Phi(const Vector & x, Vector & v)
{
   double a = 10.0*M_PI/180.0;
   double i = 15.0*M_PI/180.0;
   v(0) = cos(i)*cos(a); v(1) = cos(i)*sin(a); v(2) = sin(i);
   v *= phi(x);
}
