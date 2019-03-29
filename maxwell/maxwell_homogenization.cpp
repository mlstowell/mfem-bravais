#include "mfem.hpp"
#include "maxwell_bloch.hpp"
#include "../common/bravais.hpp"
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

int CreateDirectory(const string &dir_name, MPI_Comm & comm, int myid);

void CreateInitialVectors(int lattice_type,
                          const BravaisLattice & bravais,
                          const Vector & kappa,
                          ParFiniteElementSpace & fespace,
                          int & nev,
                          vector<HypreParVector*> & init_vecs);

class FourierVectorCoefficient
{
public:
   FourierVectorCoefficient(const string & label = "a")
      : str_(label)
   { n_.resize(3); Ar_.SetSize(3); Ai_.SetSize(3); }

   void SetLabel(const string & label) { str_ = label; }
   void SetN(int n0, int n1, int n2) { n_[0] = n0; n_[1] = n1; n_[2] = n2; }
   void SetA(const Vector & Ar, const Vector & Ai) { Ar_ = Ar; Ai_ = Ai; }

   const vector<int> & GetN()  const { return n_; }
   const Vector      & GetAr() const { return Ar_; }
   const Vector      & GetAi() const { return Ai_; }

   double GetEnergy() const { return Ar_ * Ar_ + Ai_ * Ai_; }

   void Print(ostream & os)
   {
      os << "n = (" << n_[0] << "," << n_[1] << "," << n_[2] <<  "), "
         << str_ << "r = (" << Ar_[0] << "," << Ar_[1] << "," << Ar_[2] << "), "
         << str_ << "i = (" << Ai_[0] << "," << Ai_[1] << "," << Ai_[2] << ")";
   }

   void PrintMathematica(ostream & os)
   {
      os << "{"
         << Ar_[0] << "+" << Ai_[0] << "\[ImaginaryI],"
         << Ar_[1] << "+" << Ai_[1] << "\[ImaginaryI],"
         << Ar_[2] << "+" << Ai_[2] << "\[ImaginaryI]}, ";
   }

private:
   string str_;
   vector<int> n_;
   Vector Ar_;
   Vector Ai_;
};

class FourierVectorCoefficients
{
public:
   FourierVectorCoefficients(const string & label_a = "a",
                             const string & label_b = "b",
                             const string & label_c = "c",
                             const string & label_d = "d")
      : str0_(label_a), str1_(label_b), str2_(label_c), str3_(label_d),
        omega_(NAN) {}
   ~FourierVectorCoefficients()
   {
      map<int,set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> > >::iterator mit;
      set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> >::iterator sit;
      for (mit=coefs_.begin(); mit!=coefs_.end(); mit++)
      {
         for (sit=mit->second.begin(); sit!=mit->second.end(); sit++)
         {
            delete sit->first;
            delete sit->second;
         }
      }
      for (mit=dualCoefs_.begin(); mit!=dualCoefs_.end(); mit++)
      {
         for (sit=mit->second.begin(); sit!=mit->second.end(); sit++)
         {
            delete sit->first;
            delete sit->second;
         }
      }
   }

   void   SetLabels(const string & label_a, const string & label_b)
   { str0_ = label_a; str1_ = label_b; }
   void   SetDualLabels(const string & label_a, const string & label_b)
   { str2_ = label_a; str3_ = label_b; }
   void   SetOmega( double omega) { omega_ = omega; }
   double GetOmega() const        { return omega_; }
   double GetEnergy() const
   {
      double e = 0.0;
      map<int,set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> > >::const_iterator mit;
      for (mit=coefs_.begin(); mit!=coefs_.end(); mit++)
      {
         e += this->GetEnergy(mit->first);
      }
      return e;
   }

   double GetEnergy(int tier) const
   {
      double e = 0.0;

      if ( coefs_.find(tier) != coefs_.end() )
      {
         const set<pair<FourierVectorCoefficient*,
               FourierVectorCoefficient*> > tierCoefs =
                  coefs_.find(tier)->second;
         set<pair<FourierVectorCoefficient*,
             FourierVectorCoefficient*> >::const_iterator sit;
         for (sit=tierCoefs.begin(); sit!=tierCoefs.end(); sit++)
         {
            e += sit->first->GetEnergy() + sit->second->GetEnergy();
         }
      }
      return e;
   }

   void AddCoefficients(int n0, int n1, int n2,
                        const Vector & Ar, const Vector & Ai,
                        const Vector & Br, const Vector & Bi)
   {
      int tier = n0 * n0 + n1 * n1 + n2 * n2;

      FourierVectorCoefficient * fcA = new FourierVectorCoefficient(str0_);
      fcA->SetN(n0,n1,n2);
      fcA->SetA(Ar, Ai);

      FourierVectorCoefficient * fcB = new FourierVectorCoefficient(str1_);
      fcB->SetN(n0,n1,n2);
      fcB->SetA(Br, Bi);

      coefs_[tier].insert(make_pair<FourierVectorCoefficient*,
                          FourierVectorCoefficient*>(fcA,fcB));
   }

   void AddDualCoefficients(int n0, int n1, int n2,
                            const Vector & Ar, const Vector & Ai,
                            const Vector & Br, const Vector & Bi)
   {
      int tier = n0 * n0 + n1 * n1 + n2 * n2;

      FourierVectorCoefficient * fcA = new FourierVectorCoefficient(str2_);
      fcA->SetN(n0,n1,n2);
      fcA->SetA(Ar, Ai);

      FourierVectorCoefficient * fcB = new FourierVectorCoefficient(str3_);
      fcB->SetN(n0,n1,n2);
      fcB->SetA(Br, Bi);

      dualCoefs_[tier].insert(make_pair<FourierVectorCoefficient*,
                              FourierVectorCoefficient*>(fcA,fcB));
   }

   void GetCoefficients(int n0, int n1, int n2,
                        Vector & Ar, Vector & Ai,
                        Vector & Br, Vector & Bi) const
   {
      int tier = n0 * n0 + n1 * n1 + n2 * n2;

      map<int,set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> > >::const_iterator
          mit = coefs_.find(tier);

      set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> >::const_iterator
          sit = mit->second.begin();

      Ar = sit->first->GetAr();
      Ai = sit->first->GetAi();

      Br = sit->second->GetAr();
      Bi = sit->second->GetAi();
   }

   void GetDualCoefficients(int n0, int n1, int n2,
                            Vector & Ar, Vector & Ai,
                            Vector & Br, Vector & Bi) const
   {
      int tier = n0 * n0 + n1 * n1 + n2 * n2;

      map<int,set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> > >::const_iterator
          mit = dualCoefs_.find(tier);

      set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> >::const_iterator
          sit = mit->second.begin();

      Ar = sit->first->GetAr();
      Ai = sit->first->GetAi();

      Br = sit->second->GetAr();
      Bi = sit->second->GetAi();
   }

   void Print(ostream & os)
   {
      map<int,set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> > >::iterator mit;
      set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> >::iterator sit;
      set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> >::iterator dualSit;
      os << "Omega:  " << omega_ << endl;

      double en = this->GetEnergy();
      os << "Total Energy:     " << en << endl;
      for (mit=coefs_.begin(); mit!=coefs_.end(); mit++)
      {
         os << "Energy Fraction:  " << this->GetEnergy(mit->first)/en << endl;
         for (sit=mit->second.begin(); sit!=mit->second.end(); sit++)
         {
            sit->first->Print(os); os << endl;
            sit->second->Print(os); os << endl;
         }
         int tier = mit->first;
         for (dualSit=dualCoefs_[tier].begin();
              dualSit!=dualCoefs_[tier].end(); dualSit++)
         {
            dualSit->first->Print(os); os << endl;
            dualSit->second->Print(os); os << endl;
         }
         os << endl;
      }
   }

   void PrintMathematica(ostream & os)
   {
      map<int,set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> > >::iterator mit;
      set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> >::iterator sit;
      set<pair<FourierVectorCoefficient*,
          FourierVectorCoefficient*> >::iterator dualSit;

      for (mit=coefs_.begin(); mit!=coefs_.end(); mit++)
      {
         for (sit=mit->second.begin(); sit!=mit->second.end(); sit++)
         {
            sit->first->PrintMathematica(os); os << endl;
            sit->second->PrintMathematica(os); os << endl;
         }
         int tier = mit->first;
         for (dualSit=dualCoefs_[tier].begin();
              dualSit!=dualCoefs_[tier].end(); dualSit++)
         {
            dualSit->first->PrintMathematica(os); os << endl;
            dualSit->second->PrintMathematica(os); os << endl;
         }
         os << endl;
      }
   }

private:
   string str0_;
   string str1_;
   string str2_;
   string str3_;
   double omega_;

   map<int,set<pair<FourierVectorCoefficient*,
       FourierVectorCoefficient*> > > coefs_;
   map<int,set<pair<FourierVectorCoefficient*,
       FourierVectorCoefficient*> > > dualCoefs_;
};

void ComputeFourierCoefficients(int myid, ostream & ofs_coef, int nmax,
                                HCurlFourierSeries & fourierHCurl,
                                HDivFourierSeries & fourierHDiv,
                                MaxwellBlochWaveEquation & eq,
                                ParFiniteElementSpace & HCurlFESpace,
                                ParFiniteElementSpace & HDivFESpace,
                                const vector<double> & eigenvalues,
                                map<int,FourierVectorCoefficients> & mfc);
/*
void IdentifyDegeneracies(const vector<double> & eigenvalues,
                          double zero_tol, double rel_tol,
                          vector<set<int> > & degen);
*/
void CalcCoefs(map<int,FourierVectorCoefficients> & m);

void WriteWaveVectors(ostream & os,
                      const Vector & Er, const Vector & Ei,
                      const Vector & k, double omega);

// Material Coefficients
static int prob_ = -1;
double mass_coef(const Vector &);
double stiffness_coef(const Vector &);

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
   const char *sym_pt_ptr = "X";
   string sym_pt = "X";
   int order = 1;
   int sr = 0, pr = 2;
   bool visualization = 1;
   bool visit = true;
   int nev = 0;
   int num_beta = 10;
   int num_a_per_lambda = 10;
   double a = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&lattice_type, "-bl", "--bravais-lattice",
                  "Bravais Lattice Type: "
                  " 1 - Primitive Cubic,"
                  " 2 - Body-Centered Cubic,"
                  " 3 - Face-Centered Cubic");
   args.AddOption(&sym_pt_ptr, "-sp", "--symmetry-point","");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sr, "-sr", "--serial-refinement",
                  "Number of serial refinement levels.");
   args.AddOption(&pr, "-pr", "--parallel-refinement",
                  "Number of parallel refinement levels.");
   args.AddOption(&prob_, "-p", "--problem-type",
                  "Problem Geometry.");
   args.AddOption(&num_beta, "-nb", "--number-of-wave-vectors",
                  "Number of Wave Vectors");
   args.AddOption(&a, "-a", "--lattice-size",
                  "Lattice Size");
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

   sym_pt = sym_pt_ptr;

   if (myid == 0)
   {
      cout << "Creating symmetry points for lattice " << lattice_type << endl;
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
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = pr;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   ND_ParFESpace * HCurlFESpace = new ND_ParFESpace(pmesh, order,
                                                    pmesh->Dimension());
   RT_ParFESpace * HDivFESpace  = new RT_ParFESpace(pmesh, order,
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

   if (visualization)
   {
      socketstream m_sock, k_sock;
      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      //int offx = Ww+10, offy = Wh+45; // window offsets
      int offy = Wh+45; // window offsets

      VisualizeField(m_sock, vishost, visport, *m,
                     "Mass Coefficient", Wx, Wy, Ww, Wh);
      VisualizeField(k_sock, vishost, visport, *k,
                     "Stiffness Coefficient", Wx, Wy+offy, Ww, Wh);
   }

   GridFunctionCoefficient mCoef(m);
   // GridFunctionCoefficient kCoef(k);
   ConstantCoefficient kCoef(1.0);

   MaxwellBlochWaveEquation * eq =
      new MaxwellBlochWaveEquation(*pmesh, order);

   HCurlFourierSeries fourier_hcurl(*bravais, *HCurlFESpace);
   HDivFourierSeries  fourier_hdiv(*bravais, *HDivFESpace);

   HYPRE_Int size = eq->GetHCurlFESpace()->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of complex unknowns: " << size << endl;
   }

   // eq->SetAbsoluteTolerance( 1.0e-6 / (MAXWELL_MU0 * MAXWELL_EPS0) );
   eq->SetMassCoef(mCoef);
   eq->SetStiffnessCoef(kCoef);
   eq->SetBravaisLattice(*bravais);

   ostringstream oss_prefix;
   oss_prefix << "Maxwell-Homogenization-" << lattice_label;

   CreateDirectory(oss_prefix.str(),comm,myid);

   vector<HypreParVector*> init_vecs;

   vector<Vector> lattice_vecs;
   bravais->GetLatticeVectors(lattice_vecs);

   Vector kappa0(3), kappa(3);
   int si = bravais->GetSymmetryPointIndex(sym_pt);
   bravais->GetSymmetryPoint(si,kappa0);
   string label = bravais->GetSymmetryPointLabel(si);

   kappa0 /= kappa0.Norml2();

   ostringstream oss_coef, oss_dat, oss_avg;
   oss_coef << oss_prefix.str() << "/coef-" << label << ".dat";
   oss_dat << oss_prefix.str() << "/homogenization-" << label << ".dat";
   oss_avg << oss_prefix.str() << "/averages-" << label << ".dat";

   ofstream ofs_coef, ofs_dat, ofs_avg;

   if ( myid == 0 )
   {
      ofs_coef.open(oss_coef.str().c_str());
      ofs_dat.open(oss_dat.str().c_str());
      ofs_avg.open(oss_avg.str().c_str());
   }

   vector<double> eigenvalues;
   map<int, vector<set<int> > > degen;
   map<int, map<int, FourierVectorCoefficients> > mfc;

   int c = 1;

   for (int i=1; i < num_beta; i++)
   {
      kappa = kappa0;
      double frac = (num_beta>1)?(double)i/(num_beta-1):1.0;
      kappa *= 2.0 * M_PI * frac / ( a * num_a_per_lambda );

      CreateInitialVectors(lattice_type, *bravais, kappa,
                           *eq->GetHCurlFESpace(),
                           nev, init_vecs);

      eq->GetEigenvalues(nev, kappa, init_vecs, eigenvalues);

      eq->IdentifyDegeneracies(1.0e-4, 1.0e-4, degen[c]);
      // IdentifyDegeneracies(eigenvalues, 1.0e-4, 1.0e-4, degen[c]);
      /*
      ComputeFourierCoefficients(myid, ofs_coef, 0,
                                 fourier_hcurl, fourier_hdiv, *eq,
                                 *HCurlFESpace, *HDivFESpace,
                                 eigenvalues,
                                 mfc[c]);
      */
      Vector Er, Ei, Br, Bi, Dr, Di, Hr, Hi;
      for (unsigned int l=0; l<eigenvalues.size(); l++)
      {
         eq->GetFieldAverages(l, Er, Ei, Br, Bi, Dr, Di, Hr, Hi);
         mfc[c][l].SetOmega(sqrt(fabs(eigenvalues[l])));
         mfc[c][l].SetLabels("E", "B");
         mfc[c][l].SetDualLabels("D", "H");
         mfc[c][l].AddCoefficients(0,0,0,Er,Ei,Br,Bi);
         mfc[c][l].AddDualCoefficients(0,0,0,Dr,Di,Hr,Hi);
      }

      CalcCoefs(mfc[c]);

      c++;
      /*
      if (visualization)
      {
         socketstream Er_sock, Ei_sock, Br_sock, Bi_sock;
         char vishost[] = "localhost";
         int  visport   = 19916;

         int Wx = 0, Wy = 0; // window position
         int Ww = 350, Wh = 350; // window size
         int offx = Ww+10, offy = Wh+45; // window offsets

         ParGridFunction Er(eq->GetHCurlFESpace());
         ParGridFunction Ei(eq->GetHCurlFESpace());

         ParGridFunction Br(eq->GetHDivFESpace());
         ParGridFunction Bi(eq->GetHDivFESpace());

         HypreParVector ErVec(eq->GetHCurlFESpace()->GetComm(),
                              eq->GetHCurlFESpace()->GlobalTrueVSize(),
                              NULL,
                              eq->GetHCurlFESpace()->GetTrueDofOffsets());
         HypreParVector EiVec(eq->GetHCurlFESpace()->GetComm(),
                              eq->GetHCurlFESpace()->GlobalTrueVSize(),
                              NULL,
                              eq->GetHCurlFESpace()->GetTrueDofOffsets());

         HypreParVector BrVec(eq->GetHDivFESpace()->GetComm(),
                              eq->GetHDivFESpace()->GlobalTrueVSize(),
                              NULL,
                              eq->GetHDivFESpace()->GetTrueDofOffsets());
         HypreParVector BiVec(eq->GetHDivFESpace()->GetComm(),
                              eq->GetHDivFESpace()->GlobalTrueVSize(),
                              NULL,
                              eq->GetHDivFESpace()->GetTrueDofOffsets());

         for (int e=0; e<nev; e++)
         {
            if ( myid == 0 )
            {
               cout << "Beta: " << beta << ", mode: ";
               cout << e << ":  Eigenvalue " << eigenvalues[e];
               double trans_eig = eigenvalues[e];
               if ( trans_eig > 0.0 )
               {
                  cout << ", omega " << sqrt(trans_eig);
               }
               cout << endl;
            }

            eq->GetEigenvector(e, ErVec, EiVec, BrVec, BiVec);

            Er = ErVec;
            Ei = EiVec;

            Br = BrVec;
            Bi = BiVec;

            VisualizeField(Er_sock, vishost, visport, Er,
                           "Re(E)", Wx+offx, Wy, Ww, Wh);
            VisualizeField(Ei_sock, vishost, visport, Ei,
                           "Im(E)", Wx+offx, Wy+offy, Ww, Wh);

            VisualizeField(Br_sock, vishost, visport, Br,
                           "Re(B)", Wx+2*offx, Wy, Ww, Wh);
            VisualizeField(Bi_sock, vishost, visport, Bi,
                           "Im(B)", Wx+2*offx, Wy+offy, Ww, Wh);

            char c;
            if (myid == 0)
            {
               cout << "press (q)uit or (c)ontinue --> " << flush;
               cin >> c;
            }
            MPI_Bcast(&c, 1, MPI_CHAR, 0, comm);

            if (c != 'c')
            {
               break;
            }
         }

         Er_sock.close();
         Ei_sock.close();
         Br_sock.close();
         Bi_sock.close();
      }
      */
      if ( myid == 0 )
      {
         ofs_dat << i;
         for (unsigned int j=0; j<eigenvalues.size(); j++)
         {
            ofs_dat << "\t" << eigenvalues[j];
         }
         ofs_dat << endl;
      }
   }
   if ( visit ) { eq->WriteVisitFields(oss_prefix.str(),label); }

   map<int, map<int,FourierVectorCoefficients> >::iterator mmit;
   map<int,FourierVectorCoefficients>::iterator mit;
   set<int>::iterator sit;
   for (mmit=mfc.begin(); mmit!=mfc.end(); mmit++)
   {
      kappa = kappa0;
      double frac = (num_beta>1)?(double)(mmit->first)/(num_beta-1):1.0;
      kappa *= 2.0 * M_PI * frac / ( a * num_a_per_lambda );

      ofs_coef << "Kappa index:  " << mmit->first << endl;
      ofs_coef << "Kappa: "; kappa.Print(ofs_coef); ofs_coef << endl;
      ofs_coef << "Degeneracies: " << endl;
      for (unsigned int i=0; i<degen[mmit->first].size(); i++)
      {
         for (sit=degen[mmit->first][i].begin();
              sit!=degen[mmit->first][i].end(); sit++)
         {
            ofs_coef << "\t" << *sit;
         }
         ofs_coef << endl;
      }
      for (mit=mmit->second.begin(); mit!=mmit->second.end(); mit++)
      {
         ofs_coef << "Eigenvalue Index:  " << mit->first << endl;
         mit->second.Print(ofs_coef);
         mit->second.PrintMathematica(ofs_avg);
      }
      ofs_coef << endl;
      ofs_avg << endl;

      // CalcCoefs(mmit->second);
   }
   ofs_coef.close();
   ofs_avg.close();

   for (int i=0; i<nev; i++) { delete init_vecs[i]; }

   delete HCurlFESpace;
   delete L2FESpace;
   delete bravais;
   delete eq;
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

   Array<int> bOffsets(3);
   bOffsets[0] = 0;
   bOffsets[1] = HCurlFESpace.TrueVSize();
   bOffsets[2] = HCurlFESpace.TrueVSize();
   bOffsets.PartialSum();

   int locSize = 2*HCurlFESpace.TrueVSize();
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


   BlockVector Ea(NULL,bOffsets);
   BlockVector Eb(NULL,bOffsets);

   Vector k(3);
   Array2D<int> n(1,3);
   n = 0.0;
   /*
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
         n.SetSize(9,3);
         n[ 0][0] =  0; n[ 0][1] =  0; n[ 0][2] =  0;

         n[ 1][0] =  1; n[ 1][1] =  1; n[ 1][2] =  1;
         n[ 2][0] =  1; n[ 2][1] =  1; n[ 2][2] = -1;
         n[ 3][0] =  1; n[ 3][1] = -1; n[ 3][2] =  1;
         n[ 4][0] =  1; n[ 4][1] = -1; n[ 4][2] = -1;
         n[ 5][0] = -1; n[ 5][1] =  1; n[ 5][2] =  1;
         n[ 6][0] = -1; n[ 6][1] =  1; n[ 6][2] = -1;
         n[ 7][0] = -1; n[ 7][1] = -1; n[ 7][2] =  1;
         n[ 8][0] = -1; n[ 8][1] = -1; n[ 8][2] = -1;
      }
      break;
   };
   */
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
         cout << "eigenvalues and eigenvectors" << endl;
         eval.Print(cout);
         evect.Print(cout);
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

   cout << "nev " << nev << ", size of E0 " << E0.size() << endl;

   nev = 2*E0.size();
   // if ( lattice_type == 2 ) nev += 2;
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
      Ea.SetData(init_vecs[2*i+0]->GetData());
      Eb.SetData(init_vecs[2*i+1]->GetData());

      cosCoef.SetModeIndices(n[e2n[i]][0],n[e2n[i]][1],n[e2n[i]][2]);
      sinCoef.SetModeIndices(n[e2n[i]][0],n[e2n[i]][1],n[e2n[i]][2]);

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
   /*
   if ( lattice_type == 2 )
   {
     Ea.SetData(init_vecs[nev-2]->GetData()); Ea.Randomize();
     Ea.SetData(init_vecs[nev-1]->GetData()); Ea.Randomize();
   }
   */
}

void ComputeFourierCoefficients(int myid, ostream & ofs_coef, int nmax,
                                HCurlFourierSeries & fourierHCurl,
                                HDivFourierSeries  & fourierHDiv,
                                MaxwellBlochWaveEquation & eq,
                                ParFiniteElementSpace & HCurlFESpace,
                                ParFiniteElementSpace & HDivFESpace,
                                const vector<double> & eigenvalues,
                                map<int,FourierVectorCoefficients> & mfc)
{
   HypreParVector  Er(HCurlFESpace.GetComm(),
                      HCurlFESpace.GlobalTrueVSize(),
                      NULL,
                      HCurlFESpace.GetTrueDofOffsets());

   HypreParVector  Ei(HCurlFESpace.GetComm(),
                      HCurlFESpace.GlobalTrueVSize(),
                      NULL,
                      HCurlFESpace.GetTrueDofOffsets());

   HypreParVector  Br(HDivFESpace.GetComm(),
                      HDivFESpace.GlobalTrueVSize(),
                      NULL,
                      HDivFESpace.GetTrueDofOffsets());

   HypreParVector  Bi(HDivFESpace.GetComm(),
                      HDivFESpace.GlobalTrueVSize(),
                      NULL,
                      HDivFESpace.GetTrueDofOffsets());

   double tol = 1e-6;
   Vector Err(3), Eri(3), Eir(3), Eii(3), E0r(3), E0i(3);
   Vector Brr(3), Bri(3), Bir(3), Bii(3), B0r(3), B0i(3);

   for (int i=0; i<=nmax; i++)
   {
      for (int j=(i==0)?0:-nmax; j<=nmax; j++)
      {
         for (int k=(i==0&&j==0)?0:-nmax; k<=nmax; k++)
         {
            fourierHCurl.SetMode(i, j, k);
            fourierHDiv.SetMode(i, j, k);
            for (unsigned int l=0; l<eigenvalues.size(); l++)
            {
               eq.GetEigenvector(l, Er, Ei, Br, Bi);

               fourierHCurl.GetCoefficient(Er, Err, Eri);
               fourierHCurl.GetCoefficient(Ei, Eir, Eii);

               fourierHDiv.GetCoefficient(Br, Brr, Bri);
               fourierHDiv.GetCoefficient(Bi, Bir, Bii);

               if ( myid == 0 )
               {
                  double nrmErr = Err.Normlinf();
                  double nrmEri = Eri.Normlinf();
                  double nrmEir = Eir.Normlinf();
                  double nrmEii = Eii.Normlinf();

                  double nrmBrr = Brr.Normlinf();
                  double nrmBri = Bri.Normlinf();
                  double nrmBir = Bir.Normlinf();
                  double nrmBii = Bii.Normlinf();

                  if ( nrmErr > tol || nrmEri > tol ||
                       nrmEir > tol || nrmEii > tol ||
                       nrmBrr > tol || nrmBri > tol ||
                       nrmBir > tol || nrmBii > tol )
                  {
                     /*
                     ofs_coef << kappa[0] << "\t" << kappa[1] << "\t" << kappa[2]
                         << "\t" << i << "\t" << j << "\t" << k
                         << "\t" << l << "\t" << eigenvalues[l];
                     for (int kk=0; kk<3; kk++)
                     {
                       ofs_coef << "\t" << Err[kk] << "\t" << Eri[kk]
                           << "\t" << Eir[kk] << "\t" << Eii[kk];
                     }
                     ofs_coef << endl;
                     */
                     if ( mfc.find(l) == mfc.end() )
                     {
                        mfc[l].SetOmega(sqrt(fabs(eigenvalues[l])));
                        mfc[l].SetLabels("E", "B");
                     }
                     E0r = Err; E0r -= Eii;
                     E0i = Eir; E0i += Eri;
                     B0r = Brr; B0r -= Bii;
                     B0i = Bir; B0i += Bri;
                     mfc[l].AddCoefficients(i,j,k,E0r,E0i,B0r,B0i);

                     if ( i != 0 || j != 0 || k != 0 )
                     {
                        E0r = Err; E0r += Eii;
                        E0i = Eir; E0i -= Eri;
                        B0r = Brr; B0r += Bii;
                        B0i = Bir; B0i -= Bri;
                        mfc[l].AddCoefficients(-i,-j,-k,E0r,E0i,B0r,B0i);
                     }
                  }
               }
            }
         }
      }
   }
}
/*
void IdentifyDegeneracies(const vector<double> & eigenvalues,
                          double zero_tol, double rel_tol,
                          vector<set<int> > & degen)
{
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
}
*/
void WriteWaveVectors(ostream & os,
                      const Vector & Er, const Vector & Ei,
                      const Vector & k, double omega)
{
   Vector Br(3), Bi(3), Dr(3), Di(3);
}

double mass_coef(const Vector & x)
{
   double epsr = 1.0;
   double eps1 = 10.0;
   double eps2 = 100.0;

   switch ( prob_ )
   {
      case 0:
         // Slab
         if ( fabs(x(0)) <= 0.5 ) { epsr = eps1; }
         break;
      case 1:
         // Cylinder
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= 0.5 ) { epsr = eps1; }
         break;
      case 2:
         // Sphere
         if ( x.Norml2() <= 0.5 ) { epsr = eps1; }
         break;
      case 3:
         // Sphere and 3 Rods
         if ( x.Norml2() <= 0.2 ) { epsr = eps1; }
         if ( sqrt(x(1)*x(1)+x(2)*x(2)) <= 0.1 ) { epsr = eps1; }
         if ( sqrt(x(2)*x(2)+x(0)*x(0)) <= 0.1 ) { epsr = eps1; }
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= 0.1 ) { epsr = eps1; }
         break;
      case 4:
         // Sphere and 4 Rods
         if ( x.Norml2() <= 0.2 ) { epsr = eps1; }
         {
            double r2 = x(0)*x(0)+x(1)*x(1)+x(2)*x(2);
            double a = 0.1 * sqrt(1.5);
            if ( sqrt(r2-x(0)*x(1)-x(1)*x(2)-x(2)*x(0)) <= a ) { epsr = eps1; }
            if ( sqrt(r2+x(0)*x(1)+x(1)*x(2)-x(2)*x(0)) <= a ) { epsr = eps1; }
            if ( sqrt(r2+x(0)*x(1)-x(1)*x(2)+x(2)*x(0)) <= a ) { epsr = eps1; }
            if ( sqrt(r2-x(0)*x(1)+x(1)*x(2)+x(2)*x(0)) <= a ) { epsr = eps1; }
         }
         break;
      case 5:
         // Two spheres in a BCC configuration
         if ( x.Norml2() <= 0.3 )
         {
            epsr = eps2;
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

               if ( u.Norml2() <= 0.2 ) { epsr = eps2; }
            }
         }
         break;
      case 6:
         // Sphere and 6 Rods
         if ( x.Norml2() <= 0.2 ) { epsr = eps1; }
         {
            double r2 = x(0)*x(0)+x(1)*x(1)+x(2)*x(2);
            double a = 0.05 * sqrt(2.0);
            if ( sqrt(r2+x(0)*x(0)-2.0*x(1)*x(2)) <= a ) { epsr = eps1; }
            if ( sqrt(r2+x(0)*x(0)+2.0*x(1)*x(2)) <= a ) { epsr = eps1; }
            if ( sqrt(r2+x(1)*x(1)-2.0*x(2)*x(0)) <= a ) { epsr = eps1; }
            if ( sqrt(r2+x(1)*x(1)+2.0*x(2)*x(0)) <= a ) { epsr = eps1; }
            if ( sqrt(r2+x(2)*x(2)-2.0*x(0)*x(1)) <= a ) { epsr = eps1; }
            if ( sqrt(r2+x(2)*x(2)+2.0*x(0)*x(1)) <= a ) { epsr = eps1; }
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
         if ( fabs(x(1)) + fabs(x(2)) < 0.25 ||
              fabs(x(0)) + fabs(x(2)) < 0.25 ||
              fabs(x(0)) + fabs(x(1) - 0.5) < 0.25 )
         {
            epsr = eps1;
         }
         break;
   }
   // return epsr * MAXWELL_EPS0;
   return epsr;
}

double stiffness_coef(const Vector &x)
{
   // return 1.0/mu0_;
   // return 1.0/MAXWELL_MU0;
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

#ifdef MFEM_USE_LAPACK
extern "C" void
zgesvd_(char *, char *, int *, int *, double *, int *, double *,
        double *, int *, double *, int *, double *, int *,
        double *, int *);

extern "C" void
zgels_(char *, int *, int *, int *, double *, int *, double *, int *,
       double *, int *, int *);
#endif

void CalcCoefs(map<int,FourierVectorCoefficients> & mf)
{
   unsigned int n = mf.size();

   cout << "Number of Eigenvalues: " << n << endl;

   int m = 48 + 2*11;

   double eb[m*21];
   double dh[m];

   Vector Ur, Ui, Vr, Vi;

   for (int i=0; i<m*21; i++) { eb[i] = 0.0; }

   for (int i=0; i<min(2,(int)n); i++)
   {
      // U is the E field and V is the B field
      mf[i].GetCoefficients(0,0,0,Ur,Ui,Vr,Vi);

      for (int j=0; j<3; j++)
      {
         if ( fabs(Ur[j]) < 1.0e-6 ) { Ur[j] = 0.0; }
         if ( fabs(Ui[j]) < 1.0e-6 ) { Ui[j] = 0.0; }
         if ( fabs(Vr[j]) < 1.0e-6 ) { Vr[j] = 0.0; }
         if ( fabs(Vi[j]) < 1.0e-6 ) { Vi[j] = 0.0; }
      }

      int cu = 0;
      int cl = 0;
      for (int j=0; j<6; j++)
      {
         cl = j;
         for (int k=0; k<j; k++)
         {
            if ( k < 3)
            {
               eb[12*i+m*cl+2*j+0] = Ur[k];
               eb[12*i+m*cl+2*j+1] = Ui[k];
            }
            else
            {
               eb[12*i+m*cl+2*j+0] = Vr[k-3];
               eb[12*i+m*cl+2*j+1] = Vi[k-3];
            }
            cl += 5-k;
         }

         for (int k=j; k<6; k++)
         {
            if ( k < 3)
            {
               eb[12*i+m*cu+2*j+0] = Ur[k];
               eb[12*i+m*cu+2*j+1] = Ui[k];
            }
            else
            {
               eb[12*i+m*cu+2*j+0] = Vr[k-3];
               eb[12*i+m*cu+2*j+1] = Vi[k-3];
            }
            cu++;
         }
      }

      // U is the D field and V is the H field
      mf[i].GetDualCoefficients(0,0,0,Ur,Ui,Vr,Vi);

      // Interleave the field vectors like so:
      //
      // Dr[0], Di[0], Dr[1], Di[1], Dr[2], Di[2],
      // Hr[0], Hi[0], Hr[1], Hi[1], Hr[2], Hi[2]
      for (int j=0; j<3; j++)
      {
         dh[12*i+2*j+0] = Ur[j];
         dh[12*i+2*j+1] = Ui[j];

         dh[12*i+2*j+6] = Vr[j];
         dh[12*i+2*j+7] = Vi[j];
      }
   }
   for (int i=48; i<m; i++) { dh[i] =0.0; }

   {
      // Cludge to test increasing the rank
      /*
      eb[m*1+48] = 1.0;
      eb[m*4+50] = 1.0;
      eb[m*6+52] = 1.0;
      eb[m*7+54] = 1.0;
      eb[m*8+56] = 1.0;
      eb[m*9+58] = 1.0;
      eb[m*10+60] = 1.0;
      eb[m*13+62] = 1.0;
      eb[m*16+64] = 1.0;
      eb[m*18+66] = 1.0;
      eb[m*19+68] = 1.0;
      */
      for (int i=0; i<m; i++)
      {
         for (int j=0; j<21; j++)
         {
            cout << eb[m*j+i] << " ";
         }
         cout << endl;
      }
      /*
      cout << "col 6 ";
      for (int i=0; i<m; i++)
      {
        cout << eb[m*6+i] << " ";
      }
      cout << endl;
      cout << "col 9 ";
      for (int i=0; i<m; i++)
      {
        cout << eb[m*9+i] << " ";
      }
      cout << endl;
      cout << "col 18 ";
      for (int i=0; i<m; i++)
      {
        cout << eb[m*18+i] << " ";
      }
      cout << endl;
      */
      char jobu  = 'N';
      char jobvt = 'N';
      int M = m/2;
      int N = 21;
      double S[21];
      double * U  = NULL;
      int ldu = 1;
      double * VT = NULL;
      int ldvt = 1;
      double * work = NULL;
      double dwork[2];
      int lwork = -1;
      double rwork[105];
      int info = 0;

      cout << "Calling zgesvd to compute lwork" << endl;
      dwork[0] = -1.0; dwork[1] = -1.0;
      zgesvd_(&jobu, &jobvt, &M, &N, eb, &M, S, U, &ldu, VT, &ldvt,
              dwork, &lwork, rwork, &info);
      // cout << "lwork = " << lwork << endl;
      // cout << "dwork = " << dwork[0] << "," << dwork[1] << endl;

      lwork = (int)round(dwork[0]);
      work = new double[2*lwork];

      cout << "Calling zgesvd" << endl;
      for (int i=0; i<min(M,N); i++) { S[i] = 0.0; }
      zgesvd_(&jobu, &jobvt, &M, &N, eb, &M, S, U, &ldu, VT, &ldvt,
              work, &lwork, rwork, &info);
      delete [] work;

      cout << "Singular values: ";
      for (int i=0; i<min(M,N); i++)
      {
         cout << " " << S[i];
      }
      cout << endl;
   }
   {
      char trans = 'N';
      int M = m/2;
      int N = 21;
      double dwork[2];
      double * work =  NULL;
      int lwork = -1;
      int info  =  0;
      int one = 1;

      double rhs[52];

      for (int i=0; i<52; i++)
      {
         rhs[i] = dh[i];
      }

      zgels_(&trans, &M, &N, &one, eb, &M, rhs, &M, dwork, &lwork, &info);

      lwork = (int)round(dwork[0]);
      work = new double[2*lwork];

      zgels_(&trans, &M, &N, &one, eb, &M, rhs, &M, work, &lwork, &info);

      delete [] work;

      int c = 0;
      for (int i=0; i<6; i++)
      {
         for (int j=i; j<6; j++)
         {
            cout << " ("<<rhs[c+0]<<","<<rhs[c+1]<<")";
            c += 2;
         }
         cout << endl;
      }
   }
}
