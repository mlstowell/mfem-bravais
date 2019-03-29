#include <fstream>
#include <iostream>
//#include <set>
#include <vector>

using namespace std;

double dot(const vector<double> & a, const vector<double> & b);

int main(int argc, char ** argv)
{

   vector<vector<double> > mNodes(6);

   for (int i=0; i<6; i++) { mNodes[i].resize(3); }
   mNodes[0][0] =  0.0; mNodes[0][1] = -0.5; mNodes[0][2] = -1.0;
   mNodes[1][0] =  0.0; mNodes[1][1] = -1.0; mNodes[1][2] = -0.5;
   mNodes[2][0] = -0.5; mNodes[2][1] = -1.0; mNodes[2][2] =  0.0;
   mNodes[3][0] = -1.0; mNodes[3][1] = -0.5; mNodes[3][2] =  0.0;
   mNodes[4][0] = -1.0; mNodes[4][1] =  0.0; mNodes[4][2] = -0.5;
   mNodes[5][0] = -0.5; mNodes[5][1] =  0.0; mNodes[5][2] = -1.0;

   vector<vector<double> > fNorms(7);
   for (int i=0; i<7; i++) { fNorms[i].resize(3); }
   fNorms[0][0] =  1.0; fNorms[0][1] =  1.0; fNorms[0][2] =  1.0;
   fNorms[1][0] =  0.0; fNorms[1][1] =  0.0; fNorms[1][2] =  2.0;
   fNorms[2][0] = -1.0; fNorms[2][1] =  1.0; fNorms[2][2] =  1.0;
   fNorms[3][0] =  0.0; fNorms[3][1] =  2.0; fNorms[3][2] =  0.0;
   fNorms[4][0] =  1.0; fNorms[4][1] =  1.0; fNorms[4][2] = -1.0;
   fNorms[5][0] =  2.0; fNorms[5][1] =  0.0; fNorms[5][2] =  0.0;
   fNorms[6][0] =  1.0; fNorms[6][1] = -1.0; fNorms[6][2] =  1.0;

   vector<double> fProjs(14);
   for (int i=0; i<14; i++)
   {
      if ( i < 7 )
      {
         fProjs[i] = i%2?-1.5:-2.0;
      }
      else
      {
         fProjs[i] = i%2?0.5:2.0;
      }
   }

   vector<vector<double> > vertices;
   vector<vector<int> >    vFaces;

   ifstream ifs("trunc-oct-bdr-nodes-r1.mesh");
   string buffer;
   bool eof = false;
   while (!ifs.eof() && !eof )
   {
      ifs >> buffer;
      if ( buffer == "elements" )
      {
         cout << "Found Elements" << endl;
      }
      if ( buffer == "boundary" )
      {
         cout << "Found Boundary" << endl;
      }
      if ( buffer == "vertices" )
      {
         cout << "Found Vertices" << endl;
         int nVert = -1;
         int nDim = -1;
         ifs >> nVert >> nDim;
         vertices.resize(nVert);
         vFaces.resize(nVert);
         for (int i=0; i<nVert; i++)
         {
            vertices[i].resize(nDim);
            for (int j=0; j<nDim; j++)
            {
               ifs >> vertices[i][j];
            }
            cout << i;
            for (int j=0; j<fNorms.size(); j++)
            {
               double d = dot(vertices[i],fNorms[j]);
               // cout << " " << d;
               if ( d == fProjs[j] )
               {
                  vFaces[i].push_back(j);
               }
               else if ( d == fProjs[j+fNorms.size()] )
               {
                  vFaces[i].push_back(j+fNorms.size());
               }
            }
            cout << "\t" << vFaces[i].size() << endl;
         }
         eof = true;
      }
   }
   ifs.close();

   vector<bool> vMaster(vFaces.size());
   for (unsigned int i=0; i<vFaces.size(); i++)
   {
      vMaster[i] = true;
      for (unsigned int j=0; j<vFaces[i].size(); j++)
      {
         vMaster[i] = vMaster[i] && (vFaces[i][j] < 7);
      }
      cout << "vertex " << i << " is a ";
      if ( vMaster[i] )
      {
         cout << "master";
      }
      else
      {
         cout << "slave";
      }
      cout << " vertex." << endl;
   }
}

double dot(const vector<double> & a, const vector<double> & b)
{
   double c = 0.0;
   for (int i=0; i<min(a.size(),b.size()); i++) { c += a[i] * b[i]; }
   return c;
}
