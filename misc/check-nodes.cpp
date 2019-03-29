#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

using namespace std;

int main(int argc, char ** argv)
{
   ifstream ifs("periodic-unit-rhombic-dodecahedron.mesh");

   string buf = "";
   while ( buf != "elements" ) { ifs >> buf; }

   map<int,vector<int> > e2v;
   map<int,set<int> >    v2e;

   int nelem = -1;
   ifs >> nelem;
   for (int e=0; e<nelem; e++)
   {
      int i0, i1, v;
      ifs >> i0 >> i1;
      for (int j=0; j<8; j++)
      {
         ifs >> v;
         e2v[e].push_back(v);
         v2e[v].insert(e);
      }
   }

   while ( buf != "boundary" ) { ifs >> buf; }

   map<int,vector<int> > b2v;
   map<int,set<int> >    v2b;

   int nbdr = -1;
   ifs >> nbdr;
   for (int b=0; b<nbdr; b++)
   {
      int i0, i1, v;
      ifs >> i0 >> i1;
      for (int j=0; j<4; j++)
      {
         ifs >> v;
         b2v[b].push_back(v);
         v2b[v].insert(b);
      }
   }

   ifs.close();

   cout << "Found " << e2v.size() << " elements" << endl;
   cout << "Found " << v2e.size() << " vertices" << endl;

   map<int,set<int> >::iterator mit;
   for (mit=v2e.begin(); mit!=v2e.end(); mit++)
   {
      cout << mit->first << ":  " << mit->second.size() << endl;
   }

   cout << "Found " << b2v.size() << " boundary elements" << endl;
   cout << "Found " << v2b.size() << " boundary vertices" << endl;

   //map<int,set<int> >::iterator mit;
   for (mit=v2b.begin(); mit!=v2b.end(); mit++)
   {
      cout << mit->first << ":  " << mit->second.size() << endl;
   }
}


