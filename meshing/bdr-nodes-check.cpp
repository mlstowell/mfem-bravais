#include <fstream>
#include <iostream>
#include <map>
#include <set>

using namespace std;

int main(int argc, char ** argv)
{
   // ifstream ifs("bdr-nodes.txt");
   ifstream ifs("bdr-nodes-bcc.txt");

   map<int,int> s2m;
   map<int,int> m2m;
   map<int,set<int> > m2s;

   set<int> masters;
   set<int> slaves;

   // int nv = 65;
   int nv = 365;
   for (int i=0; i<nv; i++)
   {
      int s,m;

      ifs >> s >> m;

      // if ( s != m ) {
      s2m[s] = m;
      slaves.insert(s);
      // }
      m2s[m].insert(s);
      masters.insert(m);
   }

   ifs.close();

   cout << "Found " << slaves.size() << " slave nodes" << endl;
   cout << "Found " << masters.size() << " master nodes" << endl;

   set<int> inter;
   set_intersection(slaves.begin(),slaves.end(),
                    masters.begin(),masters.end(),
                    insert_iterator<set<int> >(inter,inter.begin()));

   set<int>::iterator sit;
   for (sit=inter.begin(); sit!=inter.end(); sit++)
   {
      cout << *sit << endl;
   }

   int c = 0;
   map<int,set<int> >::iterator mit;
   for (mit=m2s.begin(); mit!=m2s.end(); mit++)
   {
      cout << mit->first << ": ";
      for (sit=mit->second.begin(); sit!=mit->second.end(); sit++)
      {
         cout << " " << *sit;
      }
      cout << endl;
      m2m[mit->first] = c;
      c++;
   }

   // ifs.open("rhombic-dodecahedron-hex-n2.mesh");
   ifs.open("trunc-oct-bdr-nodes-r1.mesh");
   string buf = "";
   while ( buf != "elements" ) { ifs >> buf; }
   int nelem;
   ifs >> nelem;
   for (int e=0; e<nelem; e++)
   {
      int i0,i1,v;
      ifs >> i0 >> i1;
      cout << i0 << " " << i1;
      for (int i=0; i<8; i++)
      {
         ifs >> v;
         cout << " " << /*v << ":" << s2m[v] << ":" <<*/ m2m[s2m[v]];
      }
      cout << endl;
   }

   while ( buf != "boundary" ) { ifs >> buf; }
   int nbdr;
   ifs >> nbdr;
   for (int b=0; b<nbdr; b++)
   {
      int i0,i1,v;
      ifs >> i0 >> i1;
      cout << i0 << " " << i1;
      for (int i=0; i<4; i++)
      {
         ifs >> v;
         cout << " " << /*v << ":" << s2m[v] << ":" <<*/ m2m[s2m[v]];
      }
      cout << endl;
   }

   ifs.close();
}
