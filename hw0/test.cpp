#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

int main(int argc, char** argv){
	stringstream ss;
	ss << argv[1] << endl;
	int userin;
    ss >> userin;
    fstream f1,f2;
    f1.open(argv[2],ios::in);
    f2.open("ans1.txt",ios::out);
    //read    
    double table[500][11];
    for (int i=0;i<500;i++)
      for (int j=0;j<11;j++)
      	f1 >> table[i][j];
    double tobesort[500];
    for (int i=0;i<500;i++)
      tobesort[i]=table[i][userin];
    sort(tobesort,tobesort+500);
    for (int i=0;i<499;i++)
      f2<<tobesort[i]<<",";
    f2<<tobesort[499];
    f1.close();
    f2.close();
	return 0;
}