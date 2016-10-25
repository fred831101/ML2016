#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cmath>

using namespace std;

#define N_VALUE 0.001
#define MAX_ITERATION 100000

int main(int argc, char** argv){
    fstream fin,fout;
    //get params
    fin.open(argv[1],ios::in);
    double finalmean[57];
    double finalstd[57];
    double bparameter=0;
    double wparameter[57];
    fin >> bparameter;
    for(int i=0;i<57;i++)
        fin>>wparameter[i];
    for(int i=0;i<57;i++)
        fin>>finalmean[i]>>finalstd[i];
    fin.close();

    //generate test data
    fin.open(argv[2],ios::in);
    fout.open(argv[3],ios::out);
    fout<<"id,label\n";
    double tostore;
    string sread;
    double totest[57];
    for(int tdase=1;tdase<=600;tdase++){
       fout<<tdase<<',';

       getline(fin,sread);
       int slength = sread.length();
       for (int j=0;j<slength;j++)
         if(sread[j]==',')
           sread[j]=' ';
       stringstream ss;
       ss.clear();
       ss.str(sread);
       ss>>tostore;
       for(int times=0;times<57;times++){
         ss >> tostore;
         totest[times]=tostore;
       }

       double iteration_ans = bparameter;
       for(int i=0;i<57;i++){
         totest[i] = (totest[i]-finalmean[i])/finalstd[i];
         iteration_ans += wparameter[i] * totest[i];
       }
       iteration_ans *= -1;
       iteration_ans = exp(iteration_ans);
       iteration_ans = 1/(1+iteration_ans);

       if(iteration_ans>0.5)
         fout<<"1";
       else
         fout<<"0";
       fout<<endl;
    }
    fin.close();
    fout.close();

}
