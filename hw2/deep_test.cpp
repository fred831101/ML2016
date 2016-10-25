#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cmath>

using namespace std;

int main(int argc, char** argv){
    fstream fin,fout;
    //get params
    fin.open(argv[1],ios::in);
    double finalmean[57];
    double finalstd[57];
    double bparameter_l1[57] = {0};
    double bparameter_l2 = 0;
    double wparameter_l1[57][57]={0};
    double wparameter_l2[57]={0};
    for(int i=0;i<57;i++){
        fin >> bparameter_l1[i];
    }
    fin >> bparameter_l2;
    for(int i=0;i<57;i++){
        for(int j=0;j<57;j++)
          fin>>wparameter_l1[i][j];
    }
    for(int i=0;i<57;i++)
        fin >> wparameter_l2[i];
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
    double l1store[57];
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
       for(int i=0;i<57;i++){
         totest[i] = (totest[i]-finalmean[i])/finalstd[i];
       }

       for(int st=0;st<57;st++){
       double iteration_ans = bparameter_l1[st];
         for(int i=0;i<57;i++){
           iteration_ans += wparameter_l1[i][st] * totest[i];
         }
       iteration_ans *= -1;
       iteration_ans = exp(iteration_ans);
       iteration_ans = 1/(1+iteration_ans);
       l1store[st] = iteration_ans;
       }
       double iteration_ans2 = bparameter_l2;
         for(int i=0;i<57;i++){
           iteration_ans2 += wparameter_l2[i] * l1store[i];
         }
       iteration_ans2 *= -1;
       iteration_ans2 = exp(iteration_ans2);
       iteration_ans2 = 1/(1+iteration_ans2);

       if(iteration_ans2>0.5)
         fout<<"1";
       else
         fout<<"0";
       fout<<endl;
    }
    fin.close();
    fout.close();
    return 0;
}
