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
    //Store all train data in a 2D array
    double traindata[4001][58];
    fstream fin, fout;
    fin.open(argv[1],ios::in);
    double tostore;
    string sread;
    for (int i=0;i<=4000;i++){
      getline(fin,sread);
      int slength = sread.length();
      for (int j=0;j<slength;j++)
        if(sread[j]==',')
          sread[j]=' ';
      stringstream ss;
      ss.clear();
      ss.str(sread);
      ss>>tostore;
      for(int times=0;times<58;times++){
         ss >> tostore;
         traindata[i][times]=tostore;
      }
    }
    fin.close();
    //Normalization
     double finalmean[57];
     double finalstd[57];
     for(int line=0;line<57;line++){
      double mean=0.0;
      double std=0.0;
      for(int tda=0;tda<=4000;tda++){
          mean+=traindata[tda][line];
      }
      mean = mean/ 4001.0 ;
      for(int tda=0;tda<=4000;tda++){
           std += (traindata[tda][line]-mean)*(traindata[tda][line]-mean);
      }
      std = std/  4001.0 ;
      std = sqrt(std);
      finalmean[line]=mean;
      finalstd[line]=std;
      for(int tda=0;tda<=4000;tda++){
         double normaldata = (traindata[tda][line]-mean)/std;
         traindata[tda][line] = normaldata;
      }
    }

    //initialization of logistic parameters
    double bparameter=0;
    double wparameter[57]={0};
    double iter_diff[4001]={0}; // for future storage of y-f(x)

    //One iteration of logistic regression
    
    for(int itercounter=0;itercounter<MAX_ITERATION;itercounter++){
     if (itercounter%250==0){ //display use
      cout<<setw(7)<<itercounter<<' ';
      cout<<setw(15)<<wparameter[13]<<setw(15)<<wparameter[47]<<endl;
     }
    //one iteration
    //calculate all iterated difference (y-f)
    for(int dat=0;dat<=4000;dat++){
        double iteration_ans = bparameter;
        for(int i=0;i<57;i++)
          iteration_ans += wparameter[i] * traindata[dat][i];
        // sigmoid
        iteration_ans *= -1;
        iteration_ans = exp(iteration_ans);
        iteration_ans = 1/(1+iteration_ans);
        // end sigmoid
        iter_diff[dat] = traindata[dat][57] - iteration_ans;
    }
    //calculate new b
    double bcount=0;
    for(int dat=0;dat<=4000;dat++)
        bcount += iter_diff[dat]; //summation of errors
    bparameter += ( N_VALUE * bcount); //N_VALUE is learning rate
    //calculate new ws
    for(int i=0;i<57;i++){
      double wcount=0;
      double tempans;
      for(int dat=0;dat<=4000;dat++){
         tempans = iter_diff[dat] * traindata[dat][i];
         wcount += tempans;
      }
      wparameter[i] += (N_VALUE * wcount);
    }
    }
    
    //save parameters
    fout.open(argv[2],ios::out);
    fout << bparameter <<endl;
    for(int i=0;i<57;i++){
       fout << wparameter[i];
       fout << endl;
    }
    fout<<endl;
    for(int i=0;i<57;i++){
       fout << finalmean[i] << ' ' << finalstd[i];
       fout << endl;
    }
    fout.close();
    //train complete
    return 0;
}
