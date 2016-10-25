#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<cmath>
#include<random>

using namespace std;

#define N_VALUE 0.001
#define MAX_ITERATION 3000


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

    std::default_random_engine generator (0);
    std::uniform_real_distribution<double> distribution (-1.0,1.0);
    double bparameter_l1[57] = {0};
    double bparameter_l2 = distribution(generator);
    double wparameter_l1[57][57]={0};
    double wparameter_l2[57]={0};
    double iter_l1[4001][57] = {0};
    double error_l1[4001][57] = {0};
    double iter_diff_l2[4001]={0};
    for(int i=1;i<57;i++){
       for(int j=1;j<57;j++){
          wparameter_l1[i][j]=distribution(generator);
       }
       bparameter_l1[i]=distribution(generator);
       wparameter_l2[i]=distribution(generator);
    }

    for(int itercounter=0;itercounter<MAX_ITERATION;itercounter++){
     if (itercounter%20==0){
      cout<<setw(7)<<itercounter<<' '<<endl;
      cout<<setw(15)<<wparameter_l1[7][3]<<setw(15)<<wparameter_l1[7][7]<<endl;
      cout<<setw(15)<<wparameter_l2[11]<<setw(15)<<wparameter_l2[43]<<endl;
     }
    //one iteration
    //calculate layer 1
    for(int dat=0;dat<=4000;dat++){
        for(int tplayer=0;tplayer<57;tplayer++){
           double iteration_ans = bparameter_l1[tplayer];
           for(int i=0;i<57;i++)
             iteration_ans += traindata[dat][i] * wparameter_l1[i][tplayer];
           iteration_ans *= -1;
           iteration_ans = exp(iteration_ans);
           iteration_ans = 1/(1+iteration_ans);
           iter_l1[dat][tplayer]=iteration_ans;
       }
    }
    //calculate layer 2 diff(y-f)
    for(int dat=0;dat<=4000;dat++){
        double iteration_ans = bparameter_l2;
        for(int i=0;i<57;i++)
          iteration_ans += iter_l1[dat][i] * wparameter_l2[i];
        iteration_ans *= -1;
        iteration_ans = exp(iteration_ans);
        iteration_ans = 1/(1+iteration_ans);
        iter_diff_l2[dat] = traindata[dat][57] - iteration_ans;
        iter_diff_l2[dat] *= (iteration_ans * (1-iteration_ans));
    }
    //calculate layer 1 error
    for(int dat=0;dat<=4000;dat++){
        for(int i=0;i<57;i++){
          error_l1[dat][i] = iter_diff_l2[dat] * wparameter_l2[i];
          error_l1[dat][i] *= (iter_l1[dat][i] * (1-iter_l1[dat][i]));
        }
    }

    //modify layer 2
    //calculate new b
    double bcount=0;
    for(int dat=0;dat<=4000;dat++)
        bcount += iter_diff_l2[dat];
    bparameter_l2 += ( N_VALUE * bcount);
    //calculate new ws
    for(int i=0;i<57;i++){
      double wcount=0;
      double tempans;
      for(int dat=0;dat<=4000;dat++){
         tempans = iter_diff_l2[dat] * iter_l1[dat][i];
         wcount += tempans;
      }
      wparameter_l2[i] += (N_VALUE * wcount);
    }
    //modify layer 1
    for(int l1w=0;l1w<57;l1w++){
        //calculate each new b
        double bcount=0;
        for(int dat=0;dat<=4000;dat++)
            bcount += error_l1[dat][l1w];
        bparameter_l1[l1w] += ( N_VALUE * bcount);
        //calculate each new w
        for(int i=0;i<57;i++){
          double wcount=0;
          double tempans;
          for(int dat=0;dat<=4000;dat++){
             tempans = error_l1[dat][l1w] * traindata[dat][i];
             wcount += tempans;
          }
          wparameter_l1[i][l1w] += (N_VALUE * wcount);
        }
    }
    }
    //save params
    fout.open(argv[2],ios::out);
    for(int i=0;i<57;i++){
        fout << bparameter_l1[i] <<' ';
    }
    fout << endl << bparameter_l2 <<endl;
    for(int i=0;i<57;i++){
        for(int j=0;j<57;j++)
          fout<< wparameter_l1[i][j] << ' ' ;
        fout<<endl;
    }
    for(int i=0;i<57;i++){
       fout << wparameter_l2[i];
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
