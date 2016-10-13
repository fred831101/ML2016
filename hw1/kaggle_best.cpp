#include<iostream>
#include<fstream>
#include<string>
#include<sstream>

using namespace std;

#define N_VALUE 0.0000000006
#define Y_VALUE 0.1
#define MAX_ITERATION 1000000

void saveparameters(double b, double w[18][9]){
   fstream fout;
   fout.open("parameter.csv",ios::out);
   fout << b <<endl;
   for(int i=0;i<18;i++){
       for(int j=0;j<9;j++)
          fout << w[i][j] <<' ';
       fout << endl;
   }
   fout.close();
}

int main()
{
    //Store all train data in a 3D array
    double traindata[12][18][480];
    string toread;
    fstream fin, fout;
    double tostore;
    fin.open("train2.csv",ios::in);
    for(int month=0;month<12;month++)
    {
      for(int date=0;date<20;date++){
         for(int line=0;line<18;line++){
            getline(fin,toread);
            stringstream ss;
            ss.clear();
            ss.str(toread);
            for(int times=0;times<24;times++){
               ss >> tostore;
               traindata[month][line][date*24+times]=tostore;
            }
         }
     }
    }

    //initialize parameters
    double bparameter_now=0;
    //double bparameter_next=0;
    double wparameter_now[18][9]={0};
    //double wparameter_next[18][9]={0};
    double iteranswer[12][471];
    double realanswer[12][471];

    //real PM2.5 answer storage(y)
    for(int month=0;month<12;month++){
      for(int hour=9;hour<480;hour++)
        realanswer[month][hour-9]=traindata[month][9][hour];
    }


    for(int itercounter=0;itercounter<MAX_ITERATION;itercounter++){
     /*if (itercounter%250==0){
      cout<<itercounter<<' ';
      cout<<wparameter_now[2][2]<<endl;
      //cout<<iteranswer[0][0]<<endl<<endl;
      //saveparameters(bparameter_now,wparameter_now);
      }*/
    //one iteration
    //calculate all iterated PM2.5
    for(int month=0;month<12;month++){
      for(int hour=0;hour<471;hour++){
        double iteratedPM = bparameter_now;
          for(int i=0;i<18;i++)
           for(int j=0;j<9;j++)
           {
              iteratedPM += wparameter_now[i][j] * traindata[month][i][hour+j];
           }
        iteranswer[month][hour] = iteratedPM;
      }
    }
    //calculate new b
    double bcount=0;
    for(int month=0;month<12;month++)
      for(int hour=0;hour<471;hour++){
        bcount += (realanswer[month][hour] - iteranswer[month][hour]);
    }
    bparameter_now = bparameter_now + ( N_VALUE * bcount);
    //calculate new ws
    for(int i=0;i<18;i++)
      for(int j=0;j<9;j++){
         double wcount=0;
         double tempans;
         for(int month=0;month<12;month++)
           for(int hour=0;hour<471;hour++){
             tempans = realanswer[month][hour] - iteranswer[month][hour];
             tempans *= traindata[month][i][hour+j];
             wcount += tempans;
         }
         wparameter_now[i][j] = wparameter_now[i][j] + (N_VALUE * wcount);
    }
    }
    //saveparameters(bparameter_now,wparameter_now);
    fin.close();
    fin.open("test2.csv",ios::in);
    fout.open("kaggle_best.csv",ios::out);
    fout<<"id,value\n";
    for(int tcase=0;tcase<240;tcase++){
       fout<<"id_"<<tcase<<',';
       double tout=0;
       for(int i=0;i<18;i++)
        for(int j=0;j<9;j++){
           double temp;
           fin >> temp;
           tout += wparameter_now[i][j]*temp;
        }
       fout<<tout;
       fout<<endl;
    }
    fin.close();
    fout.close();
    return 0;
}
