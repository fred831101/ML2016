import numpy as np
import csv
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans
from scipy import stats
import sys

X_title = open(sys.argv[1]+'title_StackOverflow.txt','r')

Out_index = open(sys.argv[1]+'check_index.csv', 'r')
Out_reader = csv.reader(Out_index)
Question = np.zeros(shape=[5000000,2],dtype='int')
next(Out_reader)
for row in Out_reader:
    arr = [int(row[x]) for x in range(3)]
    Question[arr[0]][0]=arr[1]
    Question[arr[0]][1]=arr[2]

Iteration = 10
Out_ans = np.zeros(shape=[5000000,21],dtype='int')

Fitter1 = CountVectorizer(min_df=1,stop_words='english')
X_Bow = Fitter1.fit_transform(X_title).toarray()
Fitter2 = TfidfTransformer(sublinear_tf=True)
X_TFIDF = Fitter2.fit_transform(X_Bow).toarray()
svd = TruncatedSVD(n_components=20, n_iter=Iteration)
X_TFIDF = svd.fit_transform(X_TFIDF)
norm = Normalizer(copy=False)
X_TFIDF = norm.fit_transform(X_TFIDF)
Kmeansfit = MiniBatchKMeans(n_clusters=50, init='k-means++', n_init=1,init_size=1000, batch_size=1000)

for i in range(21):
    print(repr(i)+' iter')
    X_cluster=Kmeansfit.fit_predict(X_TFIDF)
    for j in range(5000000):
    	Out_ans[j][i] = int(X_cluster[Question[j][0]]==X_cluster[Question[j][1]])

O_vote=(stats.mode(Out_ans,axis=1)).mode

f = open(sys.argv[2], 'w')
f.write('ID,Ans\n')
for i in range(5000000):
	f.write(repr(i)+','+repr(O_vote[i][0])+'\n')
