# Mengimpor library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor dataset
dataset = pd.read_csv('Review_restoran.tsv', delimiter = '\t', quoting = 3)
 
# Line 10-34 adalah proses yang dilakukan setahap demi setahap (agar mudah dipahami)
# Melihat item pertama di dataset
dataset['Review'][0]
 
# Mengimpor library re dan NLTK
import re
import nltk
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower()
review = review.split()
 
# Mendownload daftar kata yang ada (vocabulary)
nltk.download('stopwords')
from nltk.corpus import stopwords
 
# Memeriksa daftar kata di stopwords
inggris = stopwords.words('english')
indo = stopwords.words('indonesian')
 
# Menghilangkan kata yang tidak ada di stopwords
review = [word for word in review if not word in inggris]
 
# Melakukan proses stemming (penggunaan kata dasar)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
 
# Membersihkan kalimat dari kata yang ada di stopwords
review = [ps.stem(word) for word in review if not word in inggris]
review = ' '.join(review)
 
# Melakukan proses cleaning pada teks
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in inggris]
    review = ' '.join(review)
    corpus.append(review)
 
# Membuat model Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
 
# Membagi dataset ke dalam Training dan Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
 
# Menggunakan beberapa teknik klasifikasi untuk membandingkan akurasinya
# Metode Logistic Regression
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, y_train)
 
# Metode K-nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)
 
# Metode SVM
from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'linear', random_state = 0)
classifierSVM.fit(X_train, y_train)
 
# Metode Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)
 
# Metode Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDT.fit(X_train, y_train)
 
# Metode Random Forest
from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifierRF.fit(X_train, y_train)
 
# Memprediksi hasil Test Set
y_pred_LR = classifierLR.predict(X_test)    # logistic Regression
y_pred_KNN = classifierKNN.predict(X_test)  # K-nearest Neighbors
y_pred_SVM = classifierSVM.predict(X_test)  # SVM
y_pred_NB = classifierNB.predict(X_test)    # Naive Bayes
y_pred_DT = classifierDT.predict(X_test)    # Decision Tree
y_pred_RF = classifierRF.predict(X_test)    # Random Forest
 
# Membuat Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(y_test, y_pred_LR)
cm_KNN = confusion_matrix(y_test, y_pred_KNN)
cm_SVM = confusion_matrix(y_test, y_pred_SVM)
cm_NB = confusion_matrix(y_test, y_pred_NB)
cm_DT = confusion_matrix(y_test, y_pred_DT)
cm_RF = confusion_matrix(y_test, y_pred_RF)
 
# Menilai akurasi masing-masing metode
akurasi_LR = ((cm_LR[0][0]+cm_LR[1][1])/(cm_LR[0][0]+cm_LR[1][1]+cm_LR[0][1]+cm_LR[1][0]))*100
akurasi_KNN = ((cm_KNN[0][0]+cm_KNN[1][1])/(cm_KNN[0][0]+cm_KNN[1][1]+cm_KNN[0][1]+cm_KNN[1][0]))*100
akurasi_SVM = ((cm_SVM[0][0]+cm_SVM[1][1])/(cm_SVM[0][0]+cm_SVM[1][1]+cm_SVM[0][1]+cm_SVM[1][0]))*100
akurasi_NB = ((cm_NB[0][0]+cm_NB[1][1])/(cm_NB[0][0]+cm_NB[1][1]+cm_NB[0][1]+cm_NB[1][0]))*100
akurasi_DT = ((cm_DT[0][0]+cm_DT[1][1])/(cm_DT[0][0]+cm_DT[1][1]+cm_DT[0][1]+cm_DT[1][0]))*100
akurasi_RF = ((cm_RF[0][0]+cm_RF[1][1])/(cm_RF[0][0]+cm_RF[1][1]+cm_RF[0][1]+cm_RF[1][0]))*100
