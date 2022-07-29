import sys
from concurrent.futures import thread, ThreadPoolExecutor


import columns as columns
import pycebox.ice as icebox
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Convolution1D, Dense, Dropout, MaxPooling1D, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from keras import callbacks
from keras.callbacks import CSVLogger
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import plot_tree
from xgboost.dask import dask
import dask.dataframe as dd
np.random.seed(0)



pandaObjects = []
##
def read_file(filename ):
    data = pd.read_csv(filename, nrows=1000000)
    #data = dd.from_pandas(data, npartitions=10)
    #data = dd.read_csv(filename)


    normalized_df = data.drop([' Destination IP',  'SimillarHTTP', 'Flow ID',
                               ' Source IP', ' Timestamp', ' Total Fwd Packets',
                               ' Total Backward Packets'
    ,'Total Length of Fwd Packets', ' Total Length of Bwd Packets',' Fwd Packet Length Std',
     ' Bwd Packet Length Mean',' Bwd Packet Length Std', ' Flow IAT Mean', ' Flow IAT Std',
     ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
     ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total',
    ' Bwd IAT Std' , ' Bwd IAT Min',
 ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
 ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
 ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
 ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
                               ' ECE Flag Count',
 ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
 ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
 ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
                               ' Init_Win_bytes_backward',
 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
 ' Idle Std', ' Idle Max', ' Idle Min', 'Unnamed: 0',' Bwd IAT Mean',' act_data_pkt_fwd',

                               ' Flow IAT Max',  ' Bwd IAT Max',   ' Fwd Packet Length Max',
                               ' Flow Duration',   ' Packet Length Mean', ' Avg Fwd Segment Size',
                               ' Fwd Packet Length Mean'], axis=1)

    normalized_df_updated = normalized_df.dropna(axis=0)
    pandaObjects.append(normalized_df_updated)

train_files =["C:/Users/Erfan/Desktop/01-12/DrDoS_DNS.csv"
    ,"C:/Users/Erfan/Desktop/01-12/DrDoS_LDAP.csv",
              "C:/Users/Erfan/Desktop/01-12/DrDoS_MSSQL.csv"
    ,"C:/Users/Erfan/Desktop/01-12/DrDoS_NetBIOS.csv",
    "C:/Users/Erfan/Desktop/01-12/DrDoS_NTP.csv"
    ,"C:/Users/Erfan/Desktop/01-12/DrDoS_SNMP.csv"
    ,"C:/Users/Erfan/Desktop/01-12/DrDoS_SSDP.csv",
    "C:/Users/Erfan/Desktop/01-12/DrDoS_UDP.csv",
    "C:/Users/Erfan/Desktop/01-12/Syn.csv",
    "C:/Users/Erfan/Desktop/01-12/TFTP.csv",
    "C:/Users/Erfan/Desktop/01-12/UDPLag.csv" ]
for f in train_files:
    read_file(f)
print(pd.concat(pandaObjects).shape)

clean1 = pd.concat(pandaObjects)
clean1 = clean1.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
clean1['Flow Bytes/s'] = clean1['Flow Bytes/s'].replace(to_replace="Infinity", value=100000)
clean1[' Flow Packets/s'] = clean1[' Flow Packets/s'].replace(to_replace="Infinity", value=500000)
clean1['Flow Bytes/s'] = pd.to_numeric(clean1['Flow Bytes/s'])
clean1[' Flow Packets/s'] = pd.to_numeric(clean1[' Flow Packets/s'])
clean3 = clean1[clean1[' Label'] != 'WebDDoS']

y = clean3[[' Label']]
y = y.rename(columns={" Label": "Label"})

y.Label.unique()
y['Label'] = y['Label'].replace('BENIGN', '0')
y['Label'] = y['Label'].replace('DrDoS_DNS', '1')
y['Label'] = y['Label'].replace('DrDoS_LDAP', '1')
y['Label'] = y['Label'].replace('DrDoS_MSSQL', '1')
y['Label'] = y['Label'].replace('DrDoS_NetBIOS', '1')
y['Label'] = y['Label'].replace('DrDoS_NTP', '1')
y['Label'] = y['Label'].replace('DrDoS_SNMP', '1')
y['Label'] = y['Label'].replace('DrDoS_SSDP', '1')
y['Label'] = y['Label'].replace('DrDoS_UDP', '1')
y['Label'] = y['Label'].replace('Syn', '1')
y['Label'] = y['Label'].replace('TFTP', '1')
y['Label'] = y['Label'].replace('UDP-lag', '1')
y['Label'] = y['Label'].astype('int')
print(y.Label.unique())

x = clean3.drop([' Label'], axis=1)
normalized_df1 = x







#print(normalized_df1.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalized_df1, y, test_size=0.2, random_state=0)
from imblearn.over_sampling import SMOTE

smt = SMOTE(k_neighbors=5, random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)


# first neural network with keras tutorial
from numpy import loadtxt
# KNeighborsClassifier
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
from sklearn import metrics, __all__, tree
from sklearn.metrics import f1_score
print('XGBoost Classifier')

print('Accuracy = ', metrics.accuracy_score(y_test, y_pred) * 100)
print("Confusion Matrix =\n", metrics.confusion_matrix(y_test, y_pred, labels=None,
                                                           sample_weight=None))
print("Recall =", metrics.recall_score(y_test, y_pred, labels=None,
                                           pos_label=1, average='weighted',
                                           sample_weight=None))
print("Classification Report =\n", metrics.classification_report(y_test, y_pred,
                                                                     labels=None,
                                                                     target_names=None,
                                                                     sample_weight=None,
                                                                     digits=2,
                                                                     output_dict=False))

print("F1 Score = ", f1_score(y_test, y_pred, average='macro'))
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix

from sklearn.feature_selection import RFE


#creating heatmap for determining correlation
#take 10 columns at a time
import seaborn as sn
corr_mat = normalized_df1.corr()
sn.heatmap(corr_mat, annot=True)

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 40))
cor = normalized_df1.corr()
sn.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

print(normalized_df1.columns.values)
print(normalized_df1.columns)

print(normalized_df1)
print(y)
print(normalized_df1.isnull().sum())
print(normalized_df1.isnull())
print(normalized_df1.info())


print(normalized_df1.shape)
print(y.shape)



if normalized_df1.shape[0] != y.shape[0]:
  print("X and y rows are mismatched, check dataset again")





#normalized_df1 = normalized_df1.to_dask_array(lengths=True)
#y = y.to_dask_array(lengths=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(normalized_df1, y, test_size=0.2, random_state=0)
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt

y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

print('xtrain={}, ytrain={}, xtest={}, ytest={}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix





def alg1():
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    anova_filter = SelectKBest(f_classif, k=3)
    clf = LinearSVC()
    anova_svm = make_pipeline(anova_filter, clf)
    anova_svm.fit(X_train, y_train)
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix

    y_pred = anova_svm.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("alg1")
    pass


def alg2():
    print("alg2")
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=normalized_df1.columns)
    feat_importances.nlargest(30).plot(kind='barh', figsize=(20, 40))
    plt.show()
    pass
    print("alg2")


def alg3():
    print("alg3")
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(y_train.shape)

    from sklearn import metrics, __all__, tree
    from sklearn.metrics import f1_score
    print('XGBoost Classifier')

    print('Accuracy = ', metrics.accuracy_score(y_test, y_pred) * 100)
    print("Confusion Matrix =\n", metrics.confusion_matrix(y_test, y_pred, labels=None,
                                                           sample_weight=None))
    print("Recall =", metrics.recall_score(y_test, y_pred, labels=None,
                                           pos_label=1, average='weighted',
                                           sample_weight=None))
    print("Classification Report =\n", metrics.classification_report(y_test, y_pred,
                                                                     labels=None,
                                                                     target_names=None,
                                                                     sample_weight=None,
                                                                     digits=2,
                                                                     output_dict=False))

    print("F1 Score = ", f1_score(y_test, y_pred, average='macro'))
    pass

def alg4():
    print("alg4")
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    yPred = nb.predict(X_test)
    print("Naive Bayes Performance Metrics")
    print("Accuracy Score: ", accuracy_score(y_test, yPred))
    print("Precision Score: ", precision_score(y_test, yPred))
    print("Recall Score: ", recall_score(y_test, yPred))
    print("F1 Score: ", f1_score(y_test, yPred))
    tn, fp, fn, tp = confusion_matrix(y_test, yPred).ravel()
    fpr = (fp / (fp + tn)) * 100
    print("False Positive Rate: " + str(fpr))
    print("Naive Bayes Confusion Matrix:")
    print("True Positives: " + str(tp))
    print("False Positives: " + str(fp))
    print("True Negatives: " + str(tn))
    print("False Negatives: " + str(fn))
    pass

def alg5():
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    ada.fit(X_train, y_train)
    yPred = ada.predict(X_test)
    print("AdaBoostClassifier Performance Metrics")
    print("Accuracy Score: ", accuracy_score(y_test, yPred))
    print("Precision Score: ", precision_score(y_test, yPred))
    print("Recall Score: ", recall_score(y_test, yPred))
    print("F1 Score: ", f1_score(y_test, yPred))
    tn, fp, fn, tp = confusion_matrix(y_test, yPred).ravel()
    fpr = (fp / (fp + tn)) * 100
    print("False Positive Rate: " + str(fpr))
    print("AdaBoostClassifier Confusion Matrix:")
    print("True Positives: " + str(tp))
    print("False Positives: " + str(fp))
    print("True Negatives: " + str(tn))
    print("False Negatives: " + str(fn))
    pass


def alg6():
    # CART
    cart = DecisionTreeClassifier()
    cart.fit(X_train, y_train)
    yPred = cart.predict(X_test)
    print("DecisionTreeClassifier Performance Metrics")
    print("Accuracy Score: ", accuracy_score(y_test, yPred))
    print("Precision Score: ", precision_score(y_test, yPred))
    print("Recall Score: ", recall_score(y_test, yPred))
    print("F1 Score: ", f1_score(y_test, yPred))
    tn, fp, fn, tp = confusion_matrix(y_test, yPred).ravel()
    fpr = (fp / (fp + tn)) * 100
    print("False Positive Rate: " + str(fpr))
    print("DecisionTreeClassifier Confusion Matrix:")
    print("True Positives: " + str(tp))
    print("False Positives: " + str(fp))
    print("True Negatives: " + str(tn))
    print("False Negatives: " + str(fn))
    pass

def alg7():
    # KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    yPred = neigh.predict(X_test)
    print("KNN Performance Metrics")
    print("Accuracy Score: ", accuracy_score(y_test, yPred))
    print("Precision Score: ", precision_score(y_test, yPred))
    print("Recall Score: ", recall_score(y_test, yPred))
    print("F1 Score: ", f1_score(y_test, yPred))
    tn, fp, fn, tp = confusion_matrix(y_test, yPred).ravel()
    fpr = (fp / (fp + tn)) * 100
    print("False Positive Rate: " + str(fpr))
    print("KNN Confusion Matrix:")
    print("True Positives: " + str(tp))
    print("False Positives: " + str(fp))
    print("True Negatives: " + str(tn))
    print("False Negatives: " + str(fn))
    pass

def alg8():
    # RandomForestClassifier
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import plot_tree

    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=None)
    rf.fit(X_train, y_train)
    yPred = rf.predict(X_test)
    print("Random Forest Performance Metrics")
    print("Accuracy Score: ", accuracy_score(y_test, yPred))
    print("Precision Score: ", precision_score(y_test, yPred))
    print("Recall Score: ", recall_score(y_test, yPred))
    print("F1 Score: ", f1_score(y_test, yPred))
    tn, fp, fn, tp = confusion_matrix(y_test, yPred).ravel()
    fpr = (fp / (fp + tn)) * 100
    print("False Positive Rate: " + str(fpr))
    print("Random Forest Confusion Matrix:")
    print("True Positives: " + str(tp))
    print("False Positives: " + str(fp))
    print("True Negatives: " + str(tn))
    print("False Negatives: " + str(fn))

    # Calculate the feature importances
    feat_importances = pd.Series(rf.feature_importances_, index=normalized_df1.columns)
    print(feat_importances.sort_values(ascending=False).head())

    # We feed in the X-matrix, the model and one feature at a time
    Inbound_ice_df = icebox.ice(data=normalized_df1, column=' Inbound',
                                predict=rf.predict)
    # Plot the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.figure(figsize=(15, 15))
    icebox.ice_plot(Inbound_ice_df, linewidth=.5, plot_pdp=True,
                    pdp_kwargs={'c': 'red', 'linewidth': 5}, ax=ax)
    ax.set_ylabel('Predicted diabetes')
    ax.set_xlabel(' Inbound')
    plt.show()

    yPred = rf.predict(X_test)
    cart = DecisionTreeClassifier(random_state=100)
    cart.fit(X_test, yPred)

    predictions = rf.predict(normalized_df1)
    dt = DecisionTreeClassifier(random_state=100)
    dt.fit(normalized_df1, predictions)

    # Now we can plot and see how the tree looks like.

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(cart, feature_names=list(normalized_df1.columns), precision=3,
              filled=True, fontsize=12, impurity=True)
    plt.show()
    pass






##############
os_data_X = pd.DataFrame(data=X_train_res, columns=columns)
os_data_y = pd.DataFrame(data=y_train_res, columns=['Label'])
cols=['Unnamed: 0', ' Source Port', ' Destination Port', ' Protocol',
       ' Flow Duration', ' Fwd Packet Length Max', ' Fwd Packet Length Min',
       ' Fwd Packet Length Mean','Flow Bytes/s', ' Flow Packets/s',
      ' Flow IAT Max', ' Bwd IAT Mean', ' Bwd IAT Max', 'Fwd PSH Flags', ' Min Packet Length',
      ' Max Packet Length',' Packet Length Mean', ' ACK Flag Count', ' Average Packet Size',
      ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' act_data_pkt_fwd', ' min_seg_size_forward',
      ' Inbound']
X = os_data_X[cols]
y = os_data_y['Label']

import statsmodels.api as sm
logit_model=sm.Logit(y, X)
result= logit_model.fit()
print(result.summary2())


# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(30,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train_res, y_train_res, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X_train_res, y_train_res)
print('Accuracy: %.2f' % (accuracy*100))








