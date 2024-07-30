#https://brage.bibsys.no/xmlui/bitstream/handle/11250/2568871/19478_FULLTEXT.pdf?sequence=1&isAllowed=y

#Libraries used
import pandas as pd
import numpy as np
import os
#Import Sklearn for mathematical computation
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import RANSACRegressor
#from sklearn.linear_model import SGDRegressor - Number 2 - MAE-22.7
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor # Best one - MAE－19.3
from xgboost import XGBRegressor
from xgboost import plot_importance 
import matplotlib.pyplot as plt

#linux -- DATA_PATH = "/home/kuljeet/Workspace/PredictiveAnalytics/TurboFan/CMAPSSData/"
DATA_PATH = "C:\\Kuljeet\\WorkSpace\\PredictiveAnalytics\\TurboFan\\CMAPSSData"

#assign column names
def assign_columns(dframe):
    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
    col_names = index_names + setting_names + sensor_names
    dframe.columns=col_names
    return dframe
    


#Function for accuracy results
def get_accuracy(labels,preds):
    return metrics.accuracy_score(labels,preds) * 100


#Function for removing unnamed columns
def remove_unnamed(df):
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

#Function for splitting txt-file into several csv-files containing a motor each
def split_data(file_folder):
    data_type = file_folder[0].split('_')[0]
    column1 = ['unit','cycle', 'setting1', 'setting2', 'setting3']
    column2 = ['sensor{}'.format(i) for i in range(1,22)]
    cols = column1+column2
    os.makedirs('{}_data'.format(data_type))
    for file in file_folder:
        folder = file.split('.')[0]
        print("creating folder data/{}".format(folder)) 
        os.makedirs('{}_data/{}'.format(data_type,folder))
    # process files
    for j,file_name in enumerate(file_folder):
        path = "{}".format(file_name)
        data = open(path).readlines()
        my_new_data = list(map(lambda line: line.strip('\n').split(' ')[:-2], data))
        for k in list(range(1,101)):
            new_array = list(filter(lambda x: x[0]=='{}'.format(k), my_new_data))
            df = pd.DataFrame(new_array, columns=cols)
            print("processing file {} folder {}_FD00{} mortor{} data" .format(file_name,data_type,j+1,k))
            df.to_csv('{}_data/{}_FD00{}/motor{}.csv'.format(data_type,data_type,j+1,k))
    return

def assign_class(rul_in):
    lbl_class= []
    for i in rul_in:
        if(i < 50) :
            lbl_class.append(0)
        elif (i < 100):
            lbl_class.append(1)
        else :
            lbl_class.append(2)     
    return lbl_class

def assign_bin_class(rul_in):
    lbl_class= []
    for i in rul_in:
        if(i < 20) :
            lbl_class.append(0)
        else :
            lbl_class.append(1)     
    return lbl_class

#Load RUL for test data
def load_rul_test(group_no):
    rul_path= 'RUL_FD00{}.txt'.format(group_no)
    rul_data = pd.read_csv(rul_path,header = None)
    return rul_data

#Loads data from path. Return Matrix X with shape[#examples, #features] and a list Y.
def load_data(path):
    #Reads the CSV file and removes unnamed columns
    data = remove_unnamed(pd.read_csv(path))
    #Makes a list with cycle numbers. From 1 to n cycles before failure
    Y = list(list(data['cycle']))
    #Reverse the list to make Y corresponding to RUL for each example in X
    Y.reverse()
    #Y_Class = assign_class(Y)
    Y_Class = assign_bin_class(Y)
    #print('Class assignments : {}'.format(Y_Class[0:5]))
    # Retrieve values from 'setting1' to 'setting21' only
    X = data.loc[:,'setting1':'sensor21'].values #depticated - .as_matrix()
    return X, Y , Y_Class

#Load training data for one motor. Return array X and Y in format [number of examples, 24], [number of examples, 1]
def load_motor_train(num_motor):
    #Create two arrays containing string names for train and test FD001
    training_folders = ['train_FD00{}'.format(i) for i in range(1,5)]
    cols = ['sensor{}'.format(i) for i in range(1,22)]
    #Folder number. Here: train_FD001 or test_FD001
    k = 0

    #Make an empty Y train and test list
    Y_train = []
#    Y_test = []

    #Load num_motor X and Y training- and test-sets
#    for i in range (1, num_motor+1): 
    i=num_motor
    #Create training and test path
    train_path = 'train_data/{}/motor{}.csv'.format(training_folders[k],i)
    #print(train_path)
    #Load temporary X and Y sets
    X_train, Y_train, L_train = load_data(train_path)
    #Convert Matrix X to a pandas dataframe
    X_train_df = pd.DataFrame(X_train)
    #Only keep Sensors data -- KS 
    X_train_df = X_train_df.drop(X_train_df.columns[0:3],axis=1)
    X_train_df.columns=cols
    #print('Selecting only Sensor data {}'.format(X_train_df.head()))
    #Convert X and Y to NumPy arrays
    X_train = X_train_df.values
    Y_train = np.array(Y_train)
    X_train = standardize(X_train)
    return X_train, Y_train, L_train 

#Load test data for one motor. Return array X and Y in format [number of examples, 24], [number of examples, 1]
def load_motor_test(group_no, num_motor):
    #Create two arrays containing string names for train and test FD001
    test_folders = ['test_FD00{}'.format(i) for i in range(1,5)]
    cols = ['sensor{}'.format(i) for i in range(1,22)]
    #Folder number. Here: train_FD001 or test_FD001
    k = group_no
    rul_data = load_rul_test(k)
    #Make an empty Y train and test list
    Y_test = []
    #Load num_motor X and Y training- and test-sets
    i=num_motor
    #Create training and test path
    test_path = 'test_data/{}/motor{}.csv'.format(test_folders[k-1],i)
    print(test_path)
    #Load temporary X and Y sets
    X_test, Y_test, L_test = load_data(test_path)
    #Convert Matrix X to a pandas dataframe
    X_test_df = pd.DataFrame(X_test)
    #Only keep Sensors data -- KS 
    X_test_df = X_test_df.drop(X_test_df.columns[0:3],axis=1)
    X_test_df.columns = cols
    #Convert X and Y to NumPy arrays
    X_test = X_test_df.values
    Y_test = np.array(Y_test)
    rul_data = np.array(rul_data)
    #adding rul to get the actual cycles remaining in the test data
    Y_test = Y_test + rul_data[i-1]
    X_test = standardize(X_test)
    return X_test, Y_test , L_test


#Standardscaler 
# Option 1 : fit_transform every one
# Option 2 : fit_tranform first only others transform
def standardize(data):
    d = preprocessing.StandardScaler().fit_transform(data)
    return d

def process_data_files():
    #Array containing the string names of train and test txt-files
    print('#',end='')
    training_files = ['train_FD00{}.txt'.format(k) for k in range(1,5)]
    test_files = ['test_FD00{}.txt'.format(k) for k in range(1,5)]
    #Run through test and training txt-files to split data
    for file_folder in [training_files,test_files]:
#        if(os.path.isdir('train_data') == False):
            split_data(file_folder)
 #       else :
#            print('folder exists : {}',format(file_folder)) 
    return 

def get_std_deviation(data_in):
    df_std = pd.DataFrame(data_in).std()
#    print('Standard Dev= {}'.format(df_std==0))
    return df_std

def get_correlation(data_in,disp):
    corr = data_in.corr()
    print('Correlation : {}'.format(corr))
    if(disp== True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr,cmap='coolwarm',vmin=-1,vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,len(data_in.columns),1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(data_in.columns)
        ax.set_yticklabels(data_in.columns)
    # plt.ion()
        plt.show()
    return corr

def show_clusters(data_in,pred_in,cluster_in):
    print('data shape : {}'.format(data_in.shape))
    print('pred shape : {}'.format(pred_in.shape))
    print('data in : {}'.format(data_in[0:5,]))
    plt.scatter(data_in[:,0],data_in[:,1],c=pred_in,cmap='rainbow')
    plt.scatter(cluster_in[:,0],cluster_in[:,1],c='black',s=200,alpha=0.5)
    plt.show()
    return

def print_error_analysis(label_ts,label_prd):
    # Error Analysis
    print('Mean Absolute Error:',metrics.mean_absolute_error(label_ts,label_prd))
    print('Mean Squared Error:',metrics.mean_squared_error(label_ts,label_prd))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(label_ts,label_prd)))
    print('Median pridicted RUL : {}'.format(np.median(label_prd)))
    print('Median actual RUL : {}'.format(np.median(label_ts)))
    print('Mean pridicted RUL : {}'.format(np.mean(label_prd)))
    print('Mean actual RUL : {}'.format(np.mean(label_ts)))
#    print('Accuracy Score : {}'.format(get_accuracy(label_ts,label_prd)))
    return

def plot_predictions(y_test,y_pred):
    fig = plt.figure()
    plt.plot(y_pred,c='red',label='prediction')
    plt.plot(y_test,c='blue',label='label')
    plt.legend(loc='upper right')
    plt.suptitle('RUL Prediction')
    plt.show()
    return    

def main():
    os.chdir(DATA_PATH)
    train_count = 100
    with_pca = False
    test_motor = 49
    group_no = 1
    epochs=1000
    

    droplist = []
    droplistcorel = []
    for i in range(1,train_count):
        k=0
        data_tr_tmp, label_tr_tmp, class_tr_tmp = load_motor_train(i)
        if (i==1):
            print('Motor %d after Standardization \n{}'.format(i,data_tr_tmp[0:5,:]))
            for j in get_std_deviation(data_tr_tmp):
                k = k+1
                if(j==0):
                    droplist.append(k-1)
            print('Drop Columns [{}] with std_dev = 0 : \n'.format(droplist))

        if i==1 :
            data_tr = pd.DataFrame(data_tr_tmp)
            label_tr = pd.DataFrame(label_tr_tmp)
            class_tr = pd.DataFrame(class_tr_tmp)
        else:
            data_tr = pd.concat([data_tr,pd.DataFrame(data_tr_tmp)])
            label_tr = pd.concat([label_tr,pd.DataFrame(label_tr_tmp)])
            class_tr = pd.concat([class_tr,pd.DataFrame(class_tr_tmp)])
            
    
    # drop columns with Std_Dev = 0 from test data     
    print('Before Std_Dev Drop')
    print('Shape of training {}'.format(data_tr.shape))
    print(data_tr.head())
    data_tr = data_tr.drop(data_tr.columns[droplist], axis=1)

    print('Before Correl Drop')
    print('Shape of training {}'.format(data_tr.shape))
    print(data_tr.head())
    corel = get_correlation(data_tr,False)
    
    for i in range(1,len(corel.columns)-1):
        if (corel.iloc[0,i] > 0.75 or corel.iloc[0,i] < -0.75):
            droplistcorel.append(i)

    print('Drop Columns [{}] with high correlation > 0.75 or < -0.75 : \n'.format(droplistcorel))
    data_tr = data_tr.drop(data_tr.columns[droplistcorel], axis=1)
    print('After Correl Drop')
    print(data_tr.head())
    print(label_tr.head())

    label_tr=np.ravel(label_tr)
    class_tr=np.ravel(class_tr)


    print('Shape of training {}'.format( data_tr.shape))
    print('Shape of training {}'.format( label_tr.shape))
    
    # Create Regressor Model
#    model_reg = AdaBoostRegressor(random_state=0,n_estimators=100,learning_rate=0.5)
    model_reg = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=10,
                             random_state=13,n_jobs=-1)

    print('Training'),
#    for e in range(1,epochs):
    print('.'),
    model_reg.fit(data_tr,label_tr)
    print('.')

    print('Prediction'),

    predict_all=False
    if not predict_all :
        data_ts_tmp, label_ts, class_ts = load_motor_test(group_no,test_motor)
        data_ts = pd.DataFrame(data_ts_tmp)
        data_ts = data_ts.drop(data_ts.columns[droplist], axis=1)
        data_ts = data_ts.drop(data_ts.columns[droplistcorel], axis=1)
        print('Shape of test{}'.format(data_ts_tmp.shape))
        label_prd = model_reg.predict(data_ts)
        print('Pridiction for Test Motor {}'.format(test_motor))
        print(label_ts)
        print(label_prd)
        print('test RMSE:',np.sqrt(metrics.mean_squared_error(label_ts, label_prd)))
        print( 'Error Analysis :')
        print_error_analysis(label_ts, label_prd)
        print('Plot predictions ..')
        plot_predictions(label_ts, label_prd)
    else :
        rul_pred =[]
        rul_test =[]
        num_test = 100
        avg_win_len = 30
        for test_motor in range(1,num_test):
            data_ts_tmp, label_ts, class_ts = load_motor_test(group_no,test_motor)
            data_ts = pd.DataFrame(data_ts_tmp)
            data_ts = data_ts.drop(data_ts.columns[droplist], axis=1)
            data_ts = data_ts.drop(data_ts.columns[droplistcorel], axis=1)
            label_prd = model_reg.predict(data_ts)
            avg_pred = np.average(label_prd[-avg_win_len:])
            avg_test = np.average(label_ts[-avg_win_len:])
            rul_pred.append(avg_pred)
            rul_test.append(avg_test)
        print( 'Error Analysis :')
        print_error_analysis(rul_test, rul_pred)
        print('Plot predictions ..')
        plot_predictions(rul_test, rul_pred)
    return
'''    
    plt.figure(1)

    plt.subplot(221)
    plt.plot(class_ts)
    plt.ylabel('Class')
    plt.xlabel('cycles')

    plt.subplot(222)
    plt.plot(class_prd)
    
    plt.plot(pd.rolling_median(class_prd,10))
    plt.ylabel('Predicted RUL')
    plt.xlabel('cycles')
'''

if __name__ == "__main__" :
    main()    