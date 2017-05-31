    #! /usr/bin/python

import matplotlib
matplotlib.use("TkAgg")

import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, matthews_corrcoef, zero_one_loss, log_loss
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn import preprocessing, decomposition

from root_pandas import read_root

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.contrib.learn import infer_real_valued_columns_from_input, extract_pandas_data


def df_make_sample(df_sig, df_bkg):
    """
    
    :param df_sig: 
    :param df_bkg: 
    :return: 
    """
    df_train = df_sig.append(df_bkg)
    return df_train


def get_columns(df_, features=[], label=""):
    """
    
    :param df_: 
    :param features: 
    :param label: 
    :return: 
    """
    x_ = df_[features]
    y_ = df_[label]
    return x_, y_


def proba_to_list(proba):
    """
    
    :param proba: 
    :return: 
    """
    proba_g = []
    for pr in proba:
        proba_g.append(pr.tolist())
    return proba_g


if __name__ == "__main__":

    # Feature selection
    FEATURES = ['L_abs',
                    'jet_discrim', 
                    'delta_M', 
                    'cos_phi', 
                    'jet_pt', 
                    'K_lnchi2_SV', 
                    'Pi_lnchi2_SV',
                    'L_z', 
                    'L_xy', 
                    'DS_pt', 
                    'DS_eta',
                    ]
        
    # Get data to pandas dataframe
    df_sig = read_root('DS_MC_arr_chi2.root')
    df_bkg = read_root('BtoDS_MC_arr_chi2.root')

    df_sig['y'] = 1
    df_bkg['y'] = 0

    # Resize samples
    df_sig_train = df_sig.head(9000)
    df_sig_test = df_sig.tail(9000)

    df_bkg_train = df_bkg.head(9000)
    df_bkg_test = df_bkg.tail(9000)

    # define training and test dataframe
    df_train = df_make_sample(df_sig_train, df_bkg_train).sample(frac=1)
    df_test = df_make_sample(df_sig_test, df_bkg_test)

    # Separate label from features
    x_train, y_train = get_columns(df_train, FEATURES, 'y')
    x_test, y_test = get_columns(df_test, FEATURES, 'y')
    x_train.info()

    x_test_S = df_test[FEATURES][df_test['y']==1]
    x_test_B = df_test[FEATURES][df_test['y']==0]
    y_test_S = df_test['y'][df_test['y']==1]
    y_test_B = df_test['y'][df_test['y']==0]

    # prescale
    scaler = preprocessing.StandardScaler().fit(x_train.values)
    x_train = scaler.transform(x_train.values)
    x_test = scaler.transform(x_test.values)
    x_test_B = scaler.transform(x_test_B.values)
    x_test_S = scaler.transform(x_test_S.values)

    ### Decorralate
    pca = decomposition.PCA(n_components = 'mle', whiten = True)

    pca.fit(x_train)

    pca.transform(x_train)
    pca.transform(x_test)
    pca.transform(x_test_S)
    pca.transform(x_test_B)

    #pd.DataFrame.hist( pd.DataFrame(data=x_test,columns=FEATURES), figsize = [11,11]);    
    #plt.show()

    # Extract columns from features
    feature_columns = infer_real_valued_columns_from_input(x_train)

    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt

    #--------------------------------------------------
    x_train = extract_pandas_data(x_train)
    y_train = y_train.values
    x_test = extract_pandas_data(x_test)
    x_test_B = extract_pandas_data(x_test_B)
    x_test_S = extract_pandas_data(x_test_S)
    y_test = y_test.values

    # from keras import regularizers
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras import backend as K
    print K.learning_phase()

    model = Sequential()
    # model.add(Dropout(0.1, input_shape=(11,)))
    model.add(Dense(30, activation='tanh', input_dim=11))
    model.add(Dropout(0.))

    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.1))

    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.1))

    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.1))

    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(30, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    from keras.optimizers import SGD

    sgd = SGD(lr=0.1, decay=1e-5, momentum=0.3, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, batch_size=48, epochs=50, validation_data=(x_test, y_test))

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, batch_size=48, epochs=50,validation_data=(x_test, y_test))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.3, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, batch_size=32, epochs=50,validation_data=(x_test, y_test))

    y_pred = model.predict(x_test)

    score = accuracy_score(y_test, y_pred.round())
    print "Accuracy score:", score  
   
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)            

    print "0-1 loss:", zero_one_loss(y_test, y_pred.round())
    print "Matthews CorrCoef: ", matthews_corrcoef(y_test, y_pred.round())
    print 8*"-"
    print "Average precision score:",average_precision_score(y_test, y_pred)
    print "ROC AUC: ", roc
    print "Score:", score

    y_pred_S = model.predict(x_test_S)
    y_pred_B = model.predict(x_test_B)

    count_S = 0
    count_B = 0 
    for item in zip(np.squeeze(y_pred_S).tolist(), np.squeeze(y_pred_B).tolist()):
        if item[0] > 0.5:
            count_S += 1
        if item[1] > 0.5:
            count_B += 1        

    print "Precision:",count_S*1./(count_S+count_B)

    contents_B = np.array(y_pred_B)
    contents_S = np.array(y_pred_S)

    sns.distplot(contents_S, kde=False, rug=False, norm_hist=True) 
    sns.distplot(contents_B, kde=False, rug=False, norm_hist=True)
    plt.xlabel('DNN response')

    plt.figure()
    lw = 2
    plt.plot(tpr, 1-fpr, color='darkorange',
             lw=lw, label='ROC curve (area =  %0.2f)' % roc)
    plt.plot([0.9, 0.9], [0., 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.2, 1.05])
    plt.xlabel('Signal Eff.')
    plt.ylabel('Bkg Rej.')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.figure()
    bins = 100
    plt.plot(thresholds, tpr, 'r--', thresholds, fpr, 'b--')
    plt.xlim([0.0, 1.0])
    plt.xlabel('DNN response')
    plt.show()

