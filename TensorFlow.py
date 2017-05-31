#! /usr/bin/python

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef, zero_one_loss, log_loss
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn import preprocessing, decomposition

import numpy as np
from root_pandas import read_root

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.contrib.learn import DNNClassifier, infer_real_valued_columns_from_input, extract_pandas_data


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
    df_train = df_make_sample(df_sig_train, df_bkg_train)
    df_test = df_make_sample(df_sig_test, df_bkg_test)

    # Separate label from features
    x_train, y_train = get_columns(df_train, FEATURES, 'y')
    x_test, y_test = get_columns(df_test, FEATURES, 'y')
    x_train.info()

    x_test_S = df_test[df_test['y']==1][FEATURES]
    x_test_B = df_test[df_test['y']==0][FEATURES]
    y_test_S = df_test[df_test['y']==1]['y']
    y_test_B = df_test[df_test['y']==0]['y']

    # prescale
    scaler = preprocessing.StandardScaler().fit(x_train.values)
    x_train = scaler.transform(x_train.values)
    x_test = scaler.transform(x_test.values)

    ### Decorralate
    pca = decomposition.PCA(n_components = 'mle', whiten = True)

    pca.fit(x_train)

    pca.transform(x_train)
    pca.transform(x_test)

    #pd.DataFrame.hist( pd.DataFrame(data=x_test,columns=FEATURES), figsize = [11,11]);    
    #plt.show()
    # Extract columns from features
    feature_columns = infer_real_valued_columns_from_input(x_train)

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            x_test,
            y_test.values,
            every_n_steps=50)

    # Book classifier
    classifier = DNNClassifier(feature_columns=feature_columns, 
                               model_dir='model',
                               hidden_units=[30, 100, 100, 100, 100, 100, 100, 30],
                               dropout=0.1,
                               activation_fn=tf.tanh,
                               config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1),
                               optimizer=tf.train.ProximalAdagradOptimizer(
                                                                           learning_rate=0.1,
                                                                            l2_regularization_strength=0.0001,
                                                                           )
                               )

    def get_inputs_train():
        """
        define training input function
        """
        x = tf.constant(extract_pandas_data(x_train))
        y = tf.constant(y_train.values)
        return x, y

    def get_inputs_test():
        """
        define test input function
        """
        x = tf.constant(extract_pandas_data(x_test))
        y = tf.constant(y_test.values)
        return x, y

    def get_inputs_test_S():
        """
        define test input function for signal only
        """
        x = tf.constant(extract_pandas_data(x_test_S))
        y = tf.constant(y_test_S.values)
        return x, y

    def get_inputs_test_B():
        """
        define test input function for background only
        """
        x = tf.constant(extract_pandas_data(x_test_S))
        y = tf.constant(y_test_B.values)
        return x, y

    # Fit model.
    classifier.fit(input_fn=get_inputs_train, steps=5000, monitors=[validation_monitor])

    # Evaluate model
    evaluation = classifier.evaluate(input_fn=get_inputs_test, steps=1)

    # Get predictions and probabilities
    predictions = classifier.predict_classes(input_fn=get_inputs_test)

    proba = classifier.predict_proba(input_fn=get_inputs_test)
    proba_train = classifier.predict_proba(input_fn=get_inputs_train)
    proba_S = classifier.predict_proba(input_fn=get_inputs_test_S)
    proba_B = classifier.predict_proba(input_fn=get_inputs_test_B)
    proba_g = proba_to_list(proba)
    proba_g_train = proba_to_list(proba_train)
    proba_g_S = proba_to_list(proba_S)
    proba_g_B = proba_to_list(proba_B)
    
    ax = sns.distplot(np.array(proba_g_S)[:, 1])
    sns.distplot(np.array(proba_g_B)[:, 0], ax=ax)
    plt.show()

    y_pred = list(predictions)
    y_pred_g = []
    for y_ in y_pred:
        y_pred_g.append(y_.tolist())
    y_pred = np.array(y_pred_g)
    score = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, np.array(proba_g)[:, 1])

    fpr, tpr, _ = roc_curve(np.array(list(y_test.values)), np.array(proba_g)[:, 1])
    fpr_t, tpr_t, _ = roc_curve(np.array(list(y_train.values)), np.array(proba_g_train)[:, 1])

    # Classifier statistics
    print "0-1 loss:", zero_one_loss(y_test, y_pred)
    print "log loss:",log_loss(y_test, y_pred)
    print "Matthews CorrCoef: ", matthews_corrcoef(y_test, y_pred)
    print 8*"-"
    print "Average precision score:",average_precision_score(np.array(list(y_test.values)), np.array(proba_g)[:, 1])
    print "ROC AUC: ", roc
    print "Score:", score

    plt.figure()
    lw = 2
    plt.plot(tpr, 1-fpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc)
    plt.plot([0.9, 0.9], [0., 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Signal Eff.')
    plt.ylabel('Bkg Rej.')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    '''
    plt.figure()
    bins = 40
    plt.hist(tpr, bins=bins, alpha=0.5, range=(0.,1.))
    plt.hist(tpr_t, bins=bins, alpha=0.5, range=(0.,1.))
    plt.hist(fpr, bins=bins, alpha=0.5, range=(0.,1.))
    plt.hist(fpr_t, bins=bins, alpha=0.5, range=(0.,1.))
    plt.show()
    '''
