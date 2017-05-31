#! /usr/bin/python

import matplotlib
matplotlib.use("TkAgg")
from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef, zero_one_loss, log_loss
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn import preprocessing, decomposition
from root_pandas import read_root
# tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.contrib.learn import DNNClassifier, infer_real_valued_columns_from_input, SKCompat, extract_pandas_data


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
    df_sig_train = df_sig.head(15000)
    df_sig_test = df_sig.tail(3000)

    df_bkg_train = df_bkg.head(15000)
    df_bkg_test = df_bkg.tail(3000)

    # define training and test dataframe
    df_train = df_make_sample(df_sig_train, df_bkg_train).sample(frac=1)
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

    import tensorflow as tf
    import numpy as np
    from keras.utils import np_utils
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import preprocessing,metrics

    #--------------------------------------------------
    x_train = extract_pandas_data(x_train)
    y_train = y_train.values
    x_test = extract_pandas_data(x_test)
    x_test_B = extract_pandas_data(x_test_B)
    x_test_S = extract_pandas_data(x_test_S)
    y_test = y_test.values

    with tf.Session() as sess:
        layers = 4
        w_vector = [0 for _ in range(layers)]

        def add_layer(lay_n, inputs,input_dim,output_dim,activation=None,drop_out=False,keep_prob=0.5):
            Weights=tf.Variable(tf.random_normal([input_dim,output_dim]))
            w_vector[lay_n-1] = Weights

            biases=tf.Variable(tf.zeros([1,output_dim])+0.1)
            Wx_plus_b=tf.matmul(inputs, Weights)+biases
            if activation ==None:
                outputs = Wx_plus_b
            else:
                outputs = activation(Wx_plus_b)
            if drop_out:
                outputs = tf.nn.dropout(outputs,keep_prob)

            return outputs

        xs=tf.placeholder(tf.float32,[None,11])
        ys=tf.placeholder(tf.float32,[None,])

        #Create graphic
        layer1=add_layer(1,xs,11,30,activation=tf.tanh,drop_out=False)
        layer2=add_layer(2,layer1,30,100,activation=tf.tanh,drop_out=True, keep_prob=0.5)
        layer3=add_layer(3,layer2,100,30,activation=tf.tanh,drop_out=True, keep_prob=0.5)
        prediction=add_layer(4,layer3,30,1,activation=tf.nn.sigmoid)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.expand_dims(ys,1)*tf.log(prediction+1e-5)+
                                                      (1-tf.expand_dims(ys,1))*tf.log(1-prediction+1e-5),
                                                      reduction_indices=[1]))
        #cross_entropy = tf.losses.log_loss(tf.expand_dims(ys,1),prediction,epsilon=1e-05)

        l2_reg_value = 0.01

	cross_entropy = tf.reduce_mean(cross_entropy + l2_reg_value * (tf.nn.l2_loss(w_vector[0])+
                                                                       tf.nn.l2_loss(w_vector[1])+
                                                                       tf.nn.l2_loss(w_vector[2])+
                                                                       tf.nn.l2_loss(w_vector[3])))

        global_step = tf.Variable(0, trainable=False)
        increment_global_step_op = tf.assign(global_step, global_step+1)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   10, 0.96, staircase=True)

        #train_step = tf.train.ProximalAdagradOptimizer(1e-1).minimize(cross_entropy)
        #train_step2 = tf.train.ProximalAdagradOptimizer(1e-3,l2_regularization_strength=0.0001).minimize(cross_entropy)
        #train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cross_entropy)
        #train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
        #train_step = tf.train.ProximalAdagradOptimizer(1e-1,l2_regularization_strength=0.001).minimize(cross_entropy)
        train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cross_entropy)
        #train_step2 = tf.train.MomentumOptimizer(1e-3,0.9).minimize(cross_entropy)
        #train_step = tf.train.ProximalAdagradOptimizer(1e-4,l2_regularization_strength=0.01).minimize(cross_entropy)
        #train_step = tf.train.AdamOptimizer(1e-1).minimize(cross_entropy)
        #train_step = tf.train.RMSPropOptimizer(1e-3).minimize(cross_entropy)

        sess.run(tf.global_variables_initializer())
        batches = np.split(x_train,500)
        labels = np.split(y_train,500)
        steps = 1000

        with sess.as_default():
            for i in range(steps):
                for batch in zip(batches,labels):
                    train_step.run(feed_dict={xs: batch[0],
                                              ys: batch[1]
                                              }
                                   )

                sess.run(increment_global_step_op)
                if not i%10:
                    print "Step", i, "out of", steps
                    print "Cross_entropy:", learning_rate.eval(), cross_entropy.eval(feed_dict={xs: x_train, ys: y_train}), \
                        cross_entropy.eval(feed_dict={xs: x_test, ys: y_test})
            '''
            for i in range(steps):
                for batch in zip(batches,labels):
                    train_step2.run(feed_dict={xs: batch[0],
                                               ys: batch[1]
                                               }
                                    )
                if not i%10:
                    print "Step", i, "out of", steps
                    print "Cross_entropy2:", cross_entropy.eval(feed_dict={xs: x_train, ys: y_train}), \
                        cross_entropy.eval(feed_dict={xs: x_test, ys: y_test})
            '''
            a = ys.eval(feed_dict={xs: x_test,
                           ys: y_test})

            b = tf.squeeze(prediction.eval(feed_dict={xs: x_test,
                                                      ys: y_test}
                                           )
                           ).eval().round()

            c = tf.squeeze(prediction.eval(feed_dict={xs: x_test,
                                                      ys: y_test}
                                           )
                           ).eval()

            d = tf.squeeze(prediction.eval(feed_dict={xs: x_test_S,
                                                      ys: y_test_S}
                                           )
                           ).eval()

            #print a,b
            score = accuracy_score(a, b)
            fpr, tpr, _ = roc_curve(a,c)
            roc = roc_auc_score(a,c)            
            print "0-1 loss:", zero_one_loss(a, b)
            print "Cross_entropy loss:", cross_entropy.eval(feed_dict={xs: x_test,
                                             ys: y_test})
            print "Matthews CorrCoef: ", matthews_corrcoef(a, b)
            print 8*"-"
            print "Average precision score:",average_precision_score(a, c)
            print "ROC AUC: ", roc
            print "Score:", score
            
            plt.figure()
            lw = 2
            plt.plot(tpr, 1-fpr, color='darkorange',
             lw=lw, label='ROC curve (area =  %0.2f)' % roc)
            plt.plot([0.9, 0.9], [0., 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Signal Eff.')
            plt.ylabel('Bkg Rej.')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

            plt.figure()
            bins = 100
            plt.hist(tpr, bins=bins, alpha=0.5, range=(0.,1.))
            plt.hist(fpr, bins=bins, alpha=0.5, range=(0.,1.))
            plt.show()

            import sys
            sys.exit(0)
