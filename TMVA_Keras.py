#!/usr/bin/env python
 
from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile
 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
output = TFile.Open('TMVA.root', 'RECREATE')
factory = TMVA.Factory('TMVAClassification', output,
                       #'!V:!Silent:Color:DrawProgressBar:Transformations=N,P,G,P,G:AnalysisType=Classification')
                       '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')
 
data_S = TFile.Open('DS_MC_arr_chi2.root')
data_B = TFile.Open('BtoDS_MC_arr_chi2.root')

signal = data_S.Get('Analysis')
background = data_B.Get('Analysis')

dataloader = TMVA.DataLoader('dataset')
dataloader.AddVariable( "cos_phi", 'D' );
dataloader.AddVariable( "L_abs", 'D' );
dataloader.AddVariable("L_z", 'D');
dataloader.AddVariable("L_xy", 'D');
dataloader.AddVariable("DS_pt", 'D');
dataloader.AddVariable("DS_eta", 'D');
dataloader.AddVariable("jet_pt", 'D');
dataloader.AddVariable("jet_discrim", 'D');
dataloader.AddVariable("K_lnchi2_SV", 'D');
dataloader.AddVariable("Pi_lnchi2_SV", 'D');
dataloader.AddVariable("delta_M", 'D');


#for branch in signal.GetListOfBranches():
#     dataloader.AddVariable(branch.GetName())
 
dataloader.AddSignalTree(signal, 1.0)
dataloader.AddBackgroundTree(background, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''), 'nTrain_Signal=15000:nTrain_Background=15000:nTest_Signal=3000:nTest_Background=3000:SplitMode=Random:NormMode=NumEvents:!V')
# Generate model

# Define initialization
# Define model
model = Sequential()
model.add(Dense(1000, activation='tanh', W_regularizer=l2(1e-5), input_dim=11))
model.add(Dropout(0.5))

model.add(Dense(1000, activation='tanh', W_regularizer=l2(1e-5)))
model.add(Dropout(0.5))

model.add(Dense(500, activation='tanh', W_regularizer=l2(1e-5)))
model.add(Dropout(0.1))

model.add(Dense(100, activation='tanh', W_regularizer=l2(1e-5)))
model.add(Dropout(0.))

#model.add(Dense(32, init=normal, activation='relu', W_regularizer=l2(1e-5)))
model.add(Dense(2, activation='sigmoid'))

# Set loss and optimizer
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1,decay=1e-4, momentum=0.3), metrics=['accuracy',])

# Store model to file
model.save('model.h5')
model.summary()

# Book methods
factory.BookMethod(dataloader, TMVA.Types.kFisher, 'Fisher',
                   '!H:!V:VarTransform=D,G:Fisher')

factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDT","H:!V:VarTransform=N,D,G:MinNodeSize=1:NTrees=300:MaxDepth=1:BoostType=AdaBoost:AdaBoostBeta=1:SeparationType=GiniIndex:nCuts=200" );

factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'DNNKeras',
                   'H:!V:VarTransform=N,P,G:FilenameModel=model.h5:NumEpochs=30:BatchSize=32')
 
# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

