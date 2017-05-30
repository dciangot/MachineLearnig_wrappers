#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#include "TMVA/DataLoader.h"
//gSystem->Load("libFWCoreFWLite");
//   AutoLibraryLoader::enable();

using namespace std;
//using namespace edm;
using namespace TMVA;

void Training(){
TMVA::Tools::Instance();

//string file_DS="DS_MC_arr_tree.root";
//string file_BtoDS="BtoDS_MC_arr_tree.root";
string file_DS="DS_MC_arr_chi2.root";
string file_BtoDS="BtoDS_MC_arr_chi2.root";
 

TString outfileName( "Training.root" );

TFile* outputFile = TFile::Open( outfileName, "RECREATE" );


TMVA::Factory *factory = new TMVA::Factory( "cuts", outputFile,
//                                               "V:!Silent:Color:DrawProgressBar:Transformations=N:AnalysisType=Classification" );
                                               "V:!Silent:Color:DrawProgressBar:Transformations=N,D,G,D,G,N,D:AnalysisType=Classification" );

//                                               "V:!Silent:Color:DrawProgressBar:Transformations=I;D,D:AnalysisType=Classification" );


TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");

dataloader->AddVariable( "cos_phi", 'D' );
dataloader->AddVariable( "L_abs", 'D' );
dataloader->AddVariable("L_z", 'D');
dataloader->AddVariable("L_xy", 'D');
dataloader->AddVariable("DS_pt", 'D');
dataloader->AddVariable("DS_eta", 'D');
dataloader->AddVariable("jet_pt", 'D');
dataloader->AddVariable("jet_discrim", 'D');
dataloader->AddVariable("K_lnchi2_SV", 'D');
dataloader->AddVariable("Pi_lnchi2_SV", 'D');
dataloader->AddVariable("delta_M", 'D');

//dataloader->AddVariable("deltaR_min", 'D');
//dataloader->AddVariable("L_sigma", 'D');
//dataloader->AddVariable("D0_phi", 'D');
/*dataloader->AddVariable("K_pt", 'D');
dataloader->AddVariable("K_eta", 'D');
dataloader->AddVariable("K_phi", 'D');
dataloader->AddVariable("Pi_pt", 'D');
dataloader->AddVariable("Pi_eta", 'D');
dataloader->AddVariable("Pi_phi", 'D');
dataloader->AddVariable("Pis_pt", 'D');
dataloader->AddVariable("Pis_eta", 'D');
dataloader->AddVariable("Pis_phi", 'D');
dataloader->AddVariable("K_chi2_PV", 'D');
dataloader->AddVariable("K_chi2_SV", 'D');
dataloader->AddVariable("Pi_chi2_PV", 'D');
dataloader->AddVariable("Pi_chi2_SV", 'D');*/
//dataloader->AddVariable("jet_eta", 'D');
//dataloader->AddVariable("jet_phi", 'D');
//dataloader->AddVariable("K_chi2_PV", 'D');
//dataloader->AddVariable("K_chi2_SV", 'D');
//dataloader->AddVariable("K_lnchi2_PV", 'D');
//dataloader->AddVariable("Pi_chi2_PV", 'D');
//dataloader->AddVariable("Pi_chi2_SV", 'D');
//dataloader->AddVariable("Pi_lnchi2_PV", 'D');

TFile *input = TFile::Open( file_DS.c_str() );
TFile *input_test = TFile::Open( file_BtoDS.c_str() );

   std::cout << "--- TMVAClassification       : Using input file: " << input->GetName() << std::endl;

TTree *signal     = (TTree*)input->Get("Analysis");
TTree *background     = (TTree*)input_test->Get("Analysis");

cout << signal->GetEntries() << endl;

Double_t signalWeight     = 1.0;
Double_t backgroundWeight     = 1.0;
// You can add an arbitrary number of signal or background trees
dataloader->AddSignalTree    ( signal,     signalWeight );
dataloader->AddBackgroundTree( background, backgroundWeight );
//factory->SetBackgroundWeightExpression( "weight" );

// Apply additional cuts on the signal and background samples (can be different)
TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

// Tell the factory how to use the training and testing events
//

dataloader->PrepareTrainingAndTestTree( mycuts, "nTrain_Signal=9000:nTest_Signal=9000:nTrain_Background=9000:nTest_Background=9000:SplitMode=random:!V" );

// To also specify the number of testing events, use:
// dataloader->PrepareTrainingAndTestTree( mycuts,
//                                         "NSigTrain=9000:NBkgTrain=9000:NSigTest=9000:NBkgTest=9000:SplitMode=Random:!V" );
//dataloader->PrepareTrainingAndTestTree( mycuts, mycutb,
//                                        "nTrain_Signal=15000:nTest_Signal=15000:nTrain_Background=15000:nTest_Background=3000:SplitMode=Random:NormMode=NumEvents:!V" );

factory->BookMethod(dataloader, Types::kFisher, "Fisher", "H:!V" );

factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT_allv","H:!V:MinNodeSize=1:NTrees=600:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=1:SeparationType=GiniIndex:nCuts=100" );

//factory->BookMethod(dataloader, TMVA::Types::kKNN, "KNN1", "H:nkNN=300:ScaleFrac=0.5:SigmaFact=1.5:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" ); 
//factory->BookMethod(dataloader, TMVA::Types::kKNN, "KNN2", "H:nkNN=600:ScaleFrac=0.5:SigmaFact=1.5:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" ); 
//factory->BookMethod(dataloader, TMVA::Types::kKNN, "KNN3", "H:nkNN=900:ScaleFrac=0.5:SigmaFact=1.5:Kernel=Gaus:UseKernel=F:UseWeight=T:!Trim" ); 
//factory->BookMethod(dataloader, TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm" ); 

//factory->BookMethod(dataloader, TMVA::Types::kRuleFit, "RuleFit", 
//                   "H:!V:ForestType=AdaBoost:RuleMinDist=0.:NTrees=60:fEventsMin=0.001:fEventsMax=5" ); 

    // improved neural network implementation
//       TString layoutString ("Layout=TANH|(N+100)*2,LINEAR");
//       TString layoutString ("Layout=SOFTSIGN|100,SOFTSIGN|50,SOFTSIGN|20,LINEAR");
//       TString layoutString ("Layout=RELU|300,RELU|100,RELU|30,RELU|10,LINEAR");
//       TString layoutString ("Layout=SOFTSIGN|50,SOFTSIGN|30,SOFTSIGN|20,SOFTSIGN|10,LINEAR");
//       TString layoutString ("Layout=TANH|50,TANH|30,TANH|20,TANH|10,LINEAR");
//       TString layoutString ("Layout=SOFTSIGN|50,SOFTSIGN|20,LINEAR");
// TEST OK!
    TString layoutString ("Layout=TANH|30,TANH|100,TANH|100,TANH|100,TANH|100,TANH|100,TANH|30,SIGMOID");


    TString training0 ("LearningRate=1e-1,Momentum=0.0,Repetitions=1,ConvergenceSteps=300,BatchSize=24,TestRepetitions=15,WeightDecay=0.001,Regularization=None,DropConfig=0.0+0.5+0.5+0.5+0.5+0.5,DropRepetitions=1,Multithreading=True");
    TString training1 ("LearningRate=1e-2,Momentum=0.3,Repetitions=1,ConvergenceSteps=300,BatchSize=24,TestRepetitions=7,WeightDecay=0.001,Regularization=L2,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1+0.1+0.1,DropRepetitions=1");
    TString training2 ("LearningRate=1e-3,Momentum=0.3,Repetitions=1,ConvergenceSteps=300,BatchSize=70,TestRepetitions=7,WeightDecay=0.0001,Regularization=None,Multithreading=True,DropConfig=0.0+0.1+0.1+0.1+0.1+0.1,DropRepetitions=1");
    TString training3 ("LearningRate=1e-4,Momentum=0.3,Repetitions=1,ConvergenceSteps=300,BatchSize=70,TestRepetitions=7,WeightDecay=0.0001,Regularization=None,Multithreading=True,DropConfig=0.0+0.0+0.0+0.0+0.0+0.0,DropRepetitions=1");

    TString trainingStrategyString ("TrainingStrategy=");
    trainingStrategyString += training0 + "|" + training1 + "|" + training2 + "|" + training3;

// TEST OK!
 //trainingStrategyString += training0 + "|" + training1;

//       TString nnOptions ("!H:V:VarTransform=Normalize:ErrorStrategy=CROSSENTROPY");
    TString nnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=G:WeightInitialization=XAVIERUNIFORM");
//       TString nnOptions ("!H:V:VarTransform=Normalize:ErrorStrategy=CHECKGRADIENTS");
    nnOptions.Append (":");
    nnOptions.Append (layoutString);
    nnOptions.Append (":");
    nnOptions.Append (trainingStrategyString);

    TString gpuOptions = nnOptions + ":Architecture=GPU";
    TString cpuOptions = nnOptions + ":Architecture=CPU";
    factory->BookMethod(dataloader, TMVA::Types::kDNN, "DNN_GPU", gpuOptions);
    //factory->BookMethod(dataloader, TMVA::Types::kDNN, "DNN_CPU", cpuOptions);

   factory->TrainAllMethods();

   // ---- Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // ----- Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();

   // --------------------------------------------------------------

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification is done!" << std::endl;

   delete factory;
   delete dataloader;
   // Launch the GUI for the root macros
   if (!gROOT->IsBatch()) TMVA::TMVAGui( outputFile->GetName() );
}
