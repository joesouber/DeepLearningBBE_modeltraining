## This repositiory is to be used in conjunction with (https://github.com/joesouber/LSTM_BBE).

This repository is for the pre-processing of training data and the eventual training of the LSTM model. It is separated from the DeepBettor repo to keep generated data manageable and distsinct.

1. Once the BBE simulations have been run in the DeepBettor Repository, you will find a number of csv files entitled getXGBOOstTrainingData_{}.csv, with {} depicting the number of the simulation. The path to each of these files will be : ...(path_to_repository)/Core{}/TBBE_OD_XGboost/Application/getXGBOOstTrainingData_{}.csv. In order to agglomorate the training data from each core, the script merge_csv.py has been created. Please see the script for details of use.

2. Once training data has been generated into one csv, it is advisable to launch an AWS EC2 instance to process and train the data. I opted for a c5.2xlarge instance.
3. 
4. The trained model should then be saved as an .h5 file, for deployment back into the BBE as an Agent.
