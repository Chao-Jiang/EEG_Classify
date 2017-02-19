# EEG_Classify
### Some attempts to classify EEG signals from the BCI competition 

I have tried to use the Long Short-Term Memory units for classification of lateralized readiness potentials from the BCI competition II dataset IV. These don't seem to work nearly as well as convolutional neural networks. The support vector machine fails to get more than chance on the dataset. This is a bit counter-intuitive since recurrent nerual networks are more suited to sequence data. 

These are the accuracies so far:

Support Vector Machine
: ~50% 

Long Short-Term Memory
: 60%

Convolutional Neural Networks
: 100% 

To do:
* Try convolutional LSTM
* Fix optimizations and targets to get the loss to behave in LSTM
* Try using live data with the Wyrm package


