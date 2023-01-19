In this Lab we implemented and refined a convolutional neural
network for supervised video prediction. Given a set of three consecutive
frames the network predicts the next three frames. The network
architecture has been proposed for the lab and extends on the Video
Ladder Network by utilizing additional location dependent convolutions
as well as exchanging the LSTM layers to more memory efficient GRU
layers. Thus, we will call this model Location dependent Video Ladder
Network (L-VLN). In addition, we explore the possibilities for transfer
learning by using the hidden states of the video prediction network for
classification. This report will give a short overview of the used technologies
as well as an implementation.
The network was implemented using pytorch and the pytorch lightning library.
Figure 1 shows the architecture.
The training and validation datasets are loaded from UCF101 videos that
have been split into clips of sequences of six frames.
Figure 2 shows the proposed training and prediction routine.
To confirm, check and debug the L-VLN we first implemented a simplified model.
As a simplified dataset we use the horizontal translation of a circle to check if
the temporal ConvGRU modules correctly capture and predict the movement.
Figure 3 shows the simplified dataset.
Unfortunately the L-VLN model fails to predict movement within the images
on larger datasets like UCF101 and instead only blurs the regions containing movement.
This suggests that either the problem complexity is too
high for the proposed L-VLN model, or we lack the computational resources to
adequately train the 31 million parameters. We therefore propose to reduce the
problem complexity by predicting only a single frame at a time, given a sequence
of varying length. 
Figure 4 shows our proposed simplified training and prediction routine.
For training we use seeding sequences of 5 frames to predict
Figure 5 shows how results on UCF101 with the different methods.
Figure 6 shows some of our results.
For more information please refer to the seminarreport.pdf.

