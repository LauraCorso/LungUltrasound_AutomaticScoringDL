# Lung Ultrasound Video-level Covid-19 scoring
Final project of the course *Medical Imaging Diagnostic* at University of Trento (year 2023).

## Description
The aim of this project is to design a system that allows to extract from a lung ultrasound video
its corresponding label indicating the degree of severity for COVID-19 pneumonia.
An automatic labelling system for lung ultrasound videos is developed using *Deep
Learning Networks*. In particular, following the example of
[1], a Convolutional Neural Network is used to extract
spatial features from the video frames. Then, a spatial attention
layer is added to detect the presence of image artefacts.
Furthermore, the spatial feature maps are input to a Recurrent
Neural Network to extract temporal features. Finally,
the so obtained results are input to a Fully Connected
layer to obtain a 4-classes classification. The aim of the developed model is to combine the spatial
features with the temporal ones in order to obtain a classification
technique that can be directly applied at video level.
In addition, the attention layer is placed in order to refine
the extraction of the spatial features needed to obtain a more
accurate classification.

[1] H. Kerdegari, N. T. H. Phung, A. McBride, L. Pisani, H. V. Nguyen, T. B.
Duong, R. Razavi, L. Thwaites, S. Yacoub, A. Gomez, and V. Consortium,
“B-line detection and localization in lung ultrasound videos using
spatiotemporal attention,” Applied Sciences, vol. 11, no. 24, 2021, https://www.mdpi.com/2076-3417/11/24/11697
