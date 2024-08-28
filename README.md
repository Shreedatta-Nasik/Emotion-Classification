# EEG based Emotion Recognition using SpinalNet
This project presents an innovative approach for emotion recognition utilizing Electroencephalography (EEG) signals, employing a two-stage framework comprising a Power Spectral Density (PSD) feature extractor and a 
SpinalNet classifier. The PSD feature extractor transforms raw EEG signals into frequency-domain representations, capturing the underlying neural oscillations associated with emotional states. Subsequently, the 
SpinalNet classifier, incorporating both spatial and temporal dependencies through its architecture, effectively learns discriminative patterns for valence and arousal classification. The proposed model is rigorously 
evaluated on the DEAP database, adopting a subject noncontingent experimental paradigm. Results demonstrate an average accuracy of 70% and 62% for valence and arousal classification, respectively, with peak accuracies 
reaching 85% for valence and 82% for arousal. Comparative analysis against existing methodologies underscores the efficacy of the PSD-SpinalNet network in enhancing emotion detection accuracy from EEG signals, highlighting 
its potential to advance research in affective computing. In this project we also proposed a model called SpinalLSTM but did not perform well enough to beat SpinalNet. I have also used used Differential entropy and Linear
Formulation of Differential entropy. For analysing the model we are using subject non-contingent or subject independent approach. Which means that one subject is used for testing and all the other subjects are used for 
training the model. We have used k-fold=32 for 32 subjects

## DEAP dataset
The project was conducted utilizing the widely recognized dataset: the DEAP dataset, a commonly used benchmark in research. The [Database for Emotion Analysis using Physiological Signals (DEAP)](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
serves as a benchmark for affective EEG analysis, collected under controlled laboratory conditions. It comprises 32-channel EEG and 8-channel peripheral physiological signals from 32 subjects, with additional positive 
video recordings for 22 subjects. Electrode placements for EEG are illustrated below. Emotional states were elicited through 40 1-minute music videos, each corresponding to a different emotional state. 
                        ![image](https://github.com/user-attachments/assets/16543b55-5835-4bc6-8886-fc0cef1dc198)

Subjects rated arousal, valence, like/dislike, dominance, and familiarity of each trial on a scale of 1-9 using Self-Assessment Manikin (SAM). Emotions were defined according to the valence-arousal emotion model, 
dividing the two-dimensional emotional space into four regions: high valence-high arousal (HVHA), high valence-low arousal (HVLA), low valence-high arousal (LVHA), and low valence-low arousal (LVLA). The DEAP dataset 
offers two versions of physiological signal data: raw and pre-processed. While raw data may yield varied results due to pre-processing, such as noise reduction, this study ensures consistency by utilizing pre-processed 
data. The pre-processed DEAP data comprises 32 channels of EEG signals (at 128Hz) and 8 channels of peripheral physiological signals. The file is present it .dat file the data_process notebook extracts the data from the 
.dat file and saved it in a seperate .csv files. As there are 32 participants and 40 videos to watch total of 1280 labels are given. While the EEG signals are recorded in mV therefore due to the sampling rate and the 
63 seconds of the video there are 8064(128x63) data points are available. While pre-processing we discard the first 3 seconds of the data. To reduce this data so that a relationship can be established between the EEG signals
and the emotions we use feature extraction. The processed data is present in the datasets folder PSD45 is meant for LSTM and SpinalLSTM while PSD is for SpinalNet and MLP. Same nomencluture goes for DE and LFDE

## Feature Selection
As most of the EEG channels contain too much noise and redundant information we would like to select the channels that give the best accuracies. By referrring a [research paper](https://www.nature.com/articles/s41598-021-86345-5#:~:text=We%20used%20the%20zero%2Dtime,evaluated%20using%20the%20DEAP%20database.) 
which uses Zero time windowing I have selected 11 channels. They are 'Fp1','F3','F7','C3','T7','CP5','P3','O1','Fp2','F8','T8'.

## Feature Extraction
### Power Spectral Density:
It is a fundamental technique in signal processing, particularly in the analysis of EEG signals. It provides valuable understanding into the frequency content of a signal, offering a detailed understanding of the 
underlying physiological processes. PSD represents the distribution of power in a signal across different frequency components. It quantifies the contribution of each frequency to the overall signal power, making 
it an essential tool for studying the spectral characteristics of EEG signals. In our research, we utilized the Welch method for calculating PSD from EEG signals. In our analysis, we considered individual frequencies 
ranging from 0Hz to 45Hz, covering a wide spectrum relevant to EEG signals. Unlike grouping frequencies into traditional EEG frequency bands, we examined the PSD of each individual frequency separately. This approach 
allows for a more detailed exploration of the frequency-domain characteristics of EEG signals, enabling us to capture subtle variations and nuances in the signal's spectral profile.

### Differential Entropy:
It quantifies the uncertainty of a continuous random variable X. It's calculated by integrating the probability density function (PDF) f(x) multiplied by the natural logarithm of f(x) over X's support region S. 
For X following a Gaussian distribution N(μ, σ2), DE simplifies to 
![image](https://github.com/user-attachments/assets/e5f4db42-9260-4fc0-a832-886d07968dc0)

where μ is the mean and σ2 is the variance. DE is crucial for assessing randomness in data, particularly for time series governed by Gaussian distributions.

### Linear Formulation of differential Entropy:
The EEG signals are random in nature, and for analysing the nonlinearities in these signals, higher-order statistics (HOS) are utilized. HOS are employed to describe higher-order statistical characteristics of a random 
process, specifically the higherorder spectral moment. In this research, we focus on the fourth-order spectral moment (LF-DE). An EEG signal over a short time period can be represented as the time function g(t). Linear 
measures are more straightforward to integrate within both frequency and time domains. Consequently, a logarithmic differential equation is transformed into a linear form using a sigmoid equation.Using (1),
![image](https://github.com/user-attachments/assets/17d97354-ef9a-4574-a36e-f588a219e759)

![image](https://github.com/user-attachments/assets/7517bf8c-a87e-4753-b3cb-c3d8506c4748)

![image](https://github.com/user-attachments/assets/d9a31d3b-7be5-4b46-a8f2-39a1a5b4d299)

## Classification Methods
For comparative analysis we considered 4 deep learning models Multi Lyer Perceptron, LSTM, SpinalNet, SpinalLSTM.

### SpinalNet:
SpinalNet is a novel neural network structure inspired by the human somatosensory system, aimed at improving performance while reducing computational overhead. Traditional DNNs struggle with large input feature sets 
and vanishing gradients, hindering effective training. SpinalNet addresses these challenges by incorporating a gradual and distributed input mechanism, similar to the human spinal cord's processing. Comprising input,
intermediate, and output sub-layers, SpinalNet efficiently handles split inputs, minimizing computational complexity. By combining nonlinear activation functions in the intermediate layers with linear functions in 
the output layer, SpinalNet achieves robust performance with reduced overhead. Experimental results demonstrate SpinalNet’s superiority over traditional DNN architectures across various datasets.Integration of SpinalNet into popular Deep
CNNs like VGG-16 further enhances performance, highlighting its adaptability and potential for future advancements in neural network design. The architecture of the SpinalNet is shown in the below Figure. As you can see from the figure how it
distinguishes itself from the normal neural networks. For more information refer the [research paper](https://arxiv.org/pdf/2007.03347)

![image](https://github.com/user-attachments/assets/22c78a2e-1a44-4186-beb4-b164a232be41)

### Proposed SpinalLSTM:
This model was proposed by myself, But failed to meet the expectations. It was meant to combine both spinalNet and LSTM where the neurons are replaced by LSTM cells as shown in the below figure

![image](https://github.com/user-attachments/assets/2fe6d494-e46e-485c-be76-7ae624c7b39e)

![image](https://github.com/user-attachments/assets/35833ed8-2cb9-4931-9071-876c00231bfc)

![image](https://github.com/user-attachments/assets/31f7d864-3ca2-4daa-bf17-c56f6484b382)



