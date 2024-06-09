# Signal Equalizer

## Table of Contents

- [Introduction](#introduction)
- [Team Members](#Team-Members)
- [Features](#features)
- [Video Demonstration](#Video-Demonstration)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
This signal equalizer project provides a user-friendly interface for manipulating and analyzing audio signals. It allows users to open audio files, apply various equalization adjustments, and visualize the effects of these adjustments in real time.

## Team Members
The following team members have contributed to this project:
- [Mohamed Hazem Yehya](https://github.com/Mohamed-hazem-mahrous)
- [Assem Hussein](https://github.com/RushingBlast)
- [Kyrolos Emad](https://github.com/kyrillos-emad)
- [Arsany Maged](https://github.com/Arsany07)


## Features
-   **Audio File Loading:** Users can open audio files in various formats to apply equalization techniques.
    
-   **Equalization Controls:** Users can adjust multiple equalizer sliders to modify the frequency response of the audio signal, enabling them to enhance or suppress specific frequency bands.
    
-   **Synchronized Signal Viewers:** Two signal viewers display the input and output signals simultaneously, allowing users to observe the original signal and the effects of equalization in real time. These viewers are synchronized to ensure that they always show the same time-part of the signal, regardless of scrolling or zooming.
    
-   **Spectrograms:** Two spectrograms, one for the input and one for the output signals, provide a visual representation of the frequency content of the signals. The output spectrogram dynamically updates to reflect the changes made by the equalizer sliders.
    
-   **Spectrogram Toggle:** Users can toggle the visibility of the spectrograms to focus on either the signal viewers or the frequency representations.
    
-   **Equalization Modes:** Four different equalization modes are available:
    
    1.  Uniform Range Mode: Provides a general-purpose equalizer with sliders for adjusting frequencies across the entire audio spectrum.
        
    2.  Musical Instruments Mode: Offers sliders tailored to specific frequency ranges corresponding to different musical instruments, allowing users to enhance or suppress the presence of those instruments in the audio signal.
        
    3.  Animal Sounds Mode: Provides sliders tuned to frequency ranges associated with various animal sounds, enabling users to manipulate the prominence of these sounds in the audio signal.
        
    4.  ECG Abnormalities Mode: Offers sliders designed to detect and highlight potential abnormalities in electrocardiogram (ECG) signals, aiding in the analysis of heart conditions.
        
    
-   **Smoothing Window Customization:** Users can select the type of smoothing window to apply to the equalizer bands, influencing the smoothness of the frequency response adjustments. They can also visually customize the parameters of the smoothing window and observe the effects in real time.
    
-   **Equalizer Application:** Once satisfied with the equalization settings, users can apply the customized equalizer to the audio signal and save the modified audio file.


## Video Demonstration
[App Demo](https://github.com/Mohamed-hazem-mahrous/Signal-Equalizer/assets/94749599/ac998a89-ee3d-4477-93f2-71f4771dab47)



## Requirements
- PyQt5
- NumPy
- SciPy
- Matplotlib
- PyQtGraph
- librosa
- pandas
- soundfile

## Installation
Install the required dependencies using pip:
```bash
pip install PyQt5 numpy scipy matplotlib pyqtgraph librosa pandas soundfile
```


## Usage
Run the application:
```bash
python Equalizer.py
```
1.  Launch the signal equalizer application.

2.  Select the desired equalization mode from the Mode menu.
    
3.  Open an audio file using the File menu .
    
4.  Adjust the equalizer sliders to modify the frequency response of the audio signal.
    
5.  Observe the effects of equalization in the signal viewers and spectrograms.
    
6.  Toggle the visibility of the spectrograms using the Spectrogram Toggle checkbox.
    
7.  Customize the smoothing window type and parameters using the Smoothing Window panel.
    
8.  Save the modified audio file using the File > Save menu.
    



## Contributing
Contributions to Signal Equalizer are welcome! If you encounter any issues or have suggestions for improvements, please create a new issue or submit a pull request.

When contributing, please ensure that you follow the existing coding style and include clear commit messages to maintain a well-documented project history.
