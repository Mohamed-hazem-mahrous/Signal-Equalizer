import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QFileDialog
from gui import Ui_MainWindow
import io
import os
from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import soundfile as sf
from PyQt5.QtGui import QBrush, QPen, QColor


class EqualizerGUI(Ui_MainWindow):
    def setupUi(self, MainWindow):
        Ui_MainWindow.setupUi(self, MainWindow)

class Equalizer(QMainWindow):
    
    def __init__(self):
        super(Equalizer, self).__init__()
        self.gui = EqualizerGUI()
        self.gui.setupUi(self)
        self.setFocus()
        
        self.file_extension = None

        self.folder_path = None
        
        
        self.last_output_player_position = 0
        
        # Initialize various data-related attributes
        self.data = []
        self.data_fft = None
        self.time = []
        self.data_modified = []
        self.data_modified_fft = None
        self.frequencies = None
        self.data_modified_frequencies = None
        self.sample_rate = None
        self.data_ranges = []
        self.section_width = None
        self.path = None
        
        
        # infinite line to act as seeker on plot
        self.medPlayer_seeker_input = pg.InfiniteLine(pos = 0, angle = 90, pen = pg.mkPen(color = 'y', width = 2), movable =  True)
        self.medPlayer_seeker_output = pg.InfiniteLine(pos = 0, angle = 90, pen = pg.mkPen(color = 'y', width = 2), movable =  True)

        
 

        # Set default values for some parameters
        self.mult_window = "Rectangle"
        self.std = 100
        self.current_position = 0

        self.modes_sliders_labels = {
            "Uniform Range Mode": ["Slider 1", "Slider 2", "Slider 3", "Slider 4", "Slider 5", "Slider 6", "Slider 7", "Slider 8", "Slider 9", "Slider 10"],
            "Musical Instruments Mode": ["guitar", "piccolo", "triangle", "xylophone"],
            "Animal Sounds Mode": ["owl", "horse", "bat", "snake"],
            "ECG Abnormalities Mode": ["Ventricular Tachycardia", "Atrial Fibrillation", "Ventricular Couplets"],

        }

        self.modes_num_sliders = {
            "Uniform Range Mode": 10,
            "Musical Instruments Mode": 4,
            "Animal Sounds Mode": 4,
            "ECG Abnormalities Mode": 3
        }

        self.modes_freqs = {
            "Uniform Range Mode": [0],
            "Musical Instruments Mode": [0, 900, 2000, 6000, 15000],
            "Animal Sounds Mode": [0, 820, 2500, 5000, 11000],
            "ECG Abnormalities Mode": [0, 1, 30, 160, 161, 250],
        }


        self.window_mode_y = {
            "Rectangle": np.ones_like(np.linspace(0, 1, 1000)),
            "Hamming": np.hamming(1000),
            "Hanning": np.hanning(1000),
            "Gaussian": np.zeros(1000),
        }

        
        self.widgets = {
            "input": [self.gui.plot_input_sig_freq, self.gui.plot_input_sig_time, self.gui.plot_input_sig_spect,],
            "output": [self.gui.plot_output_sig_freq, self.gui.plot_output_sig_time, self.gui.plot_output_sig_spect,],
        }
           
        

        self.media_player_status = 0

        # Initialize data_ranges with None values
        self.data_ranges = [None] * 10

        # Set up lists of sliders, gains, and frequencies
        self.sliders = [
            self.gui.slider1, self.gui.slider2, self.gui.slider3, self.gui.slider4, self.gui.slider5,
            self.gui.slider6, self.gui.slider7, self.gui.slider8, self.gui.slider9, self.gui.slider10
        ]

        self.sliders_gains = [
            self.gui.lnEdit_gain_slider_1, self.gui.lnEdit_gain_slider_2, self.gui.lnEdit_gain_slider_3, self.gui.lnEdit_gain_slider_4, self.gui.lnEdit_gain_slider_5,
            self.gui.lnEdit_gain_slider_6, self.gui.lnEdit_gain_slider_7, self.gui.lnEdit_gain_slider_8, self.gui.lnEdit_gain_slider_9, self.gui.lnEdit_gain_slider_10
        ]

        self.sliders_freqs = [
            self.gui.lnEdit_freq_slider_1, self.gui.lnEdit_freq_slider_2, self.gui.lnEdit_freq_slider_3, self.gui.lnEdit_freq_slider_4, self.gui.lnEdit_freq_slider_5,
            self.gui.lnEdit_freq_slider_6, self.gui.lnEdit_freq_slider_7, self.gui.lnEdit_freq_slider_8, self.gui.lnEdit_freq_slider_9, self.gui.lnEdit_freq_slider_10
        ]

        # Set up a list of slider widgets
        self.slider_wgts = [
            self.gui.wgt_sld_1, self.gui.wgt_sld_2, self.gui.wgt_sld_3, self.gui.wgt_sld_4, self.gui.wgt_sld_5,
            self.gui.wgt_sld_6, self.gui.wgt_sld_7, self.gui.wgt_sld_8, self.gui.wgt_sld_9, self.gui.wgt_sld_10
        ]

        self.sliders_labels = [
            self.gui.label_slider1, self.gui.label_slider2, self.gui.label_slider3, self.gui.label_slider4, self.gui.label_slider5,
            self.gui.label_slider6, self.gui.label_slider7, self.gui.label_slider8, self.gui.label_slider9, self.gui.label_slider10
        ]
        
        # Set up a list of view widgets
        self.views = [
            self.gui.plot_input_sig_freq, self.gui.plot_input_sig_time, self.gui.plot_input_sig_spect,
            self.gui.plot_output_sig_freq, self.gui.plot_output_sig_time, self.gui.plot_output_sig_spect
        ]

        # For loop to connect each slider with its func.
        for i in range(10):
            self.connect_sliders(i)

        # Connect the "Open" action in the GUI to the method self.open_wav_file
        self.gui.actionOpen.triggered.connect(self.open_wav_file)

        # Connect the "Save" action in the GUI to the method self.save_output_file
        self.gui.actionSave.triggered.connect(self.save_output_file)


        
        # Shortcut to change modes using ctrl+tab
        switchTabsShortcut = QtWidgets.QShortcut(QtGui.QKeySequence("ctrl + tab"), self)
        switchTabsShortcut.activated.connect(self.cycle_modes_with_index)
        switchTabsShortcut.activated.connect(lambda: print("Shortcut Triggered"))
        
        


        # Create a QMediaPlayer instance for playing audio
        # Input #
        self.media_player_input = QMediaPlayer()
        
        # Vertical line to act as seeker on plot

        self.media_player_input.stateChanged.connect(lambda state: self.on_media_changed_state(state, media="Input"))
        self.media_player_input.positionChanged.connect(lambda position: self.medPlayer_seeker_input.setValue(position / 1000))
        
        # Output #
        self.media_player_output = QMediaPlayer()
        
        # Vertical line to act as seeker on plot
        
        self.media_player_output.stateChanged.connect(lambda state: self.on_media_changed_state(state, media="Output"))
        self.media_player_output.positionChanged.connect(lambda position: self.medPlayer_seeker_output.setValue(position / 1000))
        
        # Allow Scrubbing with seeker
        self.medPlayer_seeker_input.sigDragged.connect(self.update_player_position)
        self.medPlayer_seeker_output.sigDragged.connect(self.update_player_position)
        self.medPlayer_seeker_output.sigPositionChanged.connect(lambda: self.medPlayer_seeker_input.setValue(self.medPlayer_seeker_output.value()))
        self.medPlayer_seeker_input.sigPositionChanged.connect(lambda: self.medPlayer_seeker_output.setValue(self.medPlayer_seeker_input.value()))

        
        self.gui.btn_reset_sliders.clicked.connect(self.reset_sliders)
       
        # INPUT
        # Connect the button click event to the play, restart, seek_backward and seek_forward methods
        self.gui.btn_play_input.clicked.connect(lambda: self.play_file(self.media_player_input))
        self.gui.btn_rewind_input.clicked.connect(lambda: self.restart_file(self.media_player_input, path=self.path))
        self.gui.btn_pan_left_input.clicked.connect(lambda: self.seek(self.media_player_input, -5000))
        self.gui.btn_pan_right_input.clicked.connect(lambda: self.seek(self.media_player_input, 5000))

        
        # OUTPUT
        # Connect the button click event to the play, restart, seek_backward and seek_forward methods
        self.gui.btn_play_output.clicked.connect(lambda: self.play_file(self.media_player_output))
        self.gui.btn_rewind_output.clicked.connect(lambda: self.restart_file(self.media_player_output, path="output.wav"))
        self.gui.btn_pan_left_output.clicked.connect(lambda: self.seek(self.media_player_output, -5000))
        self.gui.btn_pan_right_output.clicked.connect(lambda: self.seek(self.media_player_output, 5000))

        # Connect checkbox to show/hide spectrograms
        self.gui.chkbx_spect.stateChanged.connect(self.hide_spectrogram)
            
        # Connect Combo boxes
        self.gui.cmbx_mode_selection.currentIndexChanged.connect(self.switch_modes)
        self.gui.cmbx_multWindow.currentIndexChanged.connect(self.update_window)
        self.plot_multwindow_all(mode="Rectangle") # Initializing the plot on multiplication window


        self.gui.slider_amplitude_2.valueChanged.connect(self.set_std)
        self.gui.slider_amplitude_2.valueChanged.connect(lambda :self.plot_multwindow_all(mode="Gaussian"))
        self.gui.slider_amplitude_2.setEnabled(False)
        
        
        # CONNECT PLOT CONTROLS
        self.gui.btn_zoom_in.clicked.connect(lambda: self.zoom(self.views[1], 0.8))
        self.gui.btn_zoom_out.clicked.connect(lambda: self.zoom(self.views[1], 1.2))
        self.gui.slider_speed.valueChanged.connect(lambda value: self.change_speed(value))
        
        
        # Window setup at first launch
        
        self.hide_spectrogram()
        self.link_views()
        for view in self.views:
            view.setLimits(xMin = 0)
        
        self.gui.wgt_multWindow_amp.setVisible(False)

    #=============================== Function Definitions ===============================#
    
    
    # Links Views
    def link_views(self):
        for input_view, output_view in zip(self.views[:3], self.views[3:]):
            input_view.setXLink(output_view) 
            input_view.setYLink(output_view) 
        pass
        
    # Zoom Function
    def zoom(self,plot: pg.PlotWidget, zoom_factor: float):
        viewBox = plot.getViewBox()
        viewBox_center = viewBox.viewRect().center()
        viewBox.scaleBy((zoom_factor, zoom_factor) , viewBox_center)

        
                
    
    def cycle_modes_with_index(self):
        if self.gui.cmbx_mode_selection.currentIndex() < 3:
            new_index = self.gui.cmbx_mode_selection.currentIndex() + 1
            self.gui.cmbx_mode_selection.setCurrentIndex(new_index)
        else:
            self.gui.cmbx_mode_selection.setCurrentIndex(0)
    


# Function to change playback speed
    def change_speed(self, value):
        speed_value = round(value/10, 1)
        self.gui.lbl_speed.setText(f"x{speed_value}")
        self.media_player_input.setPlaybackRate(speed_value)
        self.media_player_output.setPlaybackRate(speed_value)
                
                
    # Update the positions of both input and output media players to match the value of the seeker bar on the plot                
    def update_player_position(self):
        # Set the position of the input media player to the value of the seeker bar
        self.media_player_input.setPosition(int(self.medPlayer_seeker_input.value()))

        # Set the position of the output media player to the value of the seeker bar
        self.media_player_output.setPosition(int(self.medPlayer_seeker_output.value()))

        
        
        
    
############################################################## Mode Changing methods ##############################################################

    # Function to show specified sliders and change their labels
    def modifiy_sliders(self, start_index, end_index, new_slider_name):
        for i, widget in enumerate(self.slider_wgts):
            # Only show slider widgets that are in the given range
            if i in range(start_index, end_index):
                widget.setVisible(True)
            else:
                widget.setVisible(False)

            for i in range(end_index):
                self.sliders_labels[i].setText(f"{self.modes_sliders_labels[new_slider_name][i]}")


    def change_mode(self, num_sliders, mode_name):
        self.modifiy_sliders(0, num_sliders, mode_name)
        self.clear_graphs()

    def switch_modes(self):
        mode = self.gui.cmbx_mode_selection.currentText()
        self.change_mode(self.modes_num_sliders[mode], mode)
        print(mode)

    
###################################################################################################################################################

    # Reset the graphs and sliders  
    def clear_graphs(self):
        self.reset_sliders()
        for i in range(10):
            self.sliders_freqs[i].setText(str(0))

        for plot_graph in self.views:
            plot_graph.clear()


    def reset_sliders(self):
        for i in range(10):
            self.sliders[i].setValue(0)
            self.sliders_gains[i].setText(str(0))

        self.data_modified = self.data
        self.data_modified_fft = self.data_fft
        self.data_modified_frequencies = self.frequencies

        # Plot the original signal and the modified signal     
        self.plot_widgets(self.data, self.data_fft, self.frequencies, "input")
        self.plot_widgets(self.data_modified, self.data_modified_fft, self.data_modified_frequencies, "output")

        self.save_output_file()

    
    

    def set_std(self):
        self.std = self.gui.slider_amplitude_2.value()
        self.gui.lbl_value_amp_3.setText(str(self.gui.slider_amplitude_2.value()))
        
        
    def hide_spectrogram(self):
        self.gui.plot_input_sig_spect.setVisible(self.gui.chkbx_spect.isChecked())
        self.gui.plot_output_sig_spect.setVisible(self.gui.chkbx_spect.isChecked())

    


########################################################### Multiplication window methods ###########################################################
    def update_window(self, index):
        # Get the selected item from the combo box
        selected_item = self.gui.cmbx_multWindow.currentText()
        self.gui.wgt_multWindow_amp.setVisible(False)
        self.gui.slider_amplitude_2.setEnabled(False)
        self.mult_window = selected_item
        if selected_item == "Gaussian":
            self.gui.wgt_multWindow_amp.setVisible(True)
            self.gui.slider_amplitude_2.setEnabled(True)

        self.plot_multwindow_all(selected_item)

        print(f"Selected window: {self.mult_window}")


    def plot_multwindow_all(self, mode):
        # Plot the selected window in self.gui.plot_multWindow
        x = np.linspace(0, 1, 1000)
        try:
            y = self.window_mode_y[mode]
        except:
            pass

        if mode == "Gaussian":
            std = self.gui.slider_amplitude_2.value()
            y = 10 * np.exp(-(x - 0.5)**2 / (2 * (1 / std)**2))

        self.gui.plot_multWindow.clear()
        self.gui.plot_multWindow.plot(x, y, pen="r")


#####################################################################################################################################################


########################################################## Reset, play and seeking methods ##########################################################
    def seek(self, media, sec):
        current_position = media.position()
        new_position = current_position + sec
        media.setPosition(new_position)
        

    def restart_file(self, media, path):
        if self.sample_rate is not None:
            self.load_media_file(media, path)
            media.play()
    
    # Sets the media file to be played by the player
    def load_media_file(self,media: QMediaPlayer, path):
            media_content = QMediaContent(QUrl.fromLocalFile(path))
            media.setMedia(media_content)

    def clear_media(self, media: QMediaPlayer):
        media.setMedia(QMediaContent())
        
    # Governs Playing and pausing
    def play_file(self, media: QMediaPlayer):
        if self.file_extension == '.wav':
            if self.sample_rate is not None:
                if media.state() == QMediaPlayer.State.PlayingState:
                    media.pause()
                else:    
                    media.play()


    def on_media_changed_state(self, state, media):       
        # Handle media player state changes, e.g., update UI based on playback status
        if state == QMediaPlayer.PlayingState:
            print(f"{media} Audio is playing")
        elif state == QMediaPlayer.StoppedState:
            print(f"{media} Audio playback stopped")
        elif state == QMediaPlayer.PausedState:
            print(f"{media} Audio playback paused")


#####################################################################################################################################################
    def save_wav_file(self):
        modified_signal = self.data_modified * (32767 / max(self.data_modified))

        # Convert the data to the appropriate integer type for wavfile.write
        modified_signal = modified_signal.astype(np.int16)
        
        # If there is a preexisting output file, remove it
        try:
            os.remove('output.wav')
        except:
            pass
        
        # Write the WAV file
        wavfile.write('output.wav', self.sample_rate, modified_signal)
        print("Output Wav file is saved")
    
    def save_csv_file(self, filename):
        # Create a DataFrame with time and modified data
        time_values = np.arange(len(self.data_modified)) / self.sample_rate
        df = pd.DataFrame({'Time': time_values, 'Data': self.data_modified})

        try:
            os.remove('output.csv')
        except:
            pass


        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        print(f"Output CSV file is saved: {filename}")


    def save_output_file(self):
        if self.file_extension == '.wav':
            self.save_wav_file()
            
            # Force clear cache of media player
            self.clear_media(self.media_player_output)
            
            # Load the new output file
            self.load_media_file(self.media_player_output, 'output.wav')

        else:
            self.save_csv_file('output.csv')





    def set_data_ranges(self, mode, step = 1):
        if mode == "ECG Abnormalities Mode":
            step = 2

        num_sliders = self.modes_num_sliders.get(mode, 1)
        for i in range(num_sliders):
            start_idx = np.argmin(np.abs(self.frequencies - self.modes_freqs[mode][step*i]))
            end_idx = np.argmin(np.abs(self.frequencies - self.modes_freqs[mode][step*i+1]))
            self.data_ranges[i] = [start_idx, end_idx]                 

    # Open the wav file
    def open_wav_file(self):
        try:
            files_name = QFileDialog.getOpenFileName(self, 'Open File', os.getenv('HOME'), "Audio files (*.wav *.csv)")
            self.path = files_name[0]
            if self.path:
                self.data = None
                self.data_ff = None
                self.frequencies = None
                self.data_modified = None
                self.data_modified_fft = None
                self.data_modified_frequencies = None
                self.modes_freqs["Uniform Range Mode"] = [0]


                # Check the file extension to determine the file type
                self.file_extension = os.path.splitext(self.path)[-1].lower()

                self.folder_path = os.path.splitext(self.path)[-2]


                if self.file_extension == '.wav':
                    signal, sample_rate = librosa.load(self.path)

                    # sample_rate, signal = wavfile.read(self.path)
                    self.data = signal
                    self.sample_rate = sample_rate
                    
                    # Calculate time and multiply duration by 1000 to get it in milliseconds
                    self.time = np.linspace(0, librosa.get_duration(y = self.data, sr = self.sample_rate), len(list(self.data)))

                elif self.file_extension == '.csv':
                    # CSV file loading logic using pandas
                    df = pd.read_csv(self.path)

                    # Assuming your CSV has a column named 'time' and another column 'signal'
                    time_column = 'Time'
                    signal_column = 'Data'
                    # Extract data and sample rate from CSV
                    self.data = df[signal_column].values
                    self.sample_rate = 1 / (df[time_column].values[1] - df[time_column].values[0])

                    self.time = np.linspace(0, len(self.data) / self.sample_rate, len(self.data))


                if self.file_extension == '.wav':
                    # Load media file into media player for input
                    self.load_media_file(self.media_player_input, self.path)

                
                

                self.data_fft = np.fft.rfft(self.data)
                self.frequencies = np.fft.rfftfreq(len(self.data), 1 / self.sample_rate)

                self.data_modified = self.data
                self.data_modified_fft = self.data_fft
                self.data_modified_frequencies = self.frequencies

                self.section_width = len(self.frequencies) // 10

                self.save_output_file()

                mode = self.gui.cmbx_mode_selection.currentText()

                for i in range(10):
                    start_idx = i * self.section_width
                    end_idx = (i + 1) * self.section_width
                    self.modes_freqs["Uniform Range Mode"].append(self.frequencies[end_idx])

                
                self.set_data_ranges(mode)

                # Plot the original signal and the modified signal
                self.plot_widgets(self.data, self.data_fft, self.frequencies, "input")
                self.plot_widgets(self.data_modified, self.data_modified_fft, self.data_modified_frequencies, "output")
                                

                if self.file_extension == '.wav':
                    self.gui.plot_input_sig_time.addItem(self.medPlayer_seeker_input)
                    self.gui.plot_output_sig_time.addItem(self.medPlayer_seeker_output)

                self.set_bands_freq_sliders()

        except Exception as e:
            print(f"Error: {e}")



    # Plotting spectrograms
    def plot_spectrogram(self, plot, data, title):
        plot.clear()
        
        # Compute Spectrogram using the spectrogram function
        f, t, sxx = spectrogram(data, fs=self.sample_rate, mode='psd')
        
        spectrogram_data = np.log10(sxx) # Apply logarithmic scaling to enhance visualization
        spectrogram_data = spectrogram_data.T
        
        
        # Plot Spectrogram using an ImageItem
        img = pg.ImageItem()
        img.setImage(spectrogram_data, autoLevels=False)  
        img.setLevels([-10, -2])
        plot.addItem(img)

        # Set labels for the plot axes and specify units
        plot.setLabel('left', 'Frequency', units='Hz')
        plot.setLabel('bottom', 'Time', units='s')

        # # Set the color map for the Spectrogram
        colormap = pg.colormap.get('inferno')
        img.setColorMap(colormap)

        # Set the title of the plot
        plot.setTitle(title)




    def plot_widgets(self, data, data_fft, freq, mode):
        for i in range(2):
            self.widgets[mode][i].clear()
        self.widgets[mode][1].plot(y = data, x = self.time, pen="r")
        self.widgets[mode][0].plot(freq, np.abs(data_fft), pen="r")

        self.widgets[mode][1].setLabel('bottom', 'Time', units='s')
        self.widgets[mode][1].setLabel('left', 'Amplitude')

        self.widgets[mode][0].setLabel('left', 'Amplitude')
        self.widgets[mode][0].setLabel('bottom', 'Frequency', units='Hz')


        if mode == "input":
            self.widgets[mode][1].addItem(self.medPlayer_seeker_input)
        else:
            self.widgets[mode][1].addItem(self.medPlayer_seeker_output)

        self.plot_spectrogram(self.widgets[mode][2], data, f"{mode} Spectrogram")   




    # Setting the freq. sliders bands
    def set_bands_freq_sliders(self):
        mode = self.gui.cmbx_mode_selection.currentText()
        step = 1
        if mode == "ECG Abnormalities Mode":
            step = 2

        num_sliders = self.modes_num_sliders.get(mode, 1)
        for i in range(num_sliders):
            self.sliders_freqs[i].setText(f"{int(np.round(self.modes_freqs[mode][step*i]))} : {int(np.round(self.modes_freqs[mode][step*i + 1]))}")


   # Connect the sliderReleased signal of a specific slider (at the given index) to the mult_freqs method 
    def connect_sliders(self, index):
        self.sliders[index].valueChanged.connect(lambda: self.mult_freqs(index))
        self.sliders[index].sliderReleased.connect(self.save_output_file)
           

    def mult_freqs(self, index):
        # Multiply frequencies within the specified range in the FFT data by a scaling factor
        self.data_modified_fft = self.multiply_fft(
            self.data_modified_fft,
            self.data_fft,
            self.data_ranges[index][0],
            self.data_ranges[index][1],
            10**(self.sliders[index].value() / 20),
            std_gaussian=self.section_width / self.std,
            mult_window=self.mult_window
        )

        # Update the gain label with the current slider value
        self.sliders_gains[index].setText(str(self.sliders[index].value()))
        # Inverse FFT to obtain the modified time-domain signal
        self.data_modified = np.fft.irfft(self.data_modified_fft)
        # Update the modified FFT on the secondary plot
        self.plot_widgets(self.data_modified, self.data_modified_fft, self.data_modified_frequencies, "output")



    def multiply_fft(self, data, data_orig, start, end, index, std_gaussian, mult_window):   
        # Create a copy of the original FFT data
        modified_data = data.copy()
    
        # Apply windowing and multiplication based on the chosen window type
        if mult_window == "Rectangle":
            modified_data[start:end] = data_orig[start:end] * index
    
        elif mult_window == "Hamming":
            hamming_window = np.hamming(end - start) * index
            modified_data[start:end] = data_orig[start:end] * hamming_window
    
        elif mult_window == "Hanning":
            hanning_window = np.hanning(end - start) * index
            modified_data[start:end] = data_orig[start:end] * hanning_window
    
        elif mult_window == "Gaussian":
            gaussian_window = np.exp(-0.5 * ((np.arange(end - start) - (end - start) / 2) / std_gaussian) ** 2) * index
            modified_data[start:end] = data_orig[start:end] * gaussian_window
    
        # Return the modified FFT data
        return modified_data



# Run Application
app = QApplication(sys.argv)
win = Equalizer()
win.show()
app.exec()