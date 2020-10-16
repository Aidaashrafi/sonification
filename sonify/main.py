import json
import random

import numpy as np
import scipy.signal
import astropy.io.fits
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

from mypackage.core import play_midi_from_data, scale_list_to_range, quantize_x_value

def plot_cleveland_stats(stats):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.set_ylim(ymax=1)
    ax.set_xlabel('Season year')
    ax.set_ylabel('Win Percentage')
    ax.set_title("Cleveland Cavaliers' Regular Season Win Percentage")
    
    ax.annotate("LeBron's Rookie Year", xy=('03/04', 0.427), xytext=(.3, .6),
    arrowprops=dict(facecolor='black', shrink=0.2, width=1),
    )
    ax.annotate("LeBron Leaves for Miami", xy=('10/11', 0.232), xytext=('10/11', 0.05),
        arrowprops=dict(facecolor='black', shrink=0.2, width=1),
    )
    ax.annotate("LeBron Returns to Cleveland", xy=('14/15', 0.646), xytext=('13/14', .9),
        arrowprops=dict(facecolor='black', shrink=0.2, width=1),
    )

    ax.plot(*zip(*stats), 'ro')
    

def show_plots_in_chunks(time, period, percent_change):
    n_plots = 10
    plt.figure(figsize=(10, 30))
    for i in range(n_plots):
        mask = (time >= time[0] + i*period) & (time < time[0] + (i+1)*period)
        plt.subplot(n_plots, 1, i+1)
        plt.scatter(time[mask], percent_change[mask], c='C{}'.format(i))
        plt.show()

def process_kepler_data(time, period, percent_change, low_note=20, high_note=100):    
    
    x_points = []
    y_points = []
    for i in range(10):
        mask = (time >= time[0] + i*period) & (time < time[0] + (i+1)*period)
        x_points += [x for x in time[mask] - time[0] - i*period]
        y_points += [y for y in percent_change[mask]]

    normalized_x = scale_list_to_range(x_points, new_min=0, new_max=30)
    normalized_y = scale_list_to_range(y_points, new_min=low_note, new_max=high_note)
        
    normed_data = list(zip(normalized_x, normalized_y))
    
    return normed_data

def process_kepler_data_multi_track(time, period, percent_change):
    # Add period as a seperate color so we can see all the data together
    
    points = []
    plt.figure(figsize=(10, 5))
    
    for i in range(5):
        mask = (time >= time[0] + i*period) & (time < time[0] + (i+1)*period)
        new_x = [x for x in time[mask] - time[0] - i*period]
        scaled_x = scale_list_to_range(new_x, new_min=0, new_max=30)
        
        new_y = [y for y in percent_change[mask]]
        scaled_y = scale_list_to_range(new_y, new_min=0, new_max=30)
        
        quantized_x = quantize_x_value(scaled_x)
                
        points.append(list(zip(quantized_x, scaled_y)))
        
        # Create the figure!
        mask = (time >= time[0] + i*period) & (time < time[0] + (i+1)*period)
        plt.scatter(time[mask] - time[0] - i*period, percent_change[mask])
        
    plt.show()    
    return points

def main():
    
    # first example
    # # Create some data we'd like to play from
    # simple_data = [(1, 50), (2, 50), (3, 57), (4, 57), (5, 59), (6, 59), (7, 57)]
    
    # # See what it looks like
    # plt.scatter(*zip(*simple_data))
    # plt.show()
    # play_midi_from_data(simple_data)
    
    
#     # second example
#     cleveland_seasons = [
#     ('00/01', 0.366),
#     ('01/02', 0.354),
#     ('02/03', 0.207),
#     ('03/04', 0.427), # LeBron's rookie year
#     ('04/05', 0.512),
#     ('05/06', 0.610),
#     ('06/07', 0.610),
#     ('07/08', 0.549),
#     ('08/09', 0.805),
#     ('09/10', 0.744),
#     ('10/11', 0.232), # LeBron leaves for Miami
#     ('11/12', 0.318),
#     ('12/13', 0.293),
#     ('13/14', 0.402),
#     ('14/15', 0.646), # LeBron returns to Cleveland
#     ('15/16', 0.695),
#     ('16/17', 0.622),
#     ('17/18', 0.610)
# ]
#     plot_cleveland_stats(cleveland_seasons)
    
    
#     season, win_percentage = zip(*cleveland_seasons)

#     # Just use the last year in the season
#     simple_year = [int(year.split('/')[-1]) for year in season]
    
#     # Scale the Y value to fit in a good MIDI note range
#     normalized_win_percentage = scale_list_to_range(win_percentage, new_min=30, new_max=127)
    
#     # Put it all back together!
#     processed_cleveland_data = list(zip(simple_year, normalized_win_percentage))
    
#     plt.scatter(*zip(*processed_cleveland_data))
    
#     plt.show()
#     play_midi_from_data(processed_cleveland_data)
#     play_midi_from_data(
#     ['pizzicato strings'] + processed_cleveland_data, key='g_major'
# )
    
    # third example
    
    """
    Example from @GeertHub's talk on "How to find a planet"
    http://nbviewer.jupyter.org/github/barentsen/how-to-find-a-planet/blob/master/how-to-find-a-planet.ipynb
    """
    # Read in Kepler data for star number 011904151
    data = astropy.io.fits.open('sample_data/kplr011904151-2010009091648_lpd-targ.fits')[1].data
    time = data["TIME"][data['QUALITY'] == 0]
    images = data["FLUX"][data['QUALITY'] == 0]
    
    lightcurve = np.sum(images, axis=(1, 2))
    trend = scipy.signal.savgol_filter(lightcurve, 101, polyorder=3) 
    percent_change = 100 * ((lightcurve / trend) - 1)

    frequency, power = LombScargle(time, percent_change, nterms=2).autopower(minimum_frequency=1/1.5, maximum_frequency=1/0.6, samples_per_peak=10)
    period = 1 / frequency[np.argmax(power)]
    
    plt.imshow(images[0], cmap='gray')
    
    show_plots_in_chunks(time, period, percent_change)
    
    
    normed_data = process_kepler_data(time, period, percent_change)
    
    plt.scatter(*zip(*normed_data))
    plt.show()
    
    play_midi_from_data(normed_data)
    
    multitrack_data = process_kepler_data_multi_track(time, period, percent_change)
    
    instruments_to_add = [
    'steel drums', 'rock organ', 'pizzicato strings', 'oboe', 'ocarina'
]

    multitrack_data_with_instruments = []
    
    for index, track in enumerate(multitrack_data):
        multitrack_data_with_instruments.append([instruments_to_add[index]] + track)
        
    
    # While we're at it, let's add a drum track with a solid beat
    max_number_of_beats = multitrack_data_with_instruments[0][-1][0]
    
    bass_drum = []
    for beat in range(0, int(max_number_of_beats + 1)):
       bass_drum.append((beat, 1)) 
    
    beat_track = ['bass drum 1'] + bass_drum
    multitrack_data_with_instruments.append(beat_track)
    
    play_midi_from_data(multitrack_data_with_instruments, track_type='multiple', key='c_major')


if __name__ == '__main__':
    main()
