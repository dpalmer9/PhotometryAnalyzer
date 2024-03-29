# Module Load Section
import configparser
import csv
import os
import sys
import tkinter as tk
from datetime import datetime
from tkinter import END
from tkinter import filedialog
from tkinter import ttk

import numpy as np
import pandas as pd
from scipy import signal

pd.options.mode.chained_assignment = None


class PhotometryData:
    def __init__(self):

        # Initialize Folder Path Variables
        self.curr_cpu_core_count = os.cpu_count()
        self.curr_dir = os.getcwd()
        if sys.platform == 'linux' or sys.platform == 'darwin':
            self.folder_symbol = '/'
        elif sys.platform == 'win32':
            self.folder_symbol = '\\'
        self.main_folder_path = os.getcwd()
        self.data_folder_path = self.main_folder_path + self.folder_symbol + 'Data' + self.folder_symbol

        self.abet_file_path = ''
        self.abet_file = ''

        self.doric_file_path = ''
        self.doric_file = ''

        self.anymaze_file_path = ''
        self.anymaze_file = ''

        # Initialize Boolean Variables

        self.abet_loaded = False
        self.abet_searched = False
        self.anymaze_loaded = False
        self.doric_loaded = False

        # Initialize Numeric Variables

        self.abet_doric_sync_value = 0
        self.anymaze_doric_sync_value = 0

        self.extra_prior = 0
        self.extra_follow = 0

        self.sample_frequency = 0
        self.doric_time = 0

        # Initialize Descriptor Variables

        self.date = None
        self.animal_id = None

        # Initialize String Variables

        self.event_name_col = ''
        self.time_var_name = ''
        self.event_name = ''

        # Initialize List Variables

        self.abet_time_list = []
        self.anymaze_event_times = []

        # Initialize Data Objects (Tables, Series, etc)

        self.partial_dataframe = pd.DataFrame()
        self.final_dataframe = pd.DataFrame()
        self.partial_deltaf = pd.DataFrame()
        self.final_deltaf = pd.DataFrame()
        self.partial_percent = pd.DataFrame()
        self.final_percent = pd.DataFrame()
        self.abet_pd = pd.DataFrame()
        self.doric_pd = pd.DataFrame()
        self.doric_pandas = pd.DataFrame()
        self.abet_raw_data = pd.DataFrame()
        self.anymaze_pandas = pd.DataFrame()
        self.abet_pandas = pd.DataFrame()
        self.abet_event_times = pd.DataFrame()
        self.trial_definition_times = pd.DataFrame()

    """ load_abet_data - Loads in the ABET unprocessed data to the PhotometryData object. Also
    extracts the animal ID and date. csv reader import necessary due to unusual structure of
    ABET II/ABET Cognition data structures. Once the standard data table is detected, a curated
    subset of columns is collected . Output is moved to pandas dataframe.
     Arguments:
     filepath = The filepath for the ABET unprocessed csv. Generated from GUI path """

    def load_abet_data(self, filepath):
        self.abet_file_path = filepath
        self.abet_loaded = True
        abet_file = open(self.abet_file_path)
        abet_csv_reader = csv.reader(abet_file)
        abet_data_list = list()
        abet_name_list = list()
        event_time_colname = ['Evnt_Time', 'Event_Time']
        colnames_found = False
        for row in abet_csv_reader:
            if not colnames_found:
                if len(row) == 0:
                    continue
                if row[0] == 'Animal ID':
                    self.animal_id = str(row[1])
                    continue
                if row[0] == 'Date/Time':
                    self.date = str(row[1])
                    self.date = self.date.replace(':', '-')
                    self.date = self.date.replace('/', '-')
                    continue
                if row[0] in event_time_colname:
                    colnames_found = True
                    self.time_var_name = row[0]
                    self.event_name_col = row[2]
                    # Columns are 0-time, 1-Event ID, 2-Event name, 3-Item Name, 5-Group ID, 8-Arg-1 Value
                    abet_name_list = [row[0], row[1], row[2], row[3], row[5], row[8]]
                else:
                    continue
            else:
                abet_data_list.append([row[0], row[1], row[2], row[3], row[5], row[8]])
        abet_file.close()
        abet_numpy = np.array(abet_data_list)
        self.abet_pandas = pd.DataFrame(data=abet_numpy, columns=abet_name_list)

    """ load_doric_data - Loads in the doric data to the PhotometryData object. This method uses a
    simple pandas read csv function to import the data. User specified column indexes are used to grab only
    the relevant columns.
     Arguments:
     filepath = The filepath for the doric photometry csv. Generated from GUI path
     ch1_col = The column index for the isobestic channel data
     ch2_col = The column index for the active channel data
     ttl_col = The column index for the TTL data """

    def load_doric_data(self, filepath, ch1_col, ch2_col, ttl_col):
        self.doric_file_path = filepath
        self.doric_loaded = True
        colnames = ['Time', 'Control', 'Active', 'TTL']
        doric_data = pd.read_csv(self.doric_file_path, header=1)
        self.doric_pandas = doric_data.iloc[:, [0, ch1_col, ch2_col, ttl_col]]
        self.doric_pandas.columns = colnames
        self.doric_pandas = self.doric_pandas.astype('float')

    """ load_anymaze_data - Loads in AnyMaze session data into the PhotometryData object. This method
    uses a simple pandas import to grab all of the data. Unusual strings are converted to nan values.
    Arguments:
    filepath = the filepath of the AnyMaze session data. Generated from GUI path"""

    def load_anymaze_data(self, filepath):
        self.anymaze_file_path = filepath
        self.anymaze_loaded = True
        self.anymaze_pandas = pd.read_csv(self.anymaze_file_path, header=0)
        self.anymaze_pandas = self.anymaze_pandas.replace(r'^\s*$', np.nan, regex=True)
        self.anymaze_pandas = self.anymaze_pandas.astype('float')

    """ abet_trial_definition - Defines a trial structure for the components of the ABET II unprocessed data.
    This method uses the Item names of Condition Events that represent the normal start and end of a trial epoch. This
    method was expanded in PhotometryBatch to allow for multiple start and end groups.
    Arguments:
    start_event_group = the name of an ABET II Condition Event that defines the start of a trial
    end_event_group = the name of an ABET II Condition Event that defines the end of a trial
    Photometry Analyzer currently only supports start group definitions.
    Photometry Batch supports multiple start and end group definitions
    MousePAD will eventually support all definitions as well as sessions with no definition"""

    def abet_trial_definition(self, start_event_group, end_event_group):
        if not self.abet_loaded:
            return None

        if isinstance(start_event_group, list) and isinstance(end_event_group, list):
            event_group_list = start_event_group + end_event_group
            filtered_abet = self.abet_pandas.loc[:, self.abet_pandas.Item_Name.isin(event_group_list)]
        elif isinstance(start_event_group, list) and not (isinstance(end_event_group, list)):
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas.loc[:, 'Item_Name'].isin(start_event_group)) | (
                    self.abet_pandas.loc[:, 'Item_Name'] == str(end_event_group))) &
                                                 (self.abet_pandas.loc[:, 'Evnt_ID'] == '1')]
        elif isinstance(end_event_group, list) and not (isinstance(start_event_group, list)):
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas.loc[:, 'Item_Name'] == str(start_event_group)) | (
                self.abet_pandas.loc[:, 'Item_Name'].isin(end_event_group))) &
                                                 (self.abet_pandas.loc[:, 'Evnt_ID'] == '1')]
        else:
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas.loc[:, 'Item_Name'] == str(start_event_group)) | (
                    self.abet_pandas.loc[:, 'Item_Name'] == str(end_event_group))) &
                                                 (self.abet_pandas.loc[:, 'Evnt_ID'] == '1')]

        filtered_abet = filtered_abet.reset_index(drop=True)
        if filtered_abet.iloc[0, 3] != str(start_event_group):
            filtered_abet = filtered_abet.drop([0])
            print('First Trial Event not Trial Start. Removing Instance.')
        trial_times = filtered_abet.loc[:, self.time_var_name]
        trial_times = trial_times.reset_index(drop=True)
        start_times = trial_times.iloc[::2]
        start_times = start_times.reset_index(drop=True)
        start_times = pd.to_numeric(start_times, errors='coerce')
        end_times = trial_times.iloc[1::2]
        end_times = end_times.reset_index(drop=True)
        end_times = pd.to_numeric(end_times, errors='coerce')
        self.trial_definition_times = pd.concat([start_times, end_times], axis=1)
        self.trial_definition_times.columns = ['Start_Time', 'End_Time']
        self.trial_definition_times = self.trial_definition_times.reset_index(drop=True)

    """ abet_search_event - This function searches through the ABET unprocessed data 
    for events specified in the ABET GUI. These events can be Condition Events, Variable Events,
    Touch Up/Down Events, Input Transition On/Off Events. NOTE this version of the function was
    generated in PhotoBatch and contains code for filtering primary events. This feature is not
    implemented in PhotometryAnalyzer. The output of this function is a pandas dataframe with the
    start and end times for all the identified events with the user specified padding.
    Arguments:
    start_event_id = The numerical value in the ABET II unprocessed file denoting the type of event.
    E.g. Condition Event, Variable Event
    start_event_group = The numerical value denoting the group number as defined by the ABET II
    schedule designer
    start_event_item name = The name of the specific event in the Item Name column.
    start_event_position = A numerical value denoting the positional argument of the event in the case of a
    Touch Up/Down event
    filter_event_id = The numerical value in the ABET II unprocessed file denoting the type of filter event.
    filter_event_group = The numerical value denoting the group number as defined by the ABET II schedule designer
    for the filtering event
    filter_event_item_name = The name of the specific event in the Item Name column for the filter event
    filter_event_position = A numerical value denoting the positional argument of the filter event in the case of
    a Touch Up/Down event
    filter_event = A boolean value denoting whether to check for a filter
    filter_before = A boolean value denoting whether the filter is an event preceding or following the main event
    extra_prior_time = A float value denoting the amount of time prior to the main event to pad it by
    extra_follow_time = A float value denoting the amount of time following the maine vent to pad it by
    """

    def abet_search_event(self, start_event_id='1', start_event_group='', start_event_item_name='',
                          start_event_position=None,
                          filter_event_id='1', filter_event_group='', filter_event_item_name='',
                          filter_event_position=None,
                          filter_event=False, filter_before=True,
                          extra_prior_time=0, extra_follow_time=0):

        filter_event_abet = None
        if filter_event_position is None:
            filter_event_position = ['']
        if start_event_position is None:
            start_event_position = ['']
        touch_event_names = ['Touch Up Event', 'Touch Down Event', 'Whisker - Clear Image by Position']
        condition_event_names = ['Condition Event']
        variable_event_names = ['Variable Event']

        if start_event_id in touch_event_names:
            filtered_abet = self.abet_pandas.loc[
                            (self.abet_pandas.loc[:, self.event_name_col] == str(start_event_id)) & (
                                    self.abet_pandas.loc[:, 'Group_ID'] == str(start_event_group)) &
                            (self.abet_pandas.loc[:, 'Item_Name'] == str(start_event_item_name)) & (
                                    self.abet_pandas.loc[:, 'Arg1_Value'] ==
                                    str(start_event_position)), :]

        else:
            filtered_abet = self.abet_pandas.loc[(self.abet_pandas.loc[:, self.event_name_col] == str(start_event_id)) &
                                                 (self.abet_pandas.loc[:, 'Group_ID'] == str(start_event_group)) &
                                                 (self.abet_pandas.loc[:, 'Item_Name'] == str(start_event_item_name)), :
                            ]

        if filter_event:
            if filter_event_id in condition_event_names:
                filter_event_abet = self.abet_pandas.loc[
                                    (self.abet_pandas.loc[:, self.event_name_col] == str(filter_event_id)) & (
                                            self.abet_pandas.loc[:, 'Group_ID'] == str(filter_event_group)), :]
            elif filter_event_id in variable_event_names:
                filter_event_abet = self.abet_pandas.loc[
                                    (self.abet_pandas.loc[:, self.event_name_col] == str(filter_event_id)) & (
                                            self.abet_pandas.loc[:, 'Item_Name'] == str(filter_event_item_name)), :]

        self.abet_event_times = filtered_abet.loc[:, self.time_var_name]
        self.abet_event_times = self.abet_event_times.reset_index(drop=True)
        self.abet_event_times = pd.to_numeric(self.abet_event_times, errors='coerce')

        if filter_event:
            if filter_event_id in condition_event_names:
                for index, value in self.abet_event_times.items():
                    sub_values = filter_event_abet.loc[:, self.time_var_name]
                    sub_values = sub_values.astype(dtype='float64')
                    sub_values = sub_values.sub(float(value))
                    if filter_before:
                        sub_values[sub_values > 0] = np.nan
                    else:
                        sub_values[sub_values < 0] = np.nan
                    sub_index = sub_values.abs().idxmin(skipna=True)

                    filter_value = filter_event_abet.loc[sub_index, 'Item_Name']
                    if filter_value != filter_event_item_name:
                        self.abet_event_times.iloc[index, :] = np.nan

                self.abet_event_times = self.abet_event_times.dropna()
                self.abet_event_times = self.abet_event_times.reset_index(drop=True)
            elif filter_event_id in variable_event_names:
                for index, value in self.abet_event_times.items():
                    sub_values = filter_event_abet.loc[:, self.time_var_name]
                    sub_values = sub_values.astype(dtype='float64')
                    sub_values = sub_values.sub(float(value))
                    if filter_before:
                        sub_values[sub_values > 0] = np.nan
                    else:
                        sub_values[sub_values < 0] = np.nan
                    sub_index = sub_values.abs().idxmin(skipna=True)

                    filter_value = filter_event_abet.loc[sub_index, 'Arg1_Value']
                    if filter_value != filter_event_position:
                        self.abet_event_times.iloc[index, :] = np.nan

        abet_start_times = self.abet_event_times - extra_prior_time
        abet_end_times = self.abet_event_times + extra_follow_time
        self.abet_event_times = pd.concat([abet_start_times, abet_end_times], axis=1)
        self.abet_event_times.columns = ['Start_Time', 'End_Time']
        self.event_name = start_event_item_name
        self.extra_follow = extra_follow_time
        self.extra_prior = extra_prior_time

    """anymaze_search_event_or - This function searches for relevant behavioural events in the anymaze time series 
    session data. This function will accept up to three different criteria and account for boolean and non-boolean 
    measures. This function will also account for total length of relevant events, as well as which portion of the 
    event is relevant for relating to photometry.
    Arguments (for all 3 events):
    event_name = The name of the particular event in the time series
    event_operation = The name of the mathematical operation. For Boolean is True or False. 
    For numerical <. <=, =, !=, >=, & >
    event_value = The specific value assigned to the event
    Output is a pandas dataframe with the start and end times for all events identified with the criteria.
    Arguments (general):
    event_tolerance = The amount of time that must be between two events to be considered separate
    extra_prior_time = The amount of time to pad prior to the event of interest
    extra_follow_time = The amount of time to pad following the event of interest
    event_definition = A variable that tracks which part of a continuous event is relevant for centering. 
    Options include: Event Start (first time point of event), Event Center (middle point of the event), and Event
    End (final time point of event)
    """

    def anymaze_search_event_or(self, event1_name, event1_operation, event1_value=0, event2_name='None',
                                event2_operation='None', event2_value=0, event3_name='None', event3_operation='None',
                                event3_value=0,
                                event_tolerance=1.00, extra_prior_time=0, extra_follow_time=0,
                                event_definition='Event Start'):
        def operation_search(event, operation, val=0):
            if operation == 'Active':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] == 1, :]
            elif operation == 'Inactive':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] == 0, :]
            elif operation == 'Less Than':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] < val, :]
            elif operation == 'Less Than or Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] <= val, :]
            elif operation == 'Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] == val, :]
            elif operation == 'Greater Than or Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] >= val, :]
            elif operation == 'Greater Than':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] > val, :]
            elif operation == 'Not Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] != val, :]
            else:
                return

            search_index = search_data.index
            search_index = search_index.tolist()
            return search_index

        if event1_name == 'None' and event2_name == 'None' and event3_name == 'None':
            return

        elif event2_name == 'None' and event3_name == 'None' and event1_name != 'None':
            event_index = operation_search(event1_name, event1_operation, event1_value)
        elif event3_name == 'None' and event2_name != 'None' and event1_name != 'None':
            event1_index = operation_search(event1_name, event1_operation, event1_value)
            event2_index = operation_search(event2_name, event2_operation, event2_value)
            event_index_hold = event1_index + event2_index
            event_index = list()
            for item in event_index_hold:
                if event_index_hold.count(item) >= 2:
                    if item not in event_index:
                        event_index.append(item)

        else:
            event1_index = operation_search(event1_name, event1_operation, event1_value)
            event2_index = operation_search(event2_name, event2_operation, event2_value)
            event3_index = operation_search(event3_name, event3_operation, event3_value)
            event_index_hold = event1_index + event2_index + event3_index
            event_index = list()
            for item in event_index_hold:
                if event_index_hold.count(item) >= 3:
                    if item not in event_index:
                        event_index.append(item)

        search_times = self.anymaze_pandas.loc[event_index, 'Time']
        search_times = search_times.reset_index(drop=True)

        event_start_times = list()
        event_end_times = list()

        event_start_time = self.anymaze_pandas.loc[0, 'Time']

        current_time = self.anymaze_pandas.loc[0, 'Time']

        for index, value in search_times.items():
            previous_time = current_time
            current_time = value
            if event_start_time == self.anymaze_pandas.loc[0, 'Time']:
                event_start_time = current_time
                event_start_times.append(event_start_time)
                continue
            if (current_time - previous_time) >= event_tolerance:
                event_end_time = previous_time
                event_start_time = current_time
                event_start_times.append(event_start_time)
                event_end_times.append(event_end_time)
                continue
            if index >= (search_times.size - 1):
                event_end_time = current_time
                event_end_times.append(event_end_time)
                break

        final_start_times = list()
        final_end_times = list()
        if event_definition == "Event Start":
            for time_val in event_start_times:
                final_start_time = time_val - extra_prior_time
                if final_start_time <= 0:
                    continue
                final_start_times.append(final_start_time)
                final_end_time = time_val + extra_follow_time
                final_end_times.append(final_end_time)

        elif event_definition == "Event Center":
            center_times = list()
            for index in range(0, (len(event_start_times) - 1)):
                center_time = event_end_times[index] - event_start_times[index]
                center_times.append(center_time)
            for time in center_times:
                final_start_time = time - extra_prior_time
                if final_start_time <= 0:
                    continue
                final_start_times.append(final_start_time)
                final_end_time = time + extra_follow_time
                final_end_times.append(final_end_time)

        elif event_definition == "Event End":
            for time_val in event_end_times:
                final_start_time = time_val - extra_prior_time
                if final_start_time <= 0:
                    continue
                final_start_times.append(final_start_time)
                final_end_time = time_val + extra_follow_time
                final_end_times.append(final_end_time)
        self.anymaze_event_times = pd.DataFrame(final_start_times)
        self.anymaze_event_times['End_Time'] = final_end_times
        self.anymaze_event_times.columns = ['Start_Time', 'End_Time']
        self.abet_event_times = self.anymaze_event_times

    """ abet_doric_synchronize - This function searches for TTL timestamps in the ABET II raw data and
    relates it to TTL pulses detected in the photometer. The adjusted sync value is calculated and the 
    doric photometry data time is adjusted to be in reference to the ABET II file."""

    def abet_doric_synchronize(self):
        if not self.abet_loaded:
            return None
        if not self.doric_loaded:
            return None
        try:
            doric_ttl_active = self.doric_pandas.loc[(self.doric_pandas['TTL'] > 1.00),]
        except KeyError:
            print('No TTL Signal Detected. Ending Analysis.')
            return
        try:
            abet_ttl_active = self.abet_pandas.loc[(self.abet_pandas['Item_Name'] == 'TTL #1'),]
        except KeyError:
            print('ABET II File missing TTL Pulse Output. Ending Analysis.')
            return

        doric_time = doric_ttl_active.iloc[0, 0]
        doric_time = doric_time.astype(float)
        doric_time = doric_time.item(0)
        abet_time = abet_ttl_active.iloc[0, 0]
        abet_time = float(abet_time)

        self.abet_doric_sync_value = doric_time - abet_time

        self.doric_time = pd.to_numeric(self.doric_pandas['Time'])

        self.doric_pandas['Time'] = self.doric_time - self.abet_doric_sync_value

    """ anymaze_doric_synchronize - This function searches for TTL timestamps in the Anymaze time series data and
        relates it to TTL pulses detected in the photometer. The adjusted sync value is calculated and the 
        doric photometry data time is adjusted to be in reference to the Anymaze file."""

    def anymaze_doric_synchronize_or(self):
        if not self.anymaze_loaded:
            return None
        if not self.doric_loaded:
            return None

        try:
            doric_ttl_active = self.doric_pandas.loc[(self.doric_pandas['TTL'] > 1.00),]
        except KeyError:
            print('No TTL Signal Detected. Ending Analysis.')
            return

        try:
            anymaze_ttl_active = self.anymaze_pandas.loc[(self.anymaze_pandas['TTL Pulse active'] > 0),]
        except KeyError:
            print('Anymaze File missing TTL Pulse Output. Ending Analysis.')
            return

        doric_time = doric_ttl_active.iloc[0, 0]
        doric_time = doric_time.astype(float)
        doric_time = np.asscalar(doric_time)
        anymaze_time = anymaze_ttl_active.iloc[0, 0]
        anymaze_time = float(anymaze_time)

        self.anymaze_doric_sync_value = doric_time - anymaze_time

        self.doric_time = pd.to_numeric(self.doric_pandas['Time'])

        self.doric_pandas['Time'] = self.doric_time - self.anymaze_doric_sync_value

    """doric_process - This function calculates the delta-f value based on the isobestic and active channel data.
    The two channels are first put through a 2nd order low-pass butterworth filter with a user-specified cutoff. 
    Following filtering, the data is fit with least squares regression to a linear function. Finally, the fitted data
    is used to calculate a delta-F value. A pandas dataframe with the time and delta-f values is created.
    Arguments:
    filter_frequency = The cut-off frequency used for the low-pass filter"""

    def doric_process(self, filter_frequency=6):
        doric_pandas_cut = self.doric_pandas[self.doric_pandas['Time'] >= 0]

        time_data = doric_pandas_cut['Time'].to_numpy()
        f0_data = doric_pandas_cut['Control'].to_numpy()
        f_data = doric_pandas_cut['Active'].to_numpy()

        time_data = time_data.astype(float)
        f0_data = f0_data.astype(float)
        f_data = f_data.astype(float)

        self.sample_frequency = len(time_data) / (time_data[(len(time_data) - 1)] - time_data[0])
        filter_frequency / (self.sample_frequency / 2)
        butter_filter = signal.butter(N=2, Wn=filter_frequency, btype='lowpass', analog=False, output='sos',
                                      fs=self.sample_frequency)
        filtered_f0 = signal.sosfilt(butter_filter, f0_data)
        filtered_f = signal.sosfilt(butter_filter, f_data)

        filtered_poly = np.polyfit(filtered_f0, filtered_f, 1)
        filtered_lobf = np.multiply(filtered_poly[0], filtered_f0) + filtered_poly[1]

        delta_f = (filtered_f - filtered_lobf) / filtered_lobf

        self.doric_pd = pd.DataFrame(time_data)
        self.doric_pd['DeltaF'] = delta_f
        self.doric_pd = self.doric_pd.rename(columns={0: 'Time', 1: 'DeltaF'})

    """trial_separator - This function takes the extracted photometry data and parses it using the event data obtained
    from the previous functions. This function will check to make sure the events are the same length. 
    This function will also calculate the z-scores using either the entire event or the time prior to the start of a 
    trial (i.e. iti). The result of this function are a series of pandas dataframes corresponding to the different
    output types in the write_data function.
    
    Arguments:
    whole_trial_normalize = A boolean value to determine whether to use the whole event to generate z-scores
    normalize_side = Denotes whether to use the pre or post trial data to normalize if not using whole trial.
    trial_definition = A flag to indicate whether a trial definition exists or not
    trial_iti_pad = How long in the pre-trial time space for normalization"""

    def trial_separator(self, whole_trial_normalize=True, normalize_side='Left', trial_definition=False,
                        trial_iti_pad=0, center_method='mean'):
        if not self.abet_loaded and not self.anymaze_loaded:
            return
        left_selection_list = ['Left', 'Before', 'L', 'l', 'left', 'before', 1]
        right_selection_list = ['Right', 'right', 'R', 'r', 'After', 'after', 2]

        trial_definition_none_list = ['None', 0, '0', 'No', False]
        trial_definition_ind_list = ['Individual', 1, '1', 'Ind', 'Indv']
        trial_definition_overall_list = ['Overall', 2, '2']

        trial_num = 1

        self.abet_time_list = self.abet_event_times

        length_time = self.abet_time_list.iloc[0, 1] - self.abet_time_list.iloc[0, 0]
        measurements_per_interval = length_time * self.sample_frequency
        if trial_definition in trial_definition_none_list:
            for index, row in self.abet_time_list.iterrows():

                try:
                    start_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index, 'Start_Time']).abs().idxmin()
                except IndexError:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                except IndexError:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index, 'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index, 'End_Time']:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) < measurements_per_interval:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                if not whole_trial_normalize:
                    if normalize_side in left_selection_list:
                        norm_end_time = self.abet_time_list.loc[index, 'Start_Time'] + trial_iti_pad
                        iti_deltaf = trial_deltaf.loc[
                            trial_deltaf['Time'] < norm_end_time, 'DeltaF']
                    elif normalize_side in right_selection_list:
                        norm_start_time = self.abet_time_list.loc[index, 'End_Time'] - trial_iti_pad
                        iti_deltaf = trial_deltaf.loc[
                            trial_deltaf['Time'] > norm_start_time, 'DeltaF']
                    if center_method == 'mean':
                        z_mean = iti_deltaf.mean()
                        z_sd = iti_deltaf.std()
                    elif center_method == 'median':
                        z_mean = iti_deltaf.median()
                        z_dev = np.absolute(np.subtract(iti_deltaf, z_mean))
                        z_sd = z_dev.median()
                else:
                    deltaf_split = trial_deltaf.loc[:, 'DeltaF']
                    if center_method == 'mean':
                        z_mean = deltaf_split.mean()
                        z_sd = deltaf_split.std()
                    elif center_method == 'median':
                        z_mean = deltaf_split.median()
                        z_dev = np.absolute(np.subtract(deltaf_split, z_mean))
                        z_sd = z_dev.median()

                trial_deltaf.loc[:, 'zscore'] = (trial_deltaf.loc[:, 'DeltaF'] - z_mean) / z_sd
                trial_deltaf.loc[:, 'percent_change'] = trial_deltaf.loc[:, 'DeltaF'].map(
                    lambda x: ((x - z_mean) / abs(z_mean)) * 100)

                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)
                colname_3 = 'Delta-F Trial ' + str(trial_num)
                colname_4 = 'Percent-Change Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})

                    self.partial_deltaf = trial_deltaf.loc[:, 'DeltaF']
                    self.partial_deltaf = self.partial_deltaf.to_frame()
                    self.partial_deltaf = self.partial_deltaf.reset_index(drop=True)
                    self.partial_deltaf = self.partial_deltaf.rename(columns={'DeltaF': colname_2})

                    self.final_deltaf = trial_deltaf.loc[:, ('Time', 'DeltaF')]
                    self.final_deltaf = self.final_deltaf.reset_index(drop=True)
                    self.final_deltaf = self.final_deltaf.rename(columns={'Time': colname_1, 'DeltaF': colname_2})

                    self.partial_percent = trial_deltaf.loc[:, 'percent_change']
                    self.partial_percent = self.partial_percent.to_frame()
                    self.partial_percent = self.partial_percent.reset_index(drop=True)
                    self.partial_percent = self.partial_percent.rename(columns={'percent_change': colname_2})

                    self.final_percent = trial_deltaf.loc[:, ('Time', 'percent_change')]
                    self.final_percent = self.final_percent.reset_index(drop=True)
                    self.final_percent = self.final_percent.rename(
                        columns={'Time': colname_1, 'percent_change': colname_2})

                    trial_num += 1
                else:
                    trial_deltaf = trial_deltaf.reset_index(drop=True)
                    dataframe_len = len(self.final_dataframe.index)
                    trial_len = len(trial_deltaf.index)
                    if trial_len > dataframe_len:
                        len_diff = trial_len - dataframe_len
                        new_index = list(range(dataframe_len, (dataframe_len + len_diff)))
                        self.final_dataframe = self.final_dataframe.reindex(
                            self.final_dataframe.index.union(new_index))
                        self.partial_dataframe = self.partial_dataframe.reindex(
                            self.partial_dataframe.index.union(new_index))
                        self.partial_deltaf = self.partial_deltaf.reindex(
                            self.partial_deltaf.index.union(new_index))
                        self.final_deltaf = self.final_deltaf.reindex(
                            self.final_deltaf.index.union(new_index))
                        self.partial_percent = self.partial_percent.reindex(
                            self.partial_percent.index.union(new_index))
                        self.final_percent = self.final_percent.reindex(
                            self.final_percent.index.union(new_index))

                    trial_deltaf = trial_deltaf.rename(columns={'Time': colname_1, 'zscore': colname_2,
                                                                'DeltaF': colname_3, 'percent_change': colname_4})

                    self.partial_dataframe = pd.concat([self.partial_dataframe, trial_deltaf.loc[:, colname_2]],
                                                       axis=1)
                    self.partial_deltaf = pd.concat([self.partial_deltaf, trial_deltaf.loc[:, colname_3]],
                                                    axis=1)
                    self.final_dataframe = pd.concat(
                        [self.final_dataframe, trial_deltaf.loc[:, colname_1], trial_deltaf.loc[:, colname_2]],
                        axis=1)
                    self.final_deltaf = pd.concat([self.final_deltaf, trial_deltaf.loc[:, colname_1],
                                                   trial_deltaf.loc[:, colname_3]], axis=1)
                    self.partial_percent = pd.concat([self.partial_percent, trial_deltaf.loc[:, colname_4]],
                                                     axis=1)
                    self.final_percent = pd.concat(
                        [self.final_percent, trial_deltaf.loc[:, colname_1], trial_deltaf.loc[:, colname_4]],
                        axis=1)
                    trial_num += 1

        elif trial_definition in trial_definition_ind_list:
            for index, row in self.abet_time_list.iterrows():
                try:
                    start_index = self.doric_pd.loc[:, 'Time'].sub(
                        self.abet_time_list.loc[index, 'Start_Time']).abs().idxmin()
                except IndexError:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd.loc[:, 'Time'].sub(
                        self.abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                except IndexError:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index, 'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index, 'End_Time']:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) < measurements_per_interval:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                if not whole_trial_normalize:
                    if normalize_side in left_selection_list:
                        trial_start_index_diff = self.trial_definition_times.loc[:, 'Start_Time'].sub(
                            (self.abet_time_list.loc[index, 'Start_Time'] + self.extra_prior))  # .abs().idxmin()
                        trial_start_index_diff[trial_start_index_diff > 0] = np.nan
                        trial_start_index = trial_start_index_diff.abs().idxmin(skipna=True)
                        trial_start_window = self.trial_definition_times.iloc[trial_start_index, 0]
                        trial_iti_window = trial_start_window - float(trial_iti_pad)
                        iti_data = self.doric_pd.loc[(self.doric_pd.loc[:, 'Time'] >= trial_iti_window) & (
                                self.doric_pd.loc[:, 'Time'] <= trial_start_window), 'DeltaF']
                    elif normalize_side in right_selection_list:
                        trial_end_index = self.trial_definition_times.loc[:, 'End_Time'].sub(
                            self.abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                        trial_end_window = self.trial_definition_times.iloc[trial_end_index, 0]
                        trial_iti_window = trial_end_window + trial_iti_pad
                        iti_data = self.doric_pd.loc[(self.doric_pd.loc[:, 'Time'] >= trial_end_window) & (
                                self.doric_pd.loc[:, 'Time'] <= trial_iti_window), 'DeltaF']
                    else:
                        print('no specified side to normalize')
                        return
                    if center_method == 'mean':
                        z_mean = iti_data.mean()
                        z_sd = iti_data.std()
                    elif center_method == 'median':
                        z_mean = iti_data.median()
                        z_dev = np.absolute(np.subtract(iti_data, z_mean))
                        z_sd = z_dev.median()
                    else:
                        deltaf_split = trial_deltaf.loc[:, 'DeltaF']
                        if center_method == 'mean':
                            z_mean = deltaf_split.mean()
                            z_sd = deltaf_split.std()
                        elif center_method == 'median':
                            z_mean = deltaf_split.median()
                            z_dev = np.absolute(np.subtract(deltaf_split, z_mean))
                            z_sd = z_dev.median()

                z_score = trial_deltaf.loc[:, 'DeltaF']
                z_score = z_score.map(lambda x: ((x - z_mean) / z_sd))
                trial_deltaf.loc[:, 'zscore'] = z_score
                percent = trial_deltaf.loc[:, 'DeltaF']
                percent = percent.map(lambda x: ((x - z_mean) / abs(z_mean)) * 100)
                trial_deltaf.loc[:, 'percent_change'] = percent

                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)
                colname_3 = 'Delta-F Trial ' + str(trial_num)
                colname_4 = 'Percent-Change Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})

                    self.partial_deltaf = trial_deltaf.loc[:, 'DeltaF']
                    self.partial_deltaf = self.partial_deltaf.to_frame()
                    self.partial_deltaf = self.partial_deltaf.reset_index(drop=True)
                    self.partial_deltaf = self.partial_deltaf.rename(columns={'DeltaF': colname_3})

                    self.final_deltaf = trial_deltaf.loc[:, ('Time', 'DeltaF')]
                    self.final_deltaf = self.final_deltaf.reset_index(drop=True)
                    self.final_deltaf = self.final_deltaf.rename(columns={'Time': colname_1, 'DeltaF': colname_3})

                    self.partial_percent = trial_deltaf.loc[:, 'percent_change']
                    self.partial_percent = self.partial_percent.to_frame()
                    self.partial_percent = self.partial_percent.reset_index(drop=True)
                    self.partial_percent = self.partial_percent.rename(columns={'percent_change': colname_4})

                    self.final_percent = trial_deltaf.loc[:, ('Time', 'percent_change')]
                    self.final_percent = self.final_percent.reset_index(drop=True)
                    self.final_percent = self.final_percent.rename(
                        columns={'Time': colname_1, 'percent_change': colname_4})

                    trial_num += 1
                else:
                    trial_deltaf = trial_deltaf.reset_index(drop=True)
                    dataframe_len = len(self.final_dataframe.index)
                    trial_len = len(trial_deltaf.index)
                    if trial_len > dataframe_len:
                        len_diff = trial_len - dataframe_len
                        new_index = list(range(dataframe_len, (dataframe_len + len_diff)))
                        self.final_dataframe = self.final_dataframe.reindex(
                            self.final_dataframe.index.union(new_index))
                        self.partial_dataframe = self.partial_dataframe.reindex(
                            self.partial_dataframe.index.union(new_index))
                        self.partial_deltaf = self.partial_deltaf.reindex(
                            self.partial_deltaf.index.union(new_index))
                        self.final_deltaf = self.final_deltaf.reindex(
                            self.final_deltaf.index.union(new_index))
                        self.partial_percent = self.partial_percent.reindex(
                            self.partial_percent.index.union(new_index))
                        self.final_percent = self.final_percent.reindex(
                            self.final_percent.index.union(new_index))

                    trial_deltaf = trial_deltaf.rename(columns={'Time': colname_1, 'zscore': colname_2,
                                                                'DeltaF': colname_3, 'percent_change': colname_4})

                    self.partial_dataframe = pd.concat([self.partial_dataframe, trial_deltaf.loc[:, colname_2]],
                                                       axis=1)
                    self.partial_deltaf = pd.concat([self.partial_deltaf, trial_deltaf.loc[:, colname_3]],
                                                    axis=1)
                    self.final_dataframe = pd.concat(
                        [self.final_dataframe, trial_deltaf.loc[:, colname_1], trial_deltaf.loc[:, colname_2]],
                        axis=1)
                    self.final_deltaf = pd.concat([self.final_deltaf, trial_deltaf.loc[:, colname_1],
                                                   trial_deltaf.loc[:, colname_3]], axis=1)
                    self.partial_percent = pd.concat([self.partial_percent, trial_deltaf.loc[:, colname_4]],
                                                     axis=1)
                    self.final_percent = pd.concat(
                        [self.final_percent, trial_deltaf.loc[:, colname_1], trial_deltaf.loc[:, colname_4]],
                        axis=1)
                    trial_num += 1

        elif trial_definition in trial_definition_overall_list:
            mod_trial_times = self.trial_definition_times
            mod_trial_times.iloc[-1, 1] = np.nan
            mod_trial_times.iloc[0, 0] = np.nan
            mod_trial_times['Start_Time'] = mod_trial_times['Start_Time'].shift(-1)
            mod_trial_times = mod_trial_times[:-1]
            for index, row in mod_trial_times.iterrows():
                try:
                    end_index = self.doric_pd.loc[:, 'Time'].sub(
                        mod_trial_times.loc[index, 'Start_Time']).abs().idxmin()
                except IndexError:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    start_index = self.doric_pd.loc[:, 'Time'].sub(
                        mod_trial_times.loc[index, 'End_Time']).abs().idxmin()
                except IndexError:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > mod_trial_times.loc[index, 'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < mod_trial_times.loc[index, 'End_Time']:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) < measurements_per_interval:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                iti_deltaf = self.doric_pd.iloc[start_index:end_index]
                iti_deltaf = iti_deltaf.loc[:, 'DeltaF']
                if index == 0:
                    full_iti_deltaf = iti_deltaf
                else:
                    full_iti_deltaf = full_iti_deltaf.append(iti_deltaf)

            if center_method == 'mean':
                z_mean = full_iti_deltaf.mean()
                z_sd = full_iti_deltaf.std()
            elif center_method == 'median':
                z_mean = full_iti_deltaf.median()
                z_sd = full_iti_deltaf.std()

            for index, row in self.abet_time_list.iterrows():
                try:
                    start_index = self.doric_pd.loc[:, 'Time'].sub(
                        self.abet_time_list.loc[index, 'Start_Time']).abs().idxmin()
                except:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd.loc[:, 'Time'].sub(
                        self.abet_time_list.loc[index, 'End_Time']).abs().idxmin()
                except:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index, 'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index, 'End_Time']:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) < measurements_per_interval:
                    end_index += 1

                while len(range(start_index, (end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                trial_deltaf.loc[:, 'zscore'] = trial_deltaf.loc[:, 'DeltaF'].map(lambda x: ((x - z_mean) / z_sd))
                trial_deltaf.loc[:, 'percent_change'] = trial_deltaf.loc[:, 'DeltaF'].map(
                    lambda x: ((x - z_mean) / abs(z_mean)) * 100)
                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)
                colname_3 = 'Delta-F Trial ' + str(trial_num)
                colname_4 = 'Percent-Change Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})

                    self.partial_deltaf = trial_deltaf.loc[:, 'DeltaF']
                    self.partial_deltaf = self.partial_deltaf.to_frame()
                    self.partial_deltaf = self.partial_deltaf.reset_index(drop=True)
                    self.partial_deltaf = self.partial_deltaf.rename(columns={'DeltaF': colname_2})

                    self.final_deltaf = trial_deltaf.loc[:, ('Time', 'DeltaF')]
                    self.final_deltaf = self.final_deltaf.to_frame()
                    self.final_deltaf = self.final_deltaf.reset_index(drop=True)
                    self.final_deltaf = self.final_deltaf.rename(columns={'Time': colname_1, 'DeltaF': colname_2})

                    self.partial_percent = trial_deltaf.loc[:, 'percent_change']
                    self.partial_percent = self.partial_percent.to_frame()
                    self.partial_percent = self.partial_percent.reset_index(drop=True)
                    self.partial_percent = self.partial_percent.rename(columns={'percent_change': colname_2})

                    self.final_percent = trial_deltaf.loc[:, ('Time', 'percent_change')]
                    self.final_percent = self.final_percent.reset_index(drop=True)
                    self.final_percent = self.final_percent.rename(
                        columns={'Time': colname_1, 'percent_change': colname_2})

                    trial_num += 1
                else:
                    trial_deltaf = trial_deltaf.reset_index(drop=True)
                    dataframe_len = len(self.final_dataframe.index)
                    trial_len = len(trial_deltaf.index)
                    if trial_len > dataframe_len:
                        len_diff = trial_len - dataframe_len
                        new_index = list(range(dataframe_len, (dataframe_len + len_diff)))
                        self.final_dataframe = self.final_dataframe.reindex(
                            self.final_dataframe.index.union(new_index))
                        self.partial_dataframe = self.partial_dataframe.reindex(
                            self.partial_dataframe.index.union(new_index))
                        self.partial_deltaf = self.partial_deltaf.reindex(
                            self.partial_deltaf.index.union(new_index))
                        self.final_deltaf = self.final_deltaf.reindex(
                            self.final_deltaf.index.union(new_index))
                        self.partial_percent = self.partial_percent.reindex(
                            self.partial_percent.index.union(new_index))
                        self.final_percent = self.final_percent.reindex(
                            self.final_percent.index.union(new_index))

                    trial_deltaf = trial_deltaf.rename(columns={'Time': colname_1, 'zscore': colname_2,
                                                                'DeltaF': colname_3, 'percent_change': colname_4})

                    self.partial_dataframe = pd.concat([self.partial_dataframe, trial_deltaf.loc[:, colname_2]],
                                                       axis=1)
                    self.partial_deltaf = pd.concat([self.partial_deltaf, trial_deltaf.loc[:, colname_3]],
                                                    axis=1)
                    self.final_dataframe = pd.concat(
                        [self.final_dataframe, trial_deltaf.loc[:, colname_1], trial_deltaf.loc[:, colname_2]],
                        axis=1)
                    self.final_deltaf = pd.concat([self.final_deltaf, trial_deltaf.loc[:, colname_1],
                                                   trial_deltaf.loc[:, colname_3]], axis=1)
                    self.partial_percent = pd.concat([self.partial_percent, trial_deltaf.loc[:, colname_4]],
                                                     axis=1)
                    self.final_percent = pd.concat(
                        [self.final_percent, trial_deltaf.loc[:, colname_1], trial_deltaf.loc[:, colname_4]],
                        axis=1)
                    trial_num += 1

    """write_data - This function writes the relevant output to csv files. Can output several different types of csv
    files depending on the flag the user has provided. Default output string includes the animal id, date, and event
    associated with the particular analysis. The GUI object PhotometryGUI overrides this with a better naming
    convention. Overall conventions improved in PhotometryBatch.
    Arguments:
    output_data = The structure that is requested for output. can include non-transformed (Full),
    event only (Simple), and event + time (Timed)
    filename_override = A string that will override any default file naming conventions."""

    def write_data(self, output_data, filename_override=''):
        processed_list = [1, 'Full', 'full']
        partial_list = [3, 'Simple', 'simple']
        final_list = [5, 'Timed', 'timed']
        partialf_list = [2, 'SimpleF', 'simplef']
        finalf_list = [5, 'TimedF', 'timedf']
        partialp_list = [3, 'SimpleP', 'simplep']
        finalp_list = [6, 'TimedP', 'timedp']

        output_folder = self.main_folder_path + self.folder_symbol + 'Output'
        if not (os.path.isdir(output_folder)):
            os.mkdir(output_folder)
        if self.abet_loaded is True and self.anymaze_loaded is False:
            file_path_string = output_folder + self.folder_symbol + output_data + '-' + self.animal_id + ' ' + \
                               self.date + ' ' + self.event_name + '.csv'
        else:
            current_time = datetime.now()
            current_time_string = current_time.strftime('%d-%m-%Y %H-%M-%S')
            file_path_string = output_folder + self.folder_symbol + output_data + '-' + current_time_string + '.csv'

        if filename_override != '':
            file_path_string = output_folder + self.folder_symbol + filename_override

        print(file_path_string)
        if output_data in processed_list:
            self.doric_pd.to_csv(file_path_string, index=False)
        elif output_data in partial_list:
            self.partial_dataframe.to_csv(file_path_string, index=False)
        elif output_data in final_list:
            self.final_dataframe.to_csv(file_path_string, index=False)
        elif output_data in partialf_list:
            self.partial_deltaf.to_csv(file_path_string, index=False)
        elif output_data in finalf_list:
            self.final_deltaf.to_csv(file_path_string, index=False)
        elif output_data in partialp_list:
            self.partial_percent.to_csv(file_path_string, index=False)
        elif output_data in finalp_list:
            self.final_percent.to_csv(file_path_string, index=False)


class PhotometryGUI:
    def __init__(self):
        self.anymaze_event2_operation = None
        self.anymaze_event2_colname = None
        self.anymaze_event1_value = None
        self.anymaze_event1_operation = None
        self.anymaze_pandas = None
        self.abet_pandas = None
        self.abet_event_name_col = None
        self.iti_zscoring = None
        self.abet_event_gui = None
        self.abet_event_title = None
        self.event_id_type_label = None
        self.event_id_type_entry = None
        self.event_name_label = None
        self.event_name_entry = None
        self.event_group_label = None
        self.event_group_entry = None
        self.event_position_label = None
        self.event_position_entry = None
        self.event_prior_time = None
        self.event_prior_entry = None
        self.event_follow_time = None
        self.event_follow_entry = None
        self.abet_trial_definition_title = None
        self.abet_trial_start_group_label = None
        self.abet_trial_start_entry = None
        self.abet_trial_end_group_label = None
        self.abet_trial_end_entry = None
        self.abet_trial_iti_prior_label = None
        self.abet_trial_iti_entry = None
        self.abet_iti_zscore_checkbutton = None
        self.abet_event_finish_button = None
        self.anymaze_event_gui = None
        self.anymaze_event_title = None
        self.anymaze_event_type_colname = None
        self.anymaze_event_operation_colname = None
        self.anymaze_event_value_colname = None
        self.anymaze_event1_colname = None
        self.anymaze_event2_value = None
        self.anymaze_event3_colname = None
        self.anymaze_event3_operation = None
        self.anymaze_finish_button = None
        self.anymaze_event_pad_entry = None
        self.anymaze_event_pad_label = None
        self.anymaze_event_location_entry = None
        self.anymaze_event_location_list = None
        self.anymaze_event_location_label = None
        self.anymaze_extra_follow_entry = None
        self.anymaze_extra_follow_label = None
        self.anymaze_extra_prior_entry = None
        self.anymaze_extra_prior_label = None
        self.anymaze_tolerance_entry = None
        self.anymaze_tolerance_label = None
        self.anymaze_settings_label = None
        self.anymaze_event3_value = None
        self.settings_gui = None
        self.settings_title = None
        self.channel_control_label = None
        self.channel_control_entry = None
        self.channel_active_label = None
        self.channel_active_entry = None
        self.channel_ttl_label = None
        self.channel_ttl_entry = None
        self.low_pass_freq_label = None
        self.low_pass_freq_entry = None
        self.centered_z_checkbutton = None
        self.settings_finish_button = None
        self.error_window = None
        self.error_text = None
        self.error_button = None
        self.output_path = None
        self.photometry_object = None
        self.confirmation_window = None
        self.confirmation_text = None
        self.confirmation_button = None
        if sys.platform == 'linux' or sys.platform == 'darwin':
            self.folder_symbol = '/'
        elif sys.platform == 'win32':
            self.folder_symbol = '\\'

        self.root = tk.Tk()

        self.simple_var = tk.IntVar()
        self.timed_var = tk.IntVar()
        self.full_var = tk.IntVar()
        self.simplef_var = tk.IntVar()
        self.timedf_var = tk.IntVar()
        self.centered_z_var = tk.IntVar()

        self.doric_file_path = ''
        self.anymaze_file_path = ''
        self.abet_file_path = ''

        self.event_id_var = ''
        self.event_group_var = ''
        self.event_name_var = ''
        self.event_position_var = ''
        self.event_prior_var = ''
        self.event_follow_var = ''
        self.abet_trial_start_var = ''
        self.abet_trial_end_var = ''
        self.abet_trial_iti_var = ''
        self.channel_control_var = ''
        self.channel_active_var = ''
        self.channel_ttl_var = ''
        self.low_pass_var = ''
        self.iti_normalize = 1

        self.event_id_index = 0
        self.event_group_index = 0
        self.event_name_index = 0
        self.abet_trial_start_index = 0
        self.abet_trial_end_index = 0
        self.event_position_index = 0
        self.channel_control_index = 0
        self.channel_active_index = 0
        self.channel_ttl_index = 0

        self.doric_name_list = ['']
        self.abet_event_types = ['']
        self.abet_event_type_pos = 0
        self.abet_group_name = ['']
        self.abet_group_name_pos = 0
        self.abet_group_numbers = ['']
        self.abet_group_numbers_pos = 0
        self.abet_trial_stages = ['']
        self.abet_iti_group_name = ['']
        self.touch_event_names = ['Touch Up Event', 'Touch Down Event', 'Whisker - Clear Image by Position']
        self.position_numbers = ['']
        self.position_state = 'disabled'
        self.event_time_colname = ['Evnt_Time', 'Event_Time']
        self.event_name_colname = ['Event_Name', 'Evnt_Name']

        self.anymaze_boolean_list = ['Active', 'Inactive']
        self.anymaze_operation_list = ['Less Than', 'Less Than or Equal', 'Equal', 'Greater Than or Equal',
                                       'Greater Than', 'Not Equal']
        self.anymaze_column_names = ['None']
        self.anymaze_event1_bool = False
        self.anymaze_event2_bool = False
        self.anymaze_event3_bool = False
        self.anymaze_event1_value_state = 'disabled'
        self.anymaze_event2_value_state = 'disabled'
        self.anymaze_event3_value_state = 'disabled'
        self.anymaze_event1_operation_state = 'disabled'
        self.anymaze_event2_operation_state = 'disabled'
        self.anymaze_event3_operation_state = 'disabled'
        self.anymaze_event1_column_var = 'None'
        self.anymaze_event2_column_var = 'None'
        self.anymaze_event3_column_var = 'None'
        self.anymaze_event1_operation_var = 'None'
        self.anymaze_event2_operation_var = 'None'
        self.anymaze_event3_operation_var = 'None'
        self.anymaze_event1_value_var = 'None'
        self.anymaze_event2_value_var = 'None'
        self.anymaze_event3_value_var = 'None'
        self.anymaze_event1_operation_list = 'None'
        self.anymaze_event2_operation_list = 'None'
        self.anymaze_event3_operation_list = 'None'
        self.anymaze_tolerance_var = 0
        self.anymaze_extra_prior_var = 0
        self.anymaze_extra_follow_var = 0
        self.anymaze_event_location_var = 'Event Start'
        self.anymaze_event1_column_index = 0
        self.anymaze_event2_column_index = 0
        self.anymaze_event3_column_index = 0
        self.anymaze_event1_operation_index = 0
        self.anymaze_event2_operation_index = 0
        self.anymaze_event3_operation_index = 0
        self.anymaze_event1_value_index = 0
        self.anymaze_event2_value_index = 0
        self.anymaze_event3_value_index = 0
        self.anymaze_event_location_index = 0
        self.anymaze_event_pad_var = 0

        self.curr_dir = os.getcwd()

        self.config_path = self.curr_dir + self.folder_symbol + 'Config.ini'
        self.config_file = configparser.ConfigParser()
        self.config_file.read(self.config_path)

        self.doric_file_path = self.config_file['Filepaths']['doric_file_path']
        self.abet_file_path = self.config_file['Filepaths']['abet_file_path']
        self.anymaze_file_path = self.config_file['Filepaths']['anymaze_file_path']
        self.event_id_var = self.config_file['AbetII']['event_type']
        self.event_name_var = self.config_file['AbetII']['event_name']
        self.event_group_var = self.config_file['AbetII']['event_group']
        self.event_position_var = self.config_file['AbetII']['event_position']
        self.event_prior_var = self.config_file['AbetII']['event_prior']
        self.event_follow_var = self.config_file['AbetII']['event_follow']
        self.abet_trial_start_var = self.config_file['AbetII']['abet_trial_start']
        self.abet_trial_end_var = self.config_file['AbetII']['abet_trial_end']
        self.abet_trial_iti_var = self.config_file['AbetII']['abet_iti_length']
        self.anymaze_event1_column_var = str(self.config_file['Anymaze']['event1_col'])
        self.anymaze_event1_operation_var = str(self.config_file['Anymaze']['event1_op'])
        self.anymaze_event1_value_var = str(self.config_file['Anymaze']['event1_val'])
        self.anymaze_event2_column_var = str(self.config_file['Anymaze']['event2_col'])
        self.anymaze_event2_operation_var = str(self.config_file['Anymaze']['event2_op'])
        self.anymaze_event2_value_var = str(self.config_file['Anymaze']['event2_val'])
        self.anymaze_event3_column_var = str(self.config_file['Anymaze']['event3_col'])
        self.anymaze_event3_operation_var = str(self.config_file['Anymaze']['event3_op'])
        self.anymaze_event3_value_var = str(self.config_file['Anymaze']['event3_val'])
        self.anymaze_tolerance_var = self.config_file['Anymaze']['tolerance']
        self.anymaze_extra_prior_var = self.config_file['Anymaze']['event_prior']
        self.anymaze_extra_follow_var = self.config_file['Anymaze']['event_follow']
        self.anymaze_event_location_var = self.config_file['Anymaze']['centering']
        self.anymaze_event_pad_var = self.config_file['Anymaze']['event_pad']
        self.anymaze_event1_boolean = False
        self.anymaze_event2_boolean = False
        self.anymaze_event3_boolean = False
        self.channel_control_var = self.config_file['Doric']['control_channel']
        self.channel_active_var = self.config_file['Doric']['active_channel']
        self.channel_ttl_var = self.config_file['Doric']['ttl_channel']
        self.low_pass_var = self.config_file['Doric']['low_pass']
        if self.config_file['Doric']['centered_z'] != '':
            self.centered_z_var.set(int(self.config_file['Doric']['centered_z']))
        else:
            self.centered_z_var.set(0)
        self.title = tk.Label(self.root, text='Photometry Analyzer')
        self.title.grid(row=0, column=1)

        self.doric_label = tk.Label(self.root, text='Doric Filepath:')
        self.doric_label.grid(row=1, column=0)
        self.doric_field = tk.Entry(self.root)
        self.doric_field.grid(row=1, column=1)
        self.doric_button = tk.Button(self.root, text='...', command=self.doric_file_load)
        self.doric_button.grid(row=1, column=2)

        self.abet_label = tk.Label(self.root, text='ABET II Filepath:')
        self.abet_label.grid(row=2, column=0)
        self.abet_field = tk.Entry(self.root)
        self.abet_field.grid(row=2, column=1)
        self.abet_field.insert(END, self.abet_file_path)
        self.abet_button = tk.Button(self.root, text='...', command=self.abet_file_load)
        self.abet_button.grid(row=2, column=2)

        self.anymaze_label = tk.Label(self.root, text='Anymaze Filepath:')
        self.anymaze_label.grid(row=3, column=0)
        self.anymaze_field = tk.Entry(self.root)
        self.anymaze_field.grid(row=3, column=1)
        self.anymaze_field.insert(END, self.anymaze_file_path)
        self.anymaze_button = tk.Button(self.root, text='...', command=self.anymaze_file_load)
        self.anymaze_button.grid(row=3, column=2)

        self.abet_event_button = tk.Button(self.root, text='ABET Events', command=self.abet_event_definition_gui)
        self.abet_event_button.grid(row=4, column=0)

        self.anymaze_event_button = tk.Button(self.root, text='Anymaze Events',
                                              command=self.anymaze_event_description_gui)
        self.anymaze_event_button.grid(row=4, column=1)

        self.settings_button = tk.Button(self.root, text='Settings', command=self.settings_menu)
        self.settings_button.grid(row=4, column=2)

        self.output_title = tk.Label(self.root, text='Output')
        self.output_title.grid(row=6, column=1)

        self.simplef_output_check = tk.Checkbutton(self.root, text='Simple Delta-F', variable=self.simplef_var)
        self.simplef_output_check.grid(row=7, column=1)
        self.simple_output_check = tk.Checkbutton(self.root, text='Simple Z', variable=self.simple_var)
        self.simple_output_check.grid(row=7, column=2)
        self.timedf_output_check = tk.Checkbutton(self.root, text='Timed Delta-F', variable=self.timedf_var)
        self.timedf_output_check.grid(row=8, column=0)
        self.timed_output_check = tk.Checkbutton(self.root, text='Timed Z', variable=self.timed_var)
        self.timed_output_check.grid(row=8, column=1)
        self.full_output_check = tk.Checkbutton(self.root, text='Full Output', variable=self.full_var)
        self.full_output_check.grid(row=7, column=0)

        self.run_button = tk.Button(self.root, text='Run', command=self.run_photometry_analysis)
        self.run_button.grid(row=9, column=1)

        if self.doric_file_path != '':
            self.doric_file_load(path=self.doric_file_path)
        if self.abet_file_path != '':
            self.abet_file_load(path=self.abet_file_path)
        if self.anymaze_file_path != '':
            self.anymaze_file_load(path=self.anymaze_file_path)

        self.root.protocol("WM_DELETE_WINDOW", self.close_program)
        self.root.mainloop()

    def close_program(self):

        self.config_file['Filepaths']['doric_file_path'] = self.doric_file_path
        self.config_file['Filepaths']['abet_file_path'] = self.abet_file_path
        self.config_file['Filepaths']['anymaze_file_path'] = self.anymaze_file_path
        self.config_file['AbetII']['event_type'] = self.event_id_var
        self.config_file['AbetII']['event_name'] = self.event_name_var
        self.config_file['AbetII']['event_group'] = self.event_group_var
        self.config_file['AbetII']['event_position'] = self.event_position_var
        self.config_file['AbetII']['event_prior'] = self.event_prior_var
        self.config_file['AbetII']['event_follow'] = self.event_follow_var
        self.config_file['AbetII']['abet_trial_start'] = self.abet_trial_start_var
        self.config_file['AbetII']['abet_trial_end'] = self.abet_trial_end_var
        self.config_file['AbetII']['abet_iti_length'] = self.abet_trial_iti_var
        self.config_file['Anymaze']['event1_col'] = self.anymaze_event1_column_var
        self.config_file['Anymaze']['event1_op'] = self.anymaze_event1_operation_var
        self.config_file['Anymaze']['event1_val'] = self.anymaze_event1_value_var
        self.config_file['Anymaze']['event2_col'] = self.anymaze_event2_column_var
        self.config_file['Anymaze']['event2_op'] = self.anymaze_event2_operation_var
        self.config_file['Anymaze']['event2_val'] = self.anymaze_event2_value_var
        self.config_file['Anymaze']['event3_col'] = self.anymaze_event3_column_var
        self.config_file['Anymaze']['event3_op'] = self.anymaze_event3_operation_var
        self.config_file['Anymaze']['event3_val'] = self.anymaze_event3_value_var
        self.config_file['Anymaze']['tolerance'] = self.anymaze_tolerance_var
        self.config_file['Anymaze']['event_prior'] = self.anymaze_extra_prior_var
        self.config_file['Anymaze']['event_follow'] = self.anymaze_extra_follow_var
        self.config_file['Anymaze']['centering'] = self.anymaze_event_location_var
        self.config_file['Anymaze']['event_pad'] = self.anymaze_event_pad_var
        self.config_file['Doric']['control_channel'] = self.channel_control_var
        self.config_file['Doric']['active_channel'] = self.channel_active_var
        self.config_file['Doric']['ttl_channel'] = self.channel_ttl_var
        self.config_file['Doric']['low_pass'] = self.low_pass_var
        self.config_file['Doric']['centered_z'] = str(self.centered_z_var.get())

        with open(self.config_path, 'w') as configfile:
            self.config_file.write(configfile)

        self.root.destroy()

    def abet_setting_load(self):
        if self.event_id_var in self.abet_event_types:
            self.event_id_index = self.abet_event_types.index(self.event_id_var)
            self.abet_group_name = self.abet_pandas.loc[
                self.abet_pandas[self.abet_event_name_col] == self.event_id_var, 'Item_Name']
            self.abet_group_name = self.abet_group_name.unique()
            self.abet_group_name = list(self.abet_group_name)
            self.abet_group_name = sorted(self.abet_group_name)
            if self.event_name_var in self.abet_group_name:
                self.event_name_index = self.abet_group_name.index(self.event_name_var)
                self.abet_group_numbers = self.abet_pandas.loc[
                    (self.abet_pandas[self.abet_event_name_col] == self.event_id_var) &
                    (self.abet_pandas['Item_Name'] == self.event_name_var), 'Group_ID']
                self.abet_group_numbers = self.abet_group_numbers.unique()
                self.abet_group_numbers = list(self.abet_group_numbers)
                self.abet_group_numbers = sorted(self.abet_group_numbers)
                if self.event_group_var in self.abet_group_numbers:
                    self.event_group_index = self.abet_group_numbers.index(self.event_group_var)
                    if self.event_id_var in self.touch_event_names:
                        self.position_numbers = self.abet_pandas.loc[
                            (self.abet_pandas[self.abet_event_name_col] == self.event_id_var) &
                            (self.abet_pandas['Item_Name'] == self.event_name_var) &
                            (self.abet_pandas['Group_ID'] == self.event_group_var), 'Arg1_Value']
                        self.position_numbers = self.position_numbers.unique()
                        self.position_numbers = list(self.position_numbers)
                        self.position_numbers = sorted(self.position_numbers)
                        if self.event_position_var in self.position_numbers:
                            self.event_position_index = self.position_numbers.index(self.event_position_var)
                        else:
                            self.event_position_index = 0
                else:
                    self.event_group_index = 0
            else:
                self.event_name_index = 0
                self.abet_group_numbers = self.abet_pandas.loc[:, 'Group_ID']
                self.abet_group_numbers = self.abet_group_numbers.unique()
                self.abet_group_numbers = list(self.abet_group_numbers)
                self.abet_group_numbers = sorted(self.abet_group_numbers)
        else:
            self.event_id_index = 0
            self.abet_group_name = self.abet_pandas.loc[:, 'Item_Name']
            self.abet_group_name = self.abet_group_name.unique()
            self.abet_group_name = list(self.abet_group_name)
            self.abet_group_name = sorted(self.abet_group_name)

        if self.event_id_var in self.touch_event_names:
            self.position_state = 'normal'
        else:
            self.position_state = 'disabled'

    def doric_setting_load(self):
        if self.channel_control_var in self.doric_name_list:
            self.channel_control_index = self.doric_name_list.index(self.channel_control_var)
        else:
            self.channel_control_index = 0

        if self.channel_active_var in self.doric_name_list:
            self.channel_active_index = self.doric_name_list.index(self.channel_active_var)
        else:
            self.channel_active_index = 0

        if self.channel_ttl_var in self.doric_name_list:
            self.channel_ttl_index = self.doric_name_list.index(self.channel_ttl_var)
        else:
            self.channel_ttl_index = 0

    def anymaze_setting_load(self):
        if self.anymaze_event1_column_var == 'None':
            self.anymaze_event1_column_index = self.anymaze_column_names.index('None')
            self.anymaze_event1_operation_index = 0
            self.anymaze_event1_value_var = 'None'
            self.anymaze_event1_operation_state = 'disabled'
            self.anymaze_event1_value_state = 'disabled'
        else:
            try:
                self.anymaze_event1_column_index = self.anymaze_column_names.index(self.anymaze_event1_column_var)
                anymaze_event1_options = self.anymaze_pandas.loc[:, self.anymaze_event1_column_var]
                anymaze_event1_options = anymaze_event1_options.unique()
                anymaze_event1_options = list(anymaze_event1_options)
                anymaze_event1_count = len(anymaze_event1_options)
                if anymaze_event1_count == 2:
                    self.anymaze_event1_operation_list = self.anymaze_boolean_list
                    self.anymaze_event1_value_state = 'disabled'
                elif anymaze_event1_count > 2:
                    self.anymaze_event1_operation_list = self.anymaze_operation_list
                    self.anymaze_event1_value_state = 'normal'
                elif anymaze_event1_count == 1:
                    self.anymaze_event1_value_state = 'disabled'
                    self.anymaze_event1_operation_state = 'disabled'
            except ValueError:
                self.anymaze_event1_column_index = self.anymaze_column_names.index('None')
                self.anymaze_event1_operation_state = 'disabled'
                self.anymaze_event1_value_state = 'disabled'
                self.anymaze_event1_operation_index = 0
                self.anymaze_event1_value_var = 'None'

        if self.anymaze_event2_column_var == 'None':
            self.anymaze_event2_column_index = self.anymaze_column_names.index('None')
            self.anymaze_event2_operation_index = 'None'
            self.anymaze_event2_value_var = 'None'
            self.anymaze_event2_operation_state = 'disabled'
            self.anymaze_event2_value_state = 'disabled'
        else:
            try:
                self.anymaze_event2_column_index = self.anymaze_column_names.index(self.anymaze_event2_column_var)
                anymaze_event2_options = self.anymaze_pandas.loc[:, self.anymaze_event2_column_var]
                anymaze_event2_options = anymaze_event2_options.unique()
                anymaze_event2_options = list(anymaze_event2_options)
                anymaze_event2_count = len(anymaze_event2_options)
                if anymaze_event2_count == 2:
                    self.anymaze_event2_operation_list = self.anymaze_boolean_list
                    self.anymaze_event2_value_state = 'disabled'
                elif anymaze_event2_count > 2:
                    self.anymaze_event2_operation_list = self.anymaze_operation_list
                    self.anymaze_event2_value_state = 'normal'
                elif anymaze_event2_count == 1:
                    self.anymaze_event2_value_state = 'disabled'
                    self.anymaze_event2_operation_state = 'disabled'
            except ValueError:
                self.anymaze_event2_column_index = self.anymaze_column_names.index('None')
                self.anymaze_event2_operation_state = 'disabled'
                self.anymaze_event2_value_state = 'disabled'
                self.anymaze_event2_operation_index = 0
                self.anymaze_event2_value_var = 'None'

        if self.anymaze_event3_column_var == 'None':
            self.anymaze_event3_column_index = self.anymaze_column_names.index('None')
            self.anymaze_event3_operation_index = 0
            self.anymaze_event3_value_var = 'None'
            self.anymaze_event3_operation_state = 'disabled'
            self.anymaze_event3_value_state = 'disabled'
        else:
            try:
                self.anymaze_event3_column_index = self.anymaze_column_names.index(self.anymaze_event3_column_var)
                anymaze_event3_options = self.anymaze_pandas.loc[:, self.anymaze_event3_column_var]
                anymaze_event3_options = anymaze_event3_options.unique()
                anymaze_event3_options = list(anymaze_event3_options)
                anymaze_event3_count = len(anymaze_event3_options)
                if anymaze_event3_count == 2:
                    self.anymaze_event3_operation_list = self.anymaze_boolean_list
                    self.anymaze_event3_value_state = 'disabled'
                elif anymaze_event3_count > 2:
                    self.anymaze_event3_operation_list = self.anymaze_operation_list
                    self.anymaze_event3_value_state = 'normal'
                elif anymaze_event3_count == 1:
                    self.anymaze_event3_value_state = 'disabled'
                    self.anymaze_event3_operation_state = 'disabled'
            except ValueError:
                self.anymaze_event3_column_index = self.anymaze_column_names.index('None')
                self.anymaze_event3_operation_state = 'disabled'
                self.anymaze_event3_value_state = 'disabled'
                self.anymaze_event3_operation_index = 0
                self.anymaze_event3_value_var = 0

    def doric_file_load(self, path=''):
        if path == '':
            self.doric_file_path = filedialog.askopenfilename(title='Select Doric File',
                                                              filetypes=(('csv files', '*.csv'), ('all files', '*.')))
        else:
            self.doric_file_path = path
            if not os.path.isfile(path):
                self.doric_file_path = ''
                self.doric_field.delete(0, END)
                self.doric_field.insert(END, str(self.doric_file_path))
                return
        self.doric_field.delete(0, END)
        self.doric_field.insert(END, str(self.doric_file_path))

        try:
            doric_file = open(self.doric_file_path)
            doric_csv_reader = csv.reader(doric_file)
            first_row_read = False
            second_row_read = False
            self.doric_name_list = list()
            for row in doric_csv_reader:
                if not first_row_read:
                    first_row_read = True
                    continue
                if second_row_read is False and first_row_read is True:
                    self.doric_name_list = row
                    break
            doric_file.close()
            if (len(self.doric_name_list) - 1) < self.channel_control_index:
                self.channel_control_index = 0
            if self.channel_control_var in self.doric_name_list:
                self.channel_control_index = self.doric_name_list.index(self.channel_control_var)
            else:
                self.channel_control_index = 0

            if (len(self.doric_name_list) - 1) < self.channel_active_index:
                self.channel_active_index = 0
            if self.channel_active_var in self.doric_name_list:
                self.channel_active_index = self.doric_name_list.index(self.channel_active_var)
            else:
                self.channel_active_index = 0

            if (len(self.doric_name_list) - 1) < self.channel_ttl_index:
                self.channel_ttl_index = 0
            if self.channel_ttl_var in self.doric_name_list:
                self.channel_ttl_index = self.doric_name_list.index(self.channel_ttl_var)
            else:
                self.channel_ttl_index = 0
        except:
            self.doric_name_list = ['']
            self.channel_control_var = 0
            self.channel_active_var = 0
            self.channel_ttl_var = 0

        if path != '':
            self.doric_setting_load()

    def abet_file_load(self, path=''):
        if path == '':
            self.abet_file_path = filedialog.askopenfilename(title='Select ABETII File',
                                                             filetypes=(('csv files', '*.csv'), ('all files', '*.')))
        else:
            self.abet_file_path = path
            if not os.path.isfile(path):
                self.abet_file_path = ''
                self.abet_field.delete(0, END)
                self.abet_field.insert(END, str(self.abet_file_path))
                return
        self.abet_field.delete(0, END)
        self.abet_field.insert(END, str(self.abet_file_path))
        try:
            abet_file = open(self.abet_file_path)
            abet_csv_reader = csv.reader(abet_file)
            abet_data_list = list()
            abet_name_list = list()
            colnames_found = False
            for row in abet_csv_reader:
                if not colnames_found:
                    if len(row) == 0:
                        continue
                    if row[0] in self.event_time_colname:
                        colnames_found = True
                        abet_name_list = row
                        self.abet_event_name_col = row[2]
                    else:
                        continue
                else:
                    abet_data_list.append(row)
            abet_file.close()
            abet_numpy = np.array(abet_data_list)
            self.abet_pandas = pd.DataFrame(data=abet_numpy, columns=abet_name_list)
            self.abet_event_types = self.abet_pandas.loc[:, self.abet_event_name_col]
            self.abet_event_types = self.abet_event_types.unique()
            self.abet_event_types = list(self.abet_event_types)
            self.abet_event_types = sorted(self.abet_event_types)
            if (len(self.abet_event_types) - 1) < self.event_id_index:
                self.event_id_index = 0

            if self.event_id_var in self.abet_event_types:
                self.event_id_index = self.abet_event_types.index(self.event_id_var)
            else:
                self.event_id_index = 0

            self.abet_group_numbers = self.abet_pandas.loc[:, 'Group_ID']
            self.abet_group_numbers = self.abet_group_numbers.unique()
            self.abet_group_numbers = list(self.abet_group_numbers)
            self.abet_group_numbers = sorted(self.abet_group_numbers)
            if (len(self.abet_group_numbers) - 1) < self.event_group_index:
                self.event_group_index = 0

            if self.event_group_var in self.abet_group_numbers:
                self.event_group_index = self.abet_group_numbers.index(self.event_group_var)
            else:
                self.event_group_index = 0

            self.abet_group_name = self.abet_pandas.loc[:, 'Item_Name']
            self.abet_group_name = self.abet_group_name.unique()
            self.abet_group_name = list(self.abet_group_name)
            self.abet_group_name = sorted(self.abet_group_name)
            if (len(self.abet_group_name) - 1) < self.event_name_index:
                self.event_name_index = 0

            if self.event_name_var in self.abet_group_name:
                self.event_name_index = self.abet_group_name.index(self.event_name_var)
            else:
                self.event_name_index = 0

            self.abet_iti_group_name = self.abet_pandas.loc[
                (self.abet_pandas[self.abet_event_name_col] == 'Condition Event'), 'Item_Name']
            self.abet_iti_group_name = self.abet_iti_group_name.unique()
            self.abet_iti_group_name = list(self.abet_iti_group_name)
            self.abet_iti_group_name = sorted(self.abet_iti_group_name)
            if (len(self.abet_iti_group_name) - 1) < self.abet_trial_start_index:
                self.abet_trial_start_index = 0

            if str(self.abet_trial_start_var) in self.abet_iti_group_name:
                self.abet_trial_start_index = self.abet_iti_group_name.index(self.abet_trial_start_var)
            else:
                self.abet_trial_start_index = 0

            if (len(self.abet_iti_group_name) - 1) < self.abet_trial_end_index:
                self.abet_trial_end_index = 0

            if str(self.abet_trial_end_var) in self.abet_iti_group_name:
                self.abet_trial_end_index = self.abet_iti_group_name.index(self.abet_trial_end_var)
            else:
                self.abet_trial_end_index = 0
        except:
            self.abet_event_types = ['']
            self.abet_group_name = ['']
            self.abet_group_numbers = ['']
            self.abet_trial_stages = ['']
            self.abet_iti_group_name = ['']

        if path != '':
            self.abet_setting_load()

    def anymaze_file_load(self, path=''):
        if path == '':
            self.anymaze_file_path = filedialog.askopenfilename(title='Select Anymaze File',
                                                                filetypes=(('csv files', '*.csv'), ('all files', '*.')))
        else:
            self.anymaze_file_path = path
            if not os.path.isfile(path):
                self.anymaze_file_path = ''
                self.anymaze_field.delete(0, END)
                self.anymaze_field.insert(END, str(self.anymaze_file_path))
                return
        self.anymaze_field.delete(0, END)
        self.anymaze_field.insert(END, str(self.anymaze_file_path))

        try:
            anymaze_file = open(self.anymaze_file_path)
            anymaze_csv = csv.reader(anymaze_file)
            colname_found = False
            anymaze_data = list()
            for row in anymaze_csv:
                if not colname_found:
                    self.anymaze_column_names = row
                    colname_found = True
                else:
                    anymaze_data.append(row)
            anymaze_file.close()
            self.anymaze_column_names.append('None')
            anymaze_numpy = np.array(anymaze_data)
            self.anymaze_pandas = pd.DataFrame(data=anymaze_numpy, columns=self.anymaze_column_names[
                                                                           0:(len(self.anymaze_column_names) - 1)])
        except:
            self.anymaze_column_names = ['None']

        if path != '':
            self.anymaze_setting_load()

    def abet_event_name_check(self, event):
        self.abet_group_name = self.abet_pandas.loc[
            self.abet_pandas[self.abet_event_name_col] == str(self.event_id_type_entry.get()), 'Item_Name']
        self.abet_group_name = self.abet_group_name.unique()
        self.abet_group_name = list(self.abet_group_name)
        self.abet_group_name = sorted(self.abet_group_name)
        self.event_name_entry['values'] = self.abet_group_name
        if str(self.event_id_type_entry.get()) in self.touch_event_names:
            self.event_position_entry.config(state='normal')
        else:
            self.event_position_entry.config(state='disabled')
            self.event_position_index = 0
        try:
            self.abet_group_numbers = self.abet_pandas.loc[
                self.abet_pandas[self.abet_event_name_col] == str(self.event_id_type_entry.get()), 'Group_ID']
            self.abet_group_numbers = self.abet_group_numbers.unique()
            self.abet_group_numbers = list(self.abet_group_numbers)
            self.abet_group_numbers = sorted(self.abet_group_numbers)
            self.event_group_entry.config(values=self.abet_group_numbers)
            if self.abet_group_numbers == 1:
                self.event_group_entry.current(0)
                self.event_group_entry.set(self.abet_group_numbers[0])
        except:
            self.abet_group_numbers = self.abet_pandas.loc[:, 'Group_ID']
            self.abet_group_numbers = self.abet_group_numbers.unique()
            self.abet_group_numbers = list(self.abet_group_numbers)
            self.abet_group_numbers = sorted(self.abet_group_numbers)
            self.event_group_entry['values'] = self.abet_group_numbers
            if self.abet_group_numbers == 1:
                self.event_group_entry.current(0)
                self.event_group_entry.set(self.abet_group_numbers[0])

        self.abet_event_gui.update()

    def abet_item_name_check(self, event):
        self.abet_group_numbers = self.abet_pandas.loc[
            (self.abet_pandas[self.abet_event_name_col] == str(self.event_id_type_entry.get())) &
            (self.abet_pandas['Item_Name'] == str(self.event_name_entry.get())), 'Group_ID']
        self.abet_group_numbers = self.abet_group_numbers.unique()
        self.abet_group_numbers = list(self.abet_group_numbers)
        self.abet_group_numbers = sorted(self.abet_group_numbers)
        self.event_group_entry['values'] = self.abet_group_numbers
        if self.abet_group_numbers == 1:
            self.event_group_entry.current(0)
            self.event_group_entry.set(self.abet_group_numbers[0])
        self.abet_event_gui.update()

    def abet_group_number_check(self, event):
        if str(self.event_id_type_entry.get()) in self.touch_event_names:
            self.position_numbers = self.abet_pandas.loc[
                (self.abet_pandas[self.abet_event_name_col] == str(self.event_id_type_entry.get())) &
                (self.abet_pandas['Item_Name'] == str(self.event_name_entry.get())) &
                (self.abet_pandas['Group_ID'] == str(self.event_group_entry.get())), 'Arg1_Value']
            self.position_numbers = self.position_numbers.unique()
            self.position_numbers = list(self.position_numbers)
            self.position_numbers = sorted(self.position_numbers)
            self.event_position_entry['values'] = self.position_numbers
        else:
            self.event_position_index = 0
            return
        self.abet_event_gui.update()

    def abet_event_definition_gui(self):

        self.iti_zscoring = tk.IntVar()
        self.iti_zscoring.set(self.iti_normalize)

        self.abet_event_gui = tk.Toplevel()

        self.abet_event_title = tk.Label(self.abet_event_gui, text='ABET Event Definition')
        self.abet_event_title.grid(row=0, column=1)

        self.event_id_type_label = tk.Label(self.abet_event_gui, text='Event Type')
        self.event_id_type_label.grid(row=1, column=0)
        # self.event_id_type_entry = tk.Entry(self.abet_event_gui)
        self.event_id_type_entry = ttk.Combobox(self.abet_event_gui, values=self.abet_event_types)
        self.event_id_type_entry.grid(row=2, column=0)
        self.event_id_type_entry.bind("<<ComboboxSelected>>", self.abet_event_name_check)
        self.event_id_type_entry.current(self.event_id_index)
        # self.event_id_type_entry.insert(END,self.event_id_var)

        self.event_name_label = tk.Label(self.abet_event_gui, text='Name of Event')
        self.event_name_label.grid(row=1, column=1)
        # self.event_name_entry = tk.Entry(self.abet_event_gui)
        self.event_name_entry = ttk.Combobox(self.abet_event_gui, values=self.abet_group_name)
        self.event_name_entry.grid(row=2, column=1)
        self.event_name_entry.bind("<<ComboboxSelected>>", self.abet_item_name_check)
        self.event_name_entry.current(self.event_name_index)
        # self.event_name_entry.insert(END,self.event_name_var)

        self.event_group_label = tk.Label(self.abet_event_gui, text='Event Group #')
        self.event_group_label.grid(row=1, column=2)
        # self.event_group_entry = tk.Entry(self.abet_event_gui)
        self.event_group_entry = ttk.Combobox(self.abet_event_gui, values=self.abet_group_numbers)
        self.event_group_entry.grid(row=2, column=2)
        self.event_group_entry.current(self.event_group_index)
        self.event_group_entry.bind("<<ComboboxSelected>>", self.abet_group_number_check)
        # self.event_group_entry.insert(END,self.event_group_var)

        self.event_position_label = tk.Label(self.abet_event_gui, text='Event Position #')
        self.event_position_label.grid(row=3, column=0)
        # self.event_position_entry = tk.Entry(self.abet_event_gui)
        self.event_position_entry = ttk.Combobox(self.abet_event_gui, values=self.abet_group_numbers)
        self.event_position_entry.grid(row=4, column=0)
        self.event_position_entry.current(self.event_position_index)
        if str(self.event_id_type_entry.get()) in self.touch_event_names:
            self.event_position_entry.config(state='normal')
        else:
            self.event_position_entry.config(state='disabled')
        # self.event_position_entry.insert(END,self.event_position_var)

        self.event_prior_time = tk.Label(self.abet_event_gui, text='Time Prior to Event (sec)')
        self.event_prior_time.grid(row=3, column=1)
        self.event_prior_entry = tk.Entry(self.abet_event_gui)
        self.event_prior_entry.grid(row=4, column=1)
        self.event_prior_entry.insert(END, self.event_prior_var)

        self.event_follow_time = tk.Label(self.abet_event_gui, text='Time Following Event (sec)')
        self.event_follow_time.grid(row=3, column=2)
        self.event_follow_entry = tk.Entry(self.abet_event_gui)
        self.event_follow_entry.grid(row=4, column=2)
        self.event_follow_entry.insert(END, self.event_follow_var)

        self.abet_trial_definition_title = tk.Label(self.abet_event_gui, text='ABET Trial Definition')
        self.abet_trial_definition_title.grid(row=5, column=1)

        self.abet_trial_start_group_label = tk.Label(self.abet_event_gui, text='Start Event Group #')
        self.abet_trial_start_group_label.grid(row=6, column=0)
        # self.abet_trial_start_entry = tk.Entry(self.abet_event_gui)
        self.abet_trial_start_entry = ttk.Combobox(self.abet_event_gui, values=self.abet_iti_group_name)
        self.abet_trial_start_entry.grid(row=7, column=0)
        self.abet_trial_start_entry.current(self.abet_trial_start_index)
        # self.abet_trial_start_entry.insert(END,self.abet_trial_start_var)

        self.abet_trial_end_group_label = tk.Label(self.abet_event_gui, text='End Event Group #')
        self.abet_trial_end_group_label.grid(row=6, column=1)
        # self.abet_trial_end_entry = tk.Entry(self.abet_event_gui)
        self.abet_trial_end_entry = ttk.Combobox(self.abet_event_gui, values=self.abet_iti_group_name)
        self.abet_trial_end_entry.grid(row=7, column=1)
        self.abet_trial_end_entry.current(self.abet_trial_end_index)
        # self.abet_trial_end_entry.insert(END,self.abet_trial_end_var)

        self.abet_trial_iti_prior_label = tk.Label(self.abet_event_gui, text='ITI Length Prior to Start')
        self.abet_trial_iti_prior_label.grid(row=6, column=2)
        self.abet_trial_iti_entry = tk.Entry(self.abet_event_gui)
        self.abet_trial_iti_entry.grid(row=7, column=2)
        self.abet_trial_iti_entry.insert(END, self.abet_trial_iti_var)

        self.abet_iti_zscore_checkbutton = tk.Checkbutton(self.abet_event_gui, text='Z-Score Based on ITI',
                                                          variable=self.iti_zscoring)

        self.abet_event_finish_button = tk.Button(self.abet_event_gui, text='Finish', command=self.abet_event_commit)
        self.abet_event_finish_button.grid(row=8, column=1)

    def abet_event_commit(self):
        self.event_id_var = str(self.event_id_type_entry.get())
        self.event_id_index = int(self.event_id_type_entry.current())
        self.event_group_var = str(self.event_group_entry.get())
        self.event_group_index = int(self.event_group_entry.current())
        if self.event_group_index < 0:
            self.event_group_index = 0
        self.event_position_var = str(self.event_position_entry.get())
        self.event_position_index = int(self.event_position_entry.current())
        self.event_name_var = str(self.event_name_entry.get())
        self.event_name_index = int(self.event_name_entry.current())
        self.event_prior_var = str(self.event_prior_entry.get())
        self.event_follow_var = str(self.event_follow_entry.get())
        self.abet_trial_start_var = str(self.abet_trial_start_entry.get())
        self.abet_trial_start_index = int(self.abet_trial_start_entry.current())
        self.abet_trial_end_var = str(self.abet_trial_end_entry.get())
        self.abet_trial_end_index = int(self.abet_trial_end_entry.current())
        self.abet_trial_iti_var = str(self.abet_trial_iti_entry.get())
        self.iti_normalize = self.iti_zscoring.get()

        self.abet_event_gui.destroy()

    def anymaze_event_description_gui(self):
        self.anymaze_event_gui = tk.Toplevel()

        self.anymaze_event_title = tk.Label(self.anymaze_event_gui, text='Anymaze Event Definition')
        self.anymaze_event_title.grid(row=0, column=1)

        self.anymaze_event_type_colname = tk.Label(self.anymaze_event_gui, text='Column')
        self.anymaze_event_type_colname.grid(row=1, column=0)

        self.anymaze_event_operation_colname = tk.Label(self.anymaze_event_gui, text='Function')
        self.anymaze_event_operation_colname.grid(row=1, column=1)

        self.anymaze_event_value_colname = tk.Label(self.anymaze_event_gui, text='Value')
        self.anymaze_event_value_colname.grid(row=1, column=2)

        self.anymaze_event1_colname = ttk.Combobox(self.anymaze_event_gui, values=self.anymaze_column_names)
        self.anymaze_event1_colname.grid(row=2, column=0)
        self.anymaze_event1_colname.bind("<<ComboboxSelected>>", self.anymaze_column_set_event1)
        try:
            self.anymaze_event1_column_index = self.anymaze_column_names.index(self.anymaze_event1_column_var)
            self.anymaze_event1_colname.current(self.anymaze_event1_column_index)
        except ValueError:
            self.anymaze_event1_colname.current(0)
        self.anymaze_event1_operation = ttk.Combobox(self.anymaze_event_gui, values=self.anymaze_event1_operation_list)
        self.anymaze_event1_operation.grid(row=2, column=1)
        self.anymaze_event1_operation.config(state=self.anymaze_event1_operation_state)
        try:
            self.anymaze_event1_operation_index = self.anymaze_event1_operation_list.index(
                self.anymaze_event1_operation_var)
            self.anymaze_event1_operation.current(self.anymaze_event1_operation_index)
        except ValueError:
            self.anymaze_event1_operation.current(0)
        self.anymaze_event1_value = tk.Entry(self.anymaze_event_gui)
        self.anymaze_event1_value.grid(row=2, column=2)
        self.anymaze_event1_value.config(state=self.anymaze_event1_value_state)
        if self.anymaze_event1_value_state != "disabled":
            self.anymaze_event1_value.insert(END, self.anymaze_event1_value_var)
        if self.anymaze_event1_column_var != 'None':
            self.anymaze_column_set_event1(event=self.anymaze_event1_column_var)
        self.anymaze_event1_operation.current(self.anymaze_event1_operation_index)

        self.anymaze_event2_colname = ttk.Combobox(self.anymaze_event_gui, values=self.anymaze_column_names)
        self.anymaze_event2_colname.grid(row=3, column=0)
        self.anymaze_event2_colname.bind("<<ComboboxSelected>>", self.anymaze_column_set_event2)
        try:
            self.anymaze_event2_column_index = self.anymaze_column_names.index(self.anymaze_event2_column_var)
            self.anymaze_event2_colname.current(self.anymaze_event2_column_index)
        except ValueError:
            self.anymaze_event2_colname.current(0)
        self.anymaze_event2_operation = ttk.Combobox(self.anymaze_event_gui, values=self.anymaze_event2_operation_list)
        self.anymaze_event2_operation.grid(row=3, column=1)
        self.anymaze_event2_operation.config(state=self.anymaze_event2_operation_state)
        try:
            self.anymaze_event2_operation_index = self.anymaze_event2_operation_list.index(
                self.anymaze_event2_operation_var)
            self.anymaze_event2_operation.current(self.anymaze_event2_operation_index)
        except ValueError:
            self.anymaze_event2_operation.current(0)
        self.anymaze_event2_value = tk.Entry(self.anymaze_event_gui)
        self.anymaze_event2_value.grid(row=3, column=2)
        self.anymaze_event2_value.config(state=self.anymaze_event2_value_state)
        if self.anymaze_event2_value_state != "disabled":
            self.anymaze_event2_value.insert(END, self.anymaze_event2_value_var)
        if self.anymaze_event2_column_var != 'None':
            self.anymaze_column_set_event2(event=self.anymaze_event2_column_var)
        self.anymaze_event2_operation.current(self.anymaze_event2_operation_index)

        self.anymaze_event3_colname = ttk.Combobox(self.anymaze_event_gui, values=self.anymaze_column_names)
        self.anymaze_event3_colname.grid(row=4, column=0)
        self.anymaze_event3_colname.bind("<<ComboboxSelected>>", self.anymaze_column_set_event3)
        try:
            self.anymaze_event3_column_index = self.anymaze_column_names.index(self.anymaze_event3_column_var)
            self.anymaze_event3_colname.current(self.anymaze_event3_column_index)
        except ValueError:
            self.anymaze_event3_colname.current(0)
        self.anymaze_event3_operation = ttk.Combobox(self.anymaze_event_gui, values=self.anymaze_event3_operation_list)
        self.anymaze_event3_operation.grid(row=4, column=1)
        self.anymaze_event3_operation.config(state=self.anymaze_event3_operation_state)
        try:
            self.anymaze_event3_operation_index = self.anymaze_event3_operation_list.index(
                self.anymaze_event3_operation_var)
            self.anymaze_event3_operation.current(self.anymaze_event3_operation_index)
        except ValueError:
            self.anymaze_event1_operation.current(0)
        self.anymaze_event3_value = tk.Entry(self.anymaze_event_gui)
        self.anymaze_event3_value.grid(row=4, column=2)
        self.anymaze_event3_value.config(state=self.anymaze_event3_value_state)
        if self.anymaze_event3_value_state != "disabled":
            self.anymaze_event3_value.insert(END, self.anymaze_event3_value_var)
        if self.anymaze_event3_column_var != 'None':
            self.anymaze_column_set_event3(event=self.anymaze_event3_column_var)
        self.anymaze_event3_operation.current(self.anymaze_event3_operation_index)

        self.anymaze_settings_label = tk.Label(self.anymaze_event_gui, text='Anymaze Settings')
        self.anymaze_settings_label.grid(row=5, column=1)

        self.anymaze_tolerance_label = tk.Label(self.anymaze_event_gui, text='Event Length Tolerance')
        self.anymaze_tolerance_label.grid(row=6, column=0)
        self.anymaze_tolerance_entry = tk.Entry(self.anymaze_event_gui)
        self.anymaze_tolerance_entry.grid(row=7, column=0)
        self.anymaze_tolerance_entry.insert(END, self.anymaze_tolerance_var)

        self.anymaze_extra_prior_label = tk.Label(self.anymaze_event_gui, text='Time Prior to Event (sec)')
        self.anymaze_extra_prior_label.grid(row=6, column=1)
        self.anymaze_extra_prior_entry = tk.Entry(self.anymaze_event_gui)
        self.anymaze_extra_prior_entry.grid(row=7, column=1)
        self.anymaze_extra_prior_entry.insert(END, self.anymaze_extra_prior_var)

        self.anymaze_extra_follow_label = tk.Label(self.anymaze_event_gui, text='Time Following Event (sec)')
        self.anymaze_extra_follow_label.grid(row=6, column=2)
        self.anymaze_extra_follow_entry = tk.Entry(self.anymaze_event_gui)
        self.anymaze_extra_follow_entry.grid(row=7, column=2)
        self.anymaze_extra_follow_entry.insert(END, self.anymaze_extra_follow_var)

        self.anymaze_event_location_label = tk.Label(self.anymaze_event_gui, text='Centering Location')
        self.anymaze_event_location_label.grid(row=8, column=0)
        self.anymaze_event_location_list = ['Event Start', 'Event Center', 'Event End']
        self.anymaze_event_location_entry = ttk.Combobox(self.anymaze_event_gui,
                                                         values=self.anymaze_event_location_list)
        self.anymaze_event_location_entry.grid(row=9, column=0)
        self.anymaze_event_location_entry.current(self.anymaze_event_location_index)

        self.anymaze_event_pad_label = tk.Label(self.anymaze_event_gui, text='Centering Prior')
        self.anymaze_event_pad_label.grid(row=8, column=1)
        self.anymaze_event_pad_entry = tk.Entry(self.anymaze_event_gui)
        self.anymaze_event_pad_entry.grid(row=9, column=1)
        self.anymaze_event_pad_entry.insert(END, self.anymaze_event_pad_var)

        self.anymaze_finish_button = tk.Button(self.anymaze_event_gui, text='Finish', command=self.anymaze_event_commit)
        self.anymaze_finish_button.grid(row=10, column=1)

    def anymaze_column_set_event1(self, event):
        self.anymaze_event1_column_var = self.anymaze_event1_colname.get()
        if self.anymaze_event1_column_var == 'None':
            self.anymaze_event1_operation_state = 'disabled'
            self.anymaze_event1_value_state = 'disabled'
            self.anymaze_event1_operation.config(state=self.anymaze_event1_operation_state)
            self.anymaze_event1_value.config(state=self.anymaze_event1_value_state)
        else:
            self.anymaze_event1_operation_state = 'normal'
            self.anymaze_event1_operation.config(state=self.anymaze_event1_operation_state)
            anymaze_event1_options = self.anymaze_pandas.loc[:, self.anymaze_event1_column_var]
            anymaze_event1_options = anymaze_event1_options.unique()
            anymaze_event1_options = list(anymaze_event1_options)
            anymaze_event1_count = len(anymaze_event1_options)
            if anymaze_event1_count == 2:
                self.anymaze_event1_operation_list = self.anymaze_boolean_list
                self.anymaze_event1_operation['values'] = self.anymaze_event1_operation_list
                self.anymaze_event1_value_state = 'disabled'
                self.anymaze_event1_boolean = True
            elif anymaze_event1_count > 2:
                self.anymaze_event1_operation_list = self.anymaze_operation_list
                self.anymaze_event1_operation['values'] = self.anymaze_event1_operation_list
                self.anymaze_event1_value_state = 'normal'
                self.anymaze_event1_boolean = False
            elif anymaze_event1_count == 1:
                self.anymaze_event1_value_state = 'disabled'
                self.anymaze_event1_operation_state = 'disabled'
                self.anymaze_event1_operation.config(state=self.anymaze_event1_operation_state)
            self.anymaze_event1_value.config(state=self.anymaze_event1_value_state)

    def anymaze_column_set_event2(self, event):
        self.anymaze_event2_column_var = self.anymaze_event2_colname.get()
        if self.anymaze_event2_column_var == 'None':
            self.anymaze_event2_operation_state = 'disabled'
            self.anymaze_event2_value_state = 'disabled'
            self.anymaze_event2_operation.config(state=self.anymaze_event2_operation_state)
            self.anymaze_event2_value.config(state=self.anymaze_event2_value_state)
        else:
            self.anymaze_event2_operation_state = 'normal'
            self.anymaze_event2_operation.config(state=self.anymaze_event2_operation_state)
            anymaze_event2_options = self.anymaze_pandas.loc[:, self.anymaze_event2_column_var]
            anymaze_event2_options = anymaze_event2_options.unique()
            anymaze_event2_options = list(anymaze_event2_options)
            anymaze_event2_count = len(anymaze_event2_options)
            if anymaze_event2_count == 2:
                self.anymaze_event2_operation_list = self.anymaze_boolean_list
                self.anymaze_event2_operation['values'] = self.anymaze_event2_operation_list
                self.anymaze_event2_value_state = 'disabled'
                self.anymaze_event2_boolean = True
            elif anymaze_event2_count > 2:
                self.anymaze_event2_operation_list = self.anymaze_operation_list
                self.anymaze_event2_operation['values'] = self.anymaze_event2_operation_list
                self.anymaze_event2_value_state = 'normal'
                self.anymaze_event2_boolean = False
            elif anymaze_event2_count == 1:
                self.anymaze_event2_value_state = 'disabled'
                self.anymaze_event2_operation_state = 'disabled'
                self.anymaze_event2_operation.config(state=self.anymaze_event2_operation_state)
            self.anymaze_event2_value.config(state=self.anymaze_event2_value_state)

    def anymaze_column_set_event3(self, event):
        self.anymaze_event3_column_var = self.anymaze_event3_colname.get()
        if self.anymaze_event3_column_var == 'None':
            self.anymaze_event3_operation_state = 'disabled'
            self.anymaze_event3_value_state = 'disabled'
            self.anymaze_event3_operation.config(state=self.anymaze_event3_operation_state)
            self.anymaze_event3_value.config(state=self.anymaze_event3_value_state)
        else:
            self.anymaze_event3_operation_state = 'normal'
            self.anymaze_event3_operation.config(state=self.anymaze_event3_operation_state)
            anymaze_event3_options = self.anymaze_pandas.loc[:, self.anymaze_event3_column_var]
            anymaze_event3_options = anymaze_event3_options.unique()
            anymaze_event3_options = list(anymaze_event3_options)
            anymaze_event3_count = len(anymaze_event3_options)
            if anymaze_event3_count == 2:
                self.anymaze_event3_operation_list = self.anymaze_boolean_list
                self.anymaze_event3_operation['values'] = self.anymaze_event3_operation_list
                self.anymaze_event3_value_state = 'disabled'
                self.anymaze_event3_boolean = True
            elif anymaze_event3_count > 2:
                self.anymaze_event3_operation_list = self.anymaze_operation_list
                self.anymaze_event3_operation['values'] = self.anymaze_event3_operation_list
                self.anymaze_event3_value_state = 'normal'
                self.anymaze_event3_boolean = False
            elif anymaze_event3_count == 1:
                self.anymaze_event3_value_state = 'disabled'
                self.anymaze_event3_operation_state = 'disabled'
                self.anymaze_event3_operation.config(state=self.anymaze_event3_operation_state)
            self.anymaze_event3_value.config(state=self.anymaze_event3_value_state)

    def anymaze_event_commit(self):
        self.anymaze_event1_column_var = self.anymaze_event1_colname.get()
        self.anymaze_event1_column_index = self.anymaze_column_names.index(self.anymaze_event1_column_var)
        self.anymaze_event2_column_var = self.anymaze_event2_colname.get()
        self.anymaze_event2_column_index = self.anymaze_column_names.index(self.anymaze_event2_column_var)
        self.anymaze_event3_column_var = self.anymaze_event3_colname.get()
        self.anymaze_event3_column_index = self.anymaze_column_names.index(self.anymaze_event3_column_var)
        self.anymaze_event_pad_var = self.anymaze_event_pad_entry.get()
        self.anymaze_tolerance_var = self.anymaze_tolerance_entry.get()
        self.anymaze_extra_prior_var = self.anymaze_extra_prior_entry.get()
        self.anymaze_extra_follow_var = self.anymaze_extra_follow_entry.get()
        if self.anymaze_event1_column_var == 'None':
            self.anymaze_event1_operation_var = 'None'
            self.anymaze_event1_operation_index = 0
            self.anymaze_event1_value_var = 'None'
        else:
            self.anymaze_event1_operation_var = self.anymaze_event1_operation.get()
            if self.anymaze_event1_boolean:
                self.anymaze_event1_operation_index = self.anymaze_boolean_list.index(self.anymaze_event1_operation_var)
            else:
                self.anymaze_event1_operation_index = self.anymaze_operation_list.index(
                    self.anymaze_event1_operation_var)
            self.anymaze_event1_value_var = self.anymaze_event1_value.get()

        if self.anymaze_event2_column_var == 'None':
            self.anymaze_event2_operation_var = 'None'
            self.anymaze_event2_operation_index = 0
            self.anymaze_event2_value_var = 'None'
        else:
            self.anymaze_event2_operation_var = self.anymaze_event2_operation.get()
            if self.anymaze_event2_boolean:
                self.anymaze_event2_operation_index = self.anymaze_boolean_list.index(self.anymaze_event2_operation_var)
            else:
                self.anymaze_event2_operation_index = self.anymaze_operation_list.index(
                    self.anymaze_event2_operation_var)
            self.anymaze_event2_value_var = self.anymaze_event2_value.get()

        if self.anymaze_event3_column_var == 'None':
            self.anymaze_event3_operation_var = 'None'
            self.anymaze_event3_operation_index = 0
            self.anymaze_event3_value_var = 'None'
        else:
            self.anymaze_event3_operation_var = self.anymaze_event3_operation.get()
            if self.anymaze_event3_boolean:
                self.anymaze_event3_operation_index = self.anymaze_boolean_list.index(self.anymaze_event3_operation_var)
            else:
                self.anymaze_event3_operation_index = self.anymaze_operation_list.index(
                    self.anymaze_event3_operation_var)
            self.anymaze_event3_value_var = self.anymaze_event3_value.get()

        self.anymaze_event_gui.destroy()

    def settings_menu(self):
        self.settings_gui = tk.Toplevel()

        self.settings_title = tk.Label(self.settings_gui, text='Settings')
        self.settings_title.grid(row=0, column=1)

        self.channel_control_label = tk.Label(self.settings_gui, text='Control Channel Column Number: ')
        self.channel_control_label.grid(row=1, column=0)
        self.channel_control_entry = ttk.Combobox(self.settings_gui, values=self.doric_name_list)
        # self.channel_control_entry = tk.Entry(self.settings_gui)
        self.channel_control_entry.grid(row=1, column=2)
        self.channel_control_entry.current(self.channel_control_index)

        self.channel_active_label = tk.Label(self.settings_gui, text='Active Channel Column Number: ')
        self.channel_active_label.grid(row=2, column=0)
        self.channel_active_entry = ttk.Combobox(self.settings_gui, values=self.doric_name_list)
        # self.channel_active_entry = tk.Entry(self.settings_gui)
        self.channel_active_entry.grid(row=2, column=2)
        # self.channel_active_entry.insert(END,self.channel_active_var)
        self.channel_active_entry.current(self.channel_active_index)

        self.channel_ttl_label = tk.Label(self.settings_gui, text='TTL Channel Column Number: ')
        self.channel_ttl_label.grid(row=3, column=0)
        self.channel_ttl_entry = ttk.Combobox(self.settings_gui, values=self.doric_name_list)
        # self.channel_ttl_entry = tk.Entry(self.settings_gui)
        self.channel_ttl_entry.grid(row=3, column=2)
        # self.channel_ttl_entry.insert(END,self.channel_ttl_var)
        self.channel_ttl_entry.current(self.channel_ttl_index)

        self.low_pass_freq_label = tk.Label(self.settings_gui, text='Low Pass Filter Frequency (hz): ')
        self.low_pass_freq_label.grid(row=4, column=0)
        self.low_pass_freq_entry = tk.Entry(self.settings_gui)
        self.low_pass_freq_entry.grid(row=4, column=2)
        self.low_pass_freq_entry.insert(END, self.low_pass_var)

        self.centered_z_checkbutton = tk.Checkbutton(self.settings_gui, text='Centered Z-Score',
                                                     variable=self.centered_z_var)
        self.centered_z_checkbutton.grid(row=5, column=1)

        self.settings_finish_button = tk.Button(self.settings_gui, text='Finish', command=self.settings_commit)
        self.settings_finish_button.grid(row=6, column=1)

    def settings_commit(self):
        self.channel_control_var = str(self.channel_control_entry.get())
        self.channel_active_var = str(self.channel_active_entry.get())
        self.channel_ttl_var = str(self.channel_ttl_entry.get())
        self.channel_control_index = int(self.channel_control_entry.current())
        self.channel_active_index = int(self.channel_active_entry.current())
        self.channel_ttl_index = int(self.channel_ttl_entry.current())
        self.low_pass_var = str(self.low_pass_freq_entry.get())

        self.settings_gui.destroy()

    def create_error_report(self, error_text):
        self.error_window = tk.Toplevel()

        self.error_text = tk.Label(self.error_window, text=error_text)
        self.error_text.grid(row=0, column=0)

        self.error_button = tk.Button(self.error_window, text='OK', command=self.close_error_report)
        self.error_button.grid(row=1, column=0)

    def close_error_report(self):
        self.error_window.destroy()

    def run_photometry_analysis(self):
        self.curr_dir = os.getcwd()
        if sys.platform == 'linux' or sys.platform == 'darwin':
            self.folder_symbol = '/'
        elif sys.platform == 'win32':
            self.folder_symbol = '\\'

        self.output_path = self.curr_dir + self.folder_symbol + 'Output' + self.folder_symbol

        if self.abet_file_path == '' and self.anymaze_file_path == '':
            self.create_error_report('No ABET or Anymaze file defined. Please select a filepath in order to start.')
            return

        if not os.path.isfile(self.abet_file_path) and self.abet_file_path != '':
            self.create_error_report('File path for ABET File is not valid. Please select a new filepath.')
            return

        if not os.path.isfile(self.anymaze_file_path) and self.anymaze_file_path != '':
            self.create_error_report('File path for Anymaze file is not valid. Please select a new filepath.')
            return

        if self.doric_file_path == '':
            self.create_error_report('No Doric file defined. Please select a filepath in order to start.')

        if not os.path.isfile(self.doric_file_path):
            self.create_error_report('Doric file is not valid. Please select a new filepath')

        if self.abet_file_path != '' and self.anymaze_file_path == '':

            self.photometry_object = PhotometryData()
            self.photometry_object.load_doric_data(self.doric_file_path, self.channel_control_index,
                                                   self.channel_active_index, self.channel_ttl_index)
            self.photometry_object.load_abet_data(self.abet_file_path)

            if self.iti_normalize == 1:
                self.photometry_object.abet_trial_definition(self.abet_trial_start_var, self.abet_trial_end_var)

            if self.event_position_var == '':
                self.photometry_object.abet_search_event(start_event_id=self.event_id_var,
                                                         start_event_group=self.event_group_var,
                                                         extra_prior_time=float(self.event_prior_var),
                                                         extra_follow_time=float(self.event_follow_var),
                                                         start_event_item_name=self.event_name_var)
            else:
                self.photometry_object.abet_search_event(start_event_id=self.event_id_var,
                                                         start_event_group=self.event_group_var,
                                                         extra_prior_time=float(self.event_prior_var),
                                                         extra_follow_time=float(self.event_follow_var),
                                                         start_event_position=self.event_position_var,
                                                         start_event_item_name=self.event_name_var)

            self.photometry_object.abet_doric_synchronize()

            self.photometry_object.doric_process(int(self.low_pass_var))

            if self.centered_z_var.get() == 0:
                self.photometry_object.trial_separator(whole_trial_normalize=True,
                                                       trial_definition=True,
                                                       trial_iti_pad=float(self.abet_trial_iti_var))
            elif self.centered_z_var.get() == 1:
                self.photometry_object.trial_separator(whole_trial_normalize=False,
                                                       trial_definition=True,
                                                       trial_iti_pad=float(self.abet_trial_iti_var))

            if self.simple_var.get() == 1:
                self.photometry_object.write_data('Simple')
            if self.timed_var.get() == 1:
                self.photometry_object.write_data('Timed')
            if self.full_var.get() == 1:
                self.photometry_object.write_data('Full')
            if self.simplef_var.get() == 1:
                self.photometry_object.write_data('SimpleF')
            if self.timedf_var.get() == 1:
                self.photometry_object.write_data('TimedF')

            self.confirmation_window = tk.Toplevel()
            self.confirmation_text = tk.Label(self.confirmation_window, text='Files have been generated')
            self.confirmation_text.grid(row=0, column=0)

            self.confirmation_button = tk.Button(self.confirmation_window, text='Continue',
                                                 command=self.close_confirmation)
            self.confirmation_button.grid(row=1, column=0)

        if self.anymaze_file_path != '' and self.doric_file_path != '':
            self.photometry_object = PhotometryData()
            self.photometry_object.load_doric_data(self.doric_file_path, self.channel_control_index,
                                                   self.channel_active_index, self.channel_ttl_index)
            self.photometry_object.load_anymaze_data(self.anymaze_file_path)

            self.photometry_object.anymaze_search_event_or(event1_name=self.anymaze_event1_column_var,
                                                           event1_operation=self.anymaze_event1_operation_var,
                                                           event1_value=self.anymaze_event1_value_var,
                                                           event2_name=self.anymaze_event2_column_var,
                                                           event2_operation=self.anymaze_event2_operation_var,
                                                           event2_value=self.anymaze_event2_value_var,
                                                           event3_name=self.anymaze_event3_column_var,
                                                           event3_operation=self.anymaze_event3_operation_var,
                                                           event3_value=self.anymaze_event3_value_var,
                                                           event_tolerance=float(self.anymaze_tolerance_var),
                                                           extra_prior_time=float(self.anymaze_extra_prior_var),
                                                           extra_follow_time=float(self.anymaze_extra_follow_var))

            self.photometry_object.anymaze_doric_synchronize_or()

            self.photometry_object.doric_process(int(self.low_pass_var))

            if self.centered_z_var.get() == 1:
                self.photometry_object.trial_separator(whole_trial_normalize=True,
                                                       trial_definition=False)
            elif self.centered_z_var.get() == 0:
                self.photometry_object.trial_separator(whole_trial_normalize=False,
                                                       trial_definition=False,
                                                       trial_iti_pad=float(self.anymaze_event_pad_var))

            if self.simple_var.get() == 1:
                self.photometry_object.write_data('Simple')
            if self.timed_var.get() == 1:
                self.photometry_object.write_data('Timed')
            if self.full_var.get() == 1:
                self.photometry_object.write_data('Full')
            if self.simplef_var.get() == 1:
                self.photometry_object.write_data('SimpleF')
            if self.timedf_var.get() == 1:
                self.photometry_object.write_data('TimedF')

            self.confirmation_window = tk.Toplevel()
            self.confirmation_text = tk.Label(self.confirmation_window, text='Files have been generated')
            self.confirmation_text.grid(row=0, column=0)

            self.confirmation_button = tk.Button(self.confirmation_window, text='Continue',
                                                 command=self.close_confirmation)
            self.confirmation_button.grid(row=1, column=0)

    def close_confirmation(self):
        self.confirmation_window.destroy()


Main_GUI = PhotometryGUI()
