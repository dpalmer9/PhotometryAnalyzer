## Module Load Section ##
import os
import sys
import csv
import tkinter as tk
import time
from datetime import datetime
from tkinter import N
from tkinter import S
from tkinter import W
from tkinter import E
from tkinter import END
from tkinter import BROWSE
from tkinter import VERTICAL
import numpy as np
from tkinter import filedialog
from scipy import fftpack
from scipy import signal
import pandas as pd


class Photometry_Data:
    def __init__(self):

        self.curr_dir = os.getcwd()
        if sys.platform == 'linux'or sys.platform == 'darwin':
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

        self.abet_loaded = False
        self.abet_searched = False
        self.anymaze_loaded = False

        self.abet_doric_sync_value = 0

        self.extra_prior = 0
        self.extra_follow = 0

    def load_abet_data(self,filepath):
        self.abet_file_path = filepath
        self.abet_loaded = True
        #self.abet_file = pd.read_csv(self.abet_file_path)

    def load_doric_data(self,filepath):
        self.doric_file_path = filepath
        #self.doric_file = pd.read_csv(self.doric_file_path)

    def load_anymaze_data(self,filepath):
        self.anymaze_file_path = filepath
        self.anymaze_loaded = True
        #self.doric_file = pd.read_csv(self.doric_file_path)


    def print_doric_colnames(self):
        self.doric_file = open(self.doric_file_path)
        self.doric_csv_reader = csv.reader(self.doric_file)

        curr_row = 1

        for row in self.doric_csv_reader:
            if curr_row <= 3:
                print(row)
            elif curr_row > 3:
                break
            curr_row += 1

        self.doric_file.close()

    def abet_trial_definition(self,start_event_group,end_event_group,extra_prior_time=0,extra_follow_time=0):
        if self.abet_loaded == False:
            return None




        self.series_started = False
        self.colnames_found = False

        self.extra_prior_definition = extra_prior_time
        self.extra_follow_definition = extra_follow_time

        self.abet_return_list = list()
        self.abet_trial_time_list = list()

        self.abet_file = open(self.abet_file_path)
        self.abet_csv_reader = csv.reader(self.abet_file)

        self.active_start_time = 0
        self.active_end_time = 0

        for row in self.abet_csv_reader:
            if self.colnames_found == False:
                if len(row) == 0:
                    continue
                if row[0] == 'Evnt_Time':
                    self.colnames_found = True
                else:
                    continue
            if self.series_started == False:
                if row[5] == str(start_event_group):
                    self.series_started = True
                    self.active_start_time = float(row[0]) - extra_prior_time
                    if self.active_start_time < 0:
                        self.active_start_time = 0
            else:
                if row[5] == str(end_event_group):
                    self.series_started = False
                    self.active_end_time = float(row[0]) + extra_follow_time
                    self.time_series = [self.active_start_time, self.active_end_time]
                    self.abet_trial_time_list.append(self.time_series)
                elif row[5] == str(start_event_group):
                    self.series_started = True
                    self.active_start_time = float(row[0]) - extra_follow_time


        self.abet_file.close()

    def abet_search_event(self,start_event_id='1',start_event_group='',start_event_item_name=list(''),start_event_position=[''],
                          end_event_id='1',end_event_group='',end_event_item_name=list(''),end_event_position=[''],centered_event=False,
                          extra_prior_time=0,extra_follow_time=0):
        if self.abet_loaded == False:
            return None




        self.series_started = False
        self.colnames_found = False

        self.extra_prior = extra_prior_time
        self.extra_follow = extra_follow_time

        self.abet_return_list = list()
        self.abet_time_list = list()

        self.abet_file = open(self.abet_file_path)
        self.abet_csv_reader = csv.reader(self.abet_file)

        self.active_start_time = 0
        self.active_end_time = 0

        for row in self.abet_csv_reader:
            if self.colnames_found == False:
                if len(row) == 0:
                    continue
                if row[0] == 'Evnt_Time':
                    self.colnames_found = True
                else:
                    continue

            if str(row[1]) == str(start_event_id):
                if centered_event == True:
                    if (row[5] == str(start_event_group)) and (
                            (str(row[3]) in start_event_item_name) or (start_event_item_name[0] == '')) and (
                            (str(row[8]) in start_event_position) or (start_event_position[0] == '')):
                        if (extra_prior_time <= 0) and (extra_follow_time <= 0):
                            return None

                        self.active_start_time = float(row[0]) - extra_prior_time
                        self.active_end_time = float(row[0]) + extra_follow_time

                        if self.active_start_time < 0:
                            self.active_start_time = 0

                        self.time_series = [self.active_start_time,self.active_end_time]
                        self.abet_time_list.append(self.time_series)
                else:
                    if self.series_started == False:
                        if (row[5] == str(start_event_group)) and (
                                (str(row[3]) == str(start_event_item_name)) or (start_event_item_name == '')) and (
                                (str(row[8]) == str(start_event_position)) or (start_event_position == '')):
                            self.series_started = True
                            self.active_start_time = float(row[0]) - extra_prior_time
                            if self.active_start_time < 0:
                                self.active_start_time = 0
                    else:
                        if (row[5] == str(end_event_group)) and (
                                (row[3] == str(end_event_item_name)) or (end_event_item_name == '')) and (
                                (row[8] == str(end_event_position)) or (end_event_position == '')):
                            self.series_started = False
                            self.active_end_time = float(row[0]) + extra_follow_time
                            self.time_series = [self.active_start_time, self.active_end_time]
                            self.abet_time_list.append(self.time_series)
                        elif (row[5] == str(start_event_group)) and (
                                (row[3] == str(start_event_item_name)) or (start_event_item_name == '')) and (
                                (row[7] == str(start_event_position)) or (start_event_position == '')):
                            self.series_started = True
                            self.active_start_time = float(row[0]) - extra_follow_time

        self.abet_file.close()

    def anymaze_search_event_OR(self,event_type='distance',distance_threshold=3.00,distance_event_tolerance=0.03,heading_error_threshold=20,centered_event=True,extra_prior_time=2.5,extra_follow_time=2.5):
        if event_type == 'distance':
            tracking_cols = [[10,12],[14,16]]
            self.active_start_time = 0
            self.active_end_time = 0
            self.anymaze_file = open(self.anymaze_file_path, 'r')
            self.anymaze_csv_reader = csv.reader(self.anymaze_file)
            self.abet_time_list = list()

            for row in self.anymaze_csv_reader:
                if row[0] == 'Time':
                    continue
                for col in tracking_cols:
                    if row[col[0]] == '':
                        continue
                    if row[col[1]] == '':
                        continue
                    if float(row[col[1]]) == 1 and float(row[col[0]]) <= distance_event_tolerance:
                        if (float(row[0]) - (self.active_start_time + extra_prior_time)) < distance_threshold:
                            continue
                        self.active_start_time = float(row[0]) - extra_prior_time
                        self.active_end_time = float(row[0]) + extra_follow_time
                        self.time_series = [self.active_start_time, self.active_end_time]
                        self.abet_time_list.append(self.time_series)
            self.anymaze_file.close()
        elif event_type == 'viewing':
            tracking_cols = [[10,12],[14,16]]
            self.active_start_time = 0
            self.active_end_time = 0
            self.anymaze_file = open(self.anymaze_file_path, 'r')
            self.anymaze_csv_reader = csv.reader(self.anymaze_file)
            self.abet_time_list = list()

            for row in self.anymaze_csv_reader:
                if row[0] == 'Time':
                    continue
                for col in tracking_cols:
                    if row[col[1]] == '':
                        continue
                    if row[col[0]] == '':
                        continue
                    if float(row[col[1]]) == 1 and float(row[col[0]]) > distance_event_tolerance:
                        if (float(row[0]) - (self.active_start_time + extra_prior_time)) < distance_threshold:
                            continue
                        self.active_start_time = float(row[0]) - extra_prior_time
                        self.active_end_time = float(row[0]) + extra_follow_time
                        self.time_series = [self.active_start_time, self.active_end_time]
                        self.abet_time_list.append(self.time_series)
            self.anymaze_file.close()



    def abet_doric_synchronize(self,ttl_col):
        if self.abet_loaded == False:
            return None
        self.abet_ttl_time = 0
        self.doric_ttl_time = 0

        self.ttl_col = 0

        self.abet_file = open(self.abet_file_path)
        self.abet_csv_reader = csv.reader(self.abet_file)
        self.colnames_found = False

        self.doric_file = open(self.doric_file_path)
        self.doric_csv_reader = csv.reader(self.doric_file)

        for row in self.abet_csv_reader:
            if self.colnames_found == False:
                if len(row) == 0:
                    continue

                if row[0] == 'Evnt_Time':
                    self.colnames_found = True
                    self.colnames = row
                    continue
                else:
                    continue
            else:
                if row[3] == 'TTL #1':
                    self.abet_ttl_time = float(row[0])
                    break

        self.abet_file.close()

        self.doric_row = 1
        for row in self.doric_csv_reader:
            if self.doric_row < 3:
                self.doric_row += 1
                continue
            if row[ttl_col] == '':
                continue
            if float(row[ttl_col]) > 1:
                self.doric_ttl_time = float(row[0])
                break


        self.doric_file.close()
        self.abet_doric_sync_value = self.doric_ttl_time - self.abet_ttl_time

    def anymaze_doric_synchronize_OR(self,ttl_col,ttl_interval):
        if self.anymaze_loaded == False:
            return None
        self.anymaze_ttl_time = 0
        self.doric_ttl_time = 0

        self.ttl_col = 0

        self.anymaze_file = open(self.anymaze_file_path)
        self.anymaze_csv_reader = csv.reader(self.anymaze_file)
        self.colnames_found = False
        self.interval_one_started = False
        self.interval_two_ready = False

        self.doric_file = open(self.doric_file_path)
        self.doric_csv_reader = csv.reader(self.doric_file)

        for row in self.anymaze_csv_reader:
            if row[15] == 1:
                if ttl_interval == 1:
                    self.doric_ttl_time = row[0]
                    break
                elif ttl_interval == 2 and self.interval_one_started == False:
                    self.interval_one_started = True
                    continue
                elif ttl_interval == 2 and self.interval_two_ready == True:
                    self.doric_ttl_time = row[0]
                    break
            if row[15] == 0 and self.interval_one_started == True:
                self.interval_two_ready = True


        self.anymaze_file.close()

        self.doric_row = 1
        for row in self.doric_csv_reader:
            if self.doric_row < 3:
                self.doric_row += 1
                continue
            if row[ttl_col] == '':
                continue
            if float(row[ttl_col]) > 1:
                self.doric_ttl_time = float(row[0])
                break


        self.doric_file.close()
        self.abet_doric_sync_value = self.doric_ttl_time - self.anymaze_ttl_time

    def doric_process(self,ch_405_col,ch_465_col,filter_frequency=6):
        self.doric_file = open(self.doric_file_path)
        self.doric_csv_reader = csv.reader(self.doric_file)

        self.colname_list = list()
        self.condensed_doric = list()

        self.doric_row = 1
        for row in self.doric_csv_reader:
            if self.doric_row < 3:
                self.doric_row += 1
                self.colname_list.append([row[0],row[ch_405_col],row[ch_465_col]])
                continue
            self.doric_time = float(row[0]) - self.abet_doric_sync_value
            self.condensed_doric.append([self.doric_time,row[ch_405_col],row[ch_465_col]])

        self.doric_file.close()

        self.condensed_doric = np.asarray(self.condensed_doric).astype('float')
        self.condensed_doric.dtype = 'float'

        self.time_data = self.condensed_doric[:,0].astype(float)
        self.f0_data = self.condensed_doric[:,1].astype(float)
        self.f_data = self.condensed_doric[:,2].astype(float)

        self.sample_frequency = len(self.time_data) / self.time_data[(len(self.time_data) - 1)]
        self.filter_frequency_normalized = filter_frequency / (self.sample_frequency/2)
        self.butter_filter = signal.butter(N=2,Wn=filter_frequency,
                                           btype='lowpass',analog=False,
                                           output='sos',fs=self.sample_frequency)
        self.filtered_f0 = signal.sosfilt(self.butter_filter,self.f0_data)
        self.filtered_f = signal.sosfilt(self.butter_filter,self.f_data)
        
        self.f0_a_data = np.vstack([self.filtered_f0,np.ones(len(self.filtered_f0))]).T
        self.m,self.c = np.linalg.lstsq(self.f0_a_data,self.filtered_f,rcond=None)[0]
        self.f0_fitted = (self.filtered_f0.astype(np.float) * self.m) + self.c

        self.delta_f = (self.filtered_f.astype(float) - self.f0_fitted.astype(float)) / self.f0_fitted.astype(float)

        self.doric_pd = pd.DataFrame(self.time_data)
        self.doric_pd['DeltaF'] = self.delta_f
        self.doric_pd = self.doric_pd.rename(columns={0:'Time',1:'DeltaF'})

    def trial_separator(self,normalize=True,whole_trial_normalize=True,normalize_side = 'Left',trial_definition = False):
        if self.abet_loaded == False and self.anymaze_loaded == False:
            return
        self.left_selection_list = ['Left','Before','L','l','left','before',1]
        self.right_selection_list = ['Right','right','R','r','After','after',2]

        self.trial_num = 1
        
        self.length_time = self.abet_time_list[0][1] - self.abet_time_list[0][0]
        
        self.measurements_per_interval = self.length_time * self.sample_frequency
        
        if trial_definition == False:
            for time_set in self.abet_time_list:
                self.start_index = self.doric_pd['Time'].sub(float(time_set[0])).abs().idxmin()
                self.end_index = self.doric_pd['Time'].sub(float(time_set[1])).abs().idxmin()

                if self.doric_pd.iloc[self.start_index, 0] > float(time_set[0]):
                    self.start_index -= 1

                if self.doric_pd.iloc[self.end_index, 0] < float(time_set[1]):
                    self.end_index += 1

                while len(range(self.start_index,(self.end_index + 1))) < self.measurements_per_interval:
                    self.end_index += 1
                    
                while len(range(self.start_index,(self.end_index + 1))) > self.measurements_per_interval:
                    self.end_index -= 1    
                    
                self.trial_deltaf = self.doric_pd.iloc[self.start_index:self.end_index]
                if whole_trial_normalize == False:
                    if normalize_side in self.left_selection_list:
                        self.norm_start_time = float(time_set[0])
                        self.norm_end_time = float(time_set[0]) + self.extra_prior
                        self.iti_deltaf = self.trial_deltaf.loc[
                            self.trial_deltaf['Time'] < self.norm_end_time, 'DeltaF']
                    elif normalize_side in self.right_selection_list:
                        self.norm_start_time = float(time_set[1]) - self.extra_follow
                        self.norm_end_time = float(time_set[1])
                        self.iti_deltaf = self.trial_deltaf.loc[
                            self.trial_deltaf['Time'] > self.norm_start_time, 'DeltaF']
                    self.z_mean = self.iti_deltaf.mean()
                    self.z_sd = self.iti_deltaf.std()
                else:
                    self.deltaf_split = self.trial_deltaf.loc[:, 'DeltaF']
                    self.z_mean = self.deltaf_split.mean()
                    self.z_sd = self.deltaf_split.std()
                self.trial_deltaf['zscore'] = (self.trial_deltaf['DeltaF'] - self.z_mean) / self.z_sd

                self.colname_1 = 'Time Trial ' + str(self.trial_num)
                self.colname_2 = 'Z-Score Trial ' + str(self.trial_num)

                if self.trial_num == 1:
                    self.final_dataframe = self.trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': self.colname_1, 'zscore': self.colname_2})

                    self.partial_dataframe = self.trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': self.colname_2})
                    self.trial_num += 1
                else:
                    self.trial_deltaf = self.trial_deltaf.reset_index(drop=True)
                    self.dataframe_len = len(self.final_dataframe.index)
                    self.trial_len = len(self.trial_deltaf.index)
                    if self.trial_len > self.dataframe_len:
                        self.len_diff = self.trial_len - self.dataframe_len
                        self.new_index = list(range(self.dataframe_len, (self.dataframe_len + self.len_diff)))
                        self.final_dataframe = self.final_dataframe.reindex(
                            self.final_dataframe.index.union(self.new_index))
                        self.partial_dataframe = self.partial_dataframe.reindex(
                            self.partial_dataframe.index.union(self.new_index))

                    self.partial_dataframe[self.colname_2] = self.trial_deltaf['zscore']
                    self.final_dataframe[self.colname_1] = self.trial_deltaf['Time']
                    self.final_dataframe[self.colname_2] = self.trial_deltaf['zscore']
                    self.trial_num += 1
                    
        elif trial_definition == True:
            for time_set in self.abet_trial_time_list:
                self.start_index = self.doric_pd['Time'].sub(float(time_set[0])).abs().idxmin()
                self.end_index = self.doric_pd['Time'].sub(float(time_set[1])).abs().idxmin()

                if self.doric_pd.iloc[self.start_index, 0] > float(time_set[0]):
                    self.start_index -= 1

                if self.doric_pd.iloc[self.end_index, 0] < float(time_set[1]):
                    self.end_index += 1
                    
                while len(range(self.start_index,(self.end_index + 1))) < self.measurements_per_interval:
                    self.end_index += 1
                    
                while len(range(self.start_index,(self.end_index + 1))) > self.measurements_per_interval:
                    self.end_index -= 1    

                self.trial_deltaf = self.doric_pd[self.start_index:self.end_index]
                if whole_trial_normalize == False:
                    if normalize_side in self.left_selection_list:
                        self.norm_start_time = float(time_set[0])
                        self.norm_end_time = float(time_set[0]) + self.extra_prior_definition
                        self.iti_deltaf = self.trial_deltaf.loc[
                            self.trial_deltaf['Time'] < self.norm_end_time, 'DeltaF']
                    elif normalize_side in self.right_selection_list:
                        self.norm_start_time = float(time_set[1]) - self.extra_follow_definition
                        self.norm_end_time = float(time_set[1])
                        self.iti_deltaf = self.trial_deltaf.loc[
                            self.trial_deltaf['Time'] > self.norm_start_time, 'DeltaF']
                    self.z_mean = self.iti_deltaf.mean()
                    self.z_sd = self.iti_deltaf.std()
                else:
                    self.deltaf_split = self.trial_deltaf.loc[:, 'DeltaF']
                    self.z_mean = self.deltaf_split.mean()
                    self.z_sd = self.deltaf_split.std()
                self.trial_deltaf['zscore'] = (self.trial_deltaf['DeltaF'] - self.z_mean) / self.z_sd



                if self.trial_num == 1:
                    self.trial_dataframe = self.trial_deltaf[:,('Time','zscore')]
                    self.trial_num += 1
                else:
                    self.add_frame = self.trial_deltaf[:,('Time','zscore')]
                    self.trial_dataframe = self.trial_dataframe.append(self.add_frame,sort=False)
                    self.trial_num += 1
            self.trial_num = 1
            for time_set in self.abet_time_list:
                self.start_index = self.trial_dataframe['Time'].sub(float(time_set[0])).abs().idxmin()
                self.end_index = self.trial_dataframe['Time'].sub(float(time_set[1])).abs().idxmin()
                if self.trial_dataframe.iloc[self.start_index, 0] > float(time_set[0]):
                    self.start_index -= 1

                if self.trial_dataframe.iloc[self.end_index, 0] < float(time_set[1]):
                    self.end_index += 1
                self.trial_zscore = self.trial_dataframe[self.start_index:self.end_index]
                self.colname_1 = 'Time Trial ' + str(self.trial_num)
                self.colname_2 = 'Z-Score Trial ' + str(self.trial_num)
                if self.trial_num == 1:
                    self.final_dataframe = self.trial_zscore.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': self.colname_1, 'zscore': self.colname_2})

                    self.partial_dataframe = self.trial_zscore.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': self.colname_2})
                    self.trial_num += 1
                else:
                    self.trial_deltaf = self.trial_zscore.reset_index(drop=True)
                    self.dataframe_len = len(self.final_dataframe.index)
                    self.trial_len = len(self.trial_zscore.index)
                    if self.trial_len > self.dataframe_len:
                        self.len_diff = self.trial_len - self.dataframe_len
                        self.new_index = list(range(self.dataframe_len, (self.dataframe_len + self.len_diff)))
                        self.final_dataframe = self.final_dataframe.reindex(
                            self.final_dataframe.index.union(self.new_index))
                        self.partial_dataframe = self.partial_dataframe.reindex(
                            self.partial_dataframe.index.union(self.new_index))

                    self.partial_dataframe[self.colname_2] = self.trial_zscore['zscore']
                    self.final_dataframe[self.colname_1] = self.trial_zscore['Time']
                    self.final_dataframe[self.colname_2] = self.trial_zscore['zscore']
                    self.trial_num += 1

    def write_data(self,output_data,include_abet=False):
        self.processed_list = [1,'Full','full']
        self.partial_list = [2,'Simple','simple']
        self.final_list = [3,'Timed','timed']

        if self.abet_loaded == True:
            if include_abet == True:
                self.end_path = filedialog.asksaveasfilename(title='Save Output Data',
                                                        filetypes=(('Excel File', '*.xlsx'), ('all files', '*.')))

                self.abet_file = open(self.abet_file_path)
                self.abet_csv_reader = csv.reader(self.abet_file)
                self.colnames_found = False
                self.colnames = list()
                self.abet_raw_data = list()

                for row in self.abet_csv_reader:
                    if self.colnames_found == False:
                        if len(row) == 0:
                            continue

                        if row[0] == 'Evnt_Time':
                            self.colnames_found = True
                            self.colnames = row
                            continue
                        else:
                            continue
                    else:
                        self.abet_raw_data.append(row)

                self.abet_pd = pd.DataFrame(self.abet_raw_data,columns=self.colnames)

                if output_data in self.processed_list:
                    with pd.ExcelWriter(self.end_path) as writer:
                        self.doric_pd.to_excel(writer, sheet_name='Photometry Data',index=False)
                        self.abet_pd.to_excel(writer, sheet_name='ABET Trial Data',index=False)
                elif output_data in self.partial_list:
                    with pd.ExcelWriter(self.end_path) as writer:
                        self.partial_dataframe.to_excel(writer, sheet_name='Photometry Data',index=False)
                        self.abet_pd.to_excel(writer, sheet_name='ABET Trial Data',index=False)
                elif output_data in self.final_list:
                    with pd.ExcelWriter(self.end_path) as writer:
                        self.final_dataframe.to_excel(writer, sheet_name='Photometry Data',index=False)
                        self.abet_pd.to_excel(writer, sheet_name='ABET Trial Data',index=False)

                return

        self.current_time = datetime.now()
        self.current_time_string = self.current_time.strftime('%d-%m-%Y %H-%M-%S')

        self.file_path_string = self.main_folder_path + self.folder_symbol + 'Output' + self.folder_symbol +  output_data + self.current_time_string + '.csv'

        if output_data in self.processed_list:
            self.doric_pd.to_csv(self.file_path_string,index=False)
        elif output_data in self.partial_list:
            self.partial_dataframe.to_csv(self.file_path_string,index=False)
        elif output_data in self.final_list:
            self.final_dataframe.to_csv(self.file_path_string,index=False)

class Photometry_GUI:
    def __init__(self):
        self.root = tk.Tk()

        self.simple_var = tk.IntVar()
        self.timed_var = tk.IntVar()
        self.full_var = tk.IntVar()
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
        self.iti_normalize = 0

        self.title = tk.Label(self.root,text='Photometry Analyzer')
        self.title.grid(row=0,column=1)

        self.doric_label = tk.Label(self.root,text='Doric Filepath:')
        self.doric_label.grid(row=1,column=0)
        self.doric_field = tk.Entry(self.root)
        self.doric_field.grid(row=1,column=1)
        self.doric_button = tk.Button(self.root,text='...',command=self.doric_file_load)
        self.doric_button.grid(row=1,column=2)

        self.abet_label = tk.Label(self.root,text='ABET II Filepath:')
        self.abet_label.grid(row=2,column=0)
        self.abet_field = tk.Entry(self.root)
        self.abet_field.grid(row=2,column=1)
        self.abet_button = tk.Button(self.root,text='...',command=self.abet_file_load)
        self.abet_button.grid(row=2,column=2)

        self.anymaze_label = tk.Label(self.root,text='Anymaze Filepath:')
        self.anymaze_label.grid(row=3,column=0)
        self.anymaze_field = tk.Entry(self.root)
        self.anymaze_field.grid(row=3,column=1)
        self.anymaze_button = tk.Button(self.root,text='...',command=self.anymaze_file_load)
        self.anymaze_button.grid(row=3,column=2)

        self.abet_event_button = tk.Button(self.root,text='ABET Events',command=self.abet_event_definition_gui)
        self.abet_event_button.grid(row=4,column=0)

        self.anymaze_event_button = tk.Button(self.root,text='Anymaze Events')
        self.anymaze_event_button.grid(row=4,column=1)

        self.settings_button = tk.Button(self.root,text='Settings',command=self.settings_menu)
        self.settings_button.grid(row=4,column=2)

        self.output_title = tk.Label(self.root,text='Output')
        self.output_title.grid(row=6,column=1)

        self.simple_output_check = tk.Checkbutton(self.root,text='Simple',variable=self.simple_var)
        self.simple_output_check.grid(row=7,column=0)
        self.timed_output_check = tk.Checkbutton(self.root,text='Timed',variable=self.timed_var)
        self.timed_output_check.grid(row=7,column=1)
        self.full_output_check = tk.Checkbutton(self.root,text='Full',variable=self.full_var)
        self.full_output_check.grid(row=7,column=2)

        self.run_button = tk.Button(self.root,text='Run',command=self.run_photometry_analysis)
        self.run_button.grid(row=8,column=1)
        
        self.root.mainloop()

    def doric_file_load(self):
        self.doric_file_path = filedialog.askopenfilename(title='Select Doric File', filetypes=(('csv files','*.csv'),('all files','*.')))
        self.doric_field.delete(0,END)
        self.doric_field.insert(END,str(self.doric_file_path))

    def abet_file_load(self):
        self.abet_file_path = filedialog.askopenfilename(title='Select ABETII File', filetypes=(('csv files','*.csv'),('all files','*.')))
        self.abet_field.delete(0,END)
        self.abet_field.insert(END,str(self.abet_file_path))

    def anymaze_file_load(self):
        self.anymaze_file_path = filedialog.askopenfilename(title='Select Anymaze File', filetypes=(('csv files','*.csv'),('all files','*.')))
        self.anymaze_field.delete(0,END)
        
        self.anymaze_field.insert(END,str(self.anymaze_file_path))

    def abet_event_definition_gui(self):

        self.iti_zscoring = tk.IntVar()
        self.iti_zscoring.set(self.iti_normalize)
        
        self.abet_event_gui = tk.Toplevel()

        self.abet_event_title = tk.Label(self.abet_event_gui,text='ABET Event Definition')
        self.abet_event_title.grid(row=0,column=1)
        
        self.event_id_type_label = tk.Label(self.abet_event_gui,text='EVENT ID #')
        self.event_id_type_label.grid(row=1,column=0)
        self.event_id_type_entry = tk.Entry(self.abet_event_gui)
        self.event_id_type_entry.grid(row=2,column=0)
        self.event_id_type_entry.insert(END,self.event_id_var)

        self.event_group_label = tk.Label(self.abet_event_gui,text='Event Group #')
        self.event_group_label.grid(row=1,column=1)
        self.event_group_entry = tk.Entry(self.abet_event_gui)
        self.event_group_entry.grid(row=2,column=1)
        self.event_group_entry.insert(END,self.event_group_var)

        self.event_position_label = tk.Label(self.abet_event_gui,text='Event Position #')
        self.event_position_label.grid(row=1,column=2)
        self.event_position_entry = tk.Entry(self.abet_event_gui)
        self.event_position_entry.grid(row=2,column=2)
        self.event_position_entry.insert(END,self.event_position_var)

        self.event_name_label = tk.Label(self.abet_event_gui,text='Name of Event')
        self.event_name_label.grid(row=3,column=0)
        self.event_name_entry = tk.Entry(self.abet_event_gui)
        self.event_name_entry.grid(row=4,column=0)
        self.event_name_entry.insert(END,self.event_name_var)


        self.event_prior_time = tk.Label(self.abet_event_gui,text='Time Prior to Event (sec)')
        self.event_prior_time.grid(row=3,column=1)
        self.event_prior_entry = tk.Entry(self.abet_event_gui)
        self.event_prior_entry.grid(row=4,column=1)
        self.event_prior_entry.insert(END,self.event_prior_var)

        self.event_follow_time = tk.Label(self.abet_event_gui, text='Time Following Event (sec)')
        self.event_follow_time.grid(row=3,column=2)
        self.event_follow_entry = tk.Entry(self.abet_event_gui)
        self.event_follow_entry.grid(row=4,column=2)
        self.event_follow_entry.insert(END,self.event_follow_var)

        self.abet_trial_definition_title = tk.Label(self.abet_event_gui,text='ABET Trial Definition')
        self.abet_trial_definition_title.grid(row=5,column=1)

        self.abet_trial_start_group_label = tk.Label(self.abet_event_gui,text='Start Event Group #')
        self.abet_trial_start_group_label.grid(row=6,column=0)
        self.abet_trial_start_entry = tk.Entry(self.abet_event_gui)
        self.abet_trial_start_entry.grid(row=7,column=0)
        self.abet_trial_start_entry.insert(END,self.abet_trial_start_var)

        self.abet_trial_end_group_label = tk.Label(self.abet_event_gui,text='End Event Group #')
        self.abet_trial_end_group_label.grid(row=6,column=1)
        self.abet_trial_end_entry = tk.Entry(self.abet_event_gui)
        self.abet_trial_end_entry.grid(row=7,column=1)
        self.abet_trial_end_entry.insert(END,self.abet_trial_end_var)

        self.abet_trial_iti_prior_label = tk.Label(self.abet_event_gui,text='ITI Length Prior to Start')
        self.abet_trial_iti_prior_label.grid(row=6,column=2)
        self.abet_trial_iti_entry = tk.Entry(self.abet_event_gui)
        self.abet_trial_iti_entry.grid(row=7,column=2)
        self.abet_trial_iti_entry.insert(END,self.abet_trial_iti_var)

        self.abet_iti_zscore_checkbutton = tk.Checkbutton(self.abet_event_gui,text='Z-Score Based on ITI',variable=self.iti_zscoring)

        self.abet_event_finish_button = tk.Button(self.abet_event_gui,text='Finish',command=self.abet_event_commit)
        self.abet_event_finish_button.grid(row=8,column=1)

    def abet_event_commit(self):
        self.event_id_var = str(self.event_id_type_entry.get())
        self.event_group_var = str(self.event_group_entry.get())
        self.event_position_var = str(self.event_position_entry.get())
        self.event_name_var = str(self.event_name_entry.get())
        self.event_prior_var = str(self.event_prior_entry.get())
        self.event_follow_var = str(self.event_follow_entry.get())
        self.abet_trial_start_var = str(self.abet_trial_start_entry.get())
        self.abet_trial_end_var = str(self.abet_trial_end_entry.get())
        self.abet_trial_iti_var = str(self.abet_trial_iti_entry.get())
        self.iti_normalize = self.iti_zscoring.get()

        self.abet_event_gui.destroy()


    def settings_menu(self):
        self.settings_gui = tk.Toplevel()
        

        self.settings_title = tk.Label(self.settings_gui,text='Settings')
        self.settings_title.grid(row=0,column=1)

        self.channel_control_label = tk.Label(self.settings_gui,text='Control Channel Column Number: ')
        self.channel_control_label.grid(row=1,column=0)
        self.channel_control_entry = tk.Entry(self.settings_gui)
        self.channel_control_entry.grid(row=1,column=2)
        self.channel_control_entry.insert(END,self.channel_control_var)

        self.channel_active_label = tk.Label(self.settings_gui,text='Active Channel Column Number: ')
        self.channel_active_label.grid(row=2,column=0)
        self.channel_active_entry = tk.Entry(self.settings_gui)
        self.channel_active_entry.grid(row=2,column=2)
        self.channel_active_entry.insert(END,self.channel_active_var)

        self.channel_ttl_label = tk.Label(self.settings_gui,text='TTL Channel Column Number: ')
        self.channel_ttl_label.grid(row=3,column=0)
        self.channel_ttl_entry = tk.Entry(self.settings_gui)
        self.channel_ttl_entry.grid(row=3,column=2)
        self.channel_ttl_entry.insert(END,self.channel_ttl_var)

        self.low_pass_freq_label = tk.Label(self.settings_gui,text='Low Pass Filter Frequency (hz): ')
        self.low_pass_freq_label.grid(row=4,column=0)
        self.low_pass_freq_entry = tk.Entry(self.settings_gui)
        self.low_pass_freq_entry.grid(row=4,column=2)
        self.low_pass_freq_entry.insert(END,self.low_pass_var)
        
        self.centered_z_checkbutton = tk.Checkbutton(self.settings_gui,text='Centered Z-Score',variable=self.centered_z_var)
        self.centered_z_checkbutton.grid(row=5,column=1)

        self.settings_finish_button = tk.Button(self.settings_gui,text='Finish',command=self.settings_commit)
        self.settings_finish_button.grid(row=6,column=1)


    def settings_commit(self):
        self.channel_control_var = str(self.channel_control_entry.get())
        self.channel_active_var = str(self.channel_active_entry.get())
        self.channel_ttl_var = str(self.channel_ttl_entry.get())
        self.low_pass_var = str(self.low_pass_freq_entry.get())

        self.settings_gui.destroy()

    def create_error_report(self,error_text):
        self.error_window = tk.Toplevel()

        self.error_text = tk.Label(self.error_window,text=error_text)
        self.error_text.grid(row=0,column=0)

        self.error_button = tk.Button(self.error_window,text='OK',command=self.close_error_report)
        self.error_button.grid(row=1,column=0)

    def close_error_report(self):
        self.error_window.destroy()

    def run_photometry_analysis(self):
        self.curr_dir = os.getcwd()
        if sys.platform == 'linux'or sys.platform == 'darwin':
            self.folder_symbol = '/'
        elif sys.platform == 'win32':
            self.folder_symbol = '\\'

        self.output_path = self.curr_dir + self.folder_symbol + 'Output' + self.folder_symbol

        if self.abet_file_path == '' and self.anymaze_file_path == '':
            self.create_error_report('No ABET or Anymaze file defined. Please select a filepath in order to start.')
            return

        if os.path.isfile(self.abet_file_path) == False and self.abet_file_path != '':
            self.create_error_report('File path for ABET File is not valid. Please select a new filepath.')
            return

        if os.path.isfile(self.anymaze_file_path) == False and self.anymaze_file_path != '':
            self.create_error_report('File path for Anymaze file is not valid. Please select a new filepath.')
            return

        if self.doric_file_path == '':
            self.create_error_report('No Doric file defined. Please select a filepath in order to start.')

        if os.path.isfile(self.doric_file_path) == False:
            self.create_error_report('Doric file is not valid. Please select a new filepath')

        if self.abet_file_path != '' and self.anymaze_file_path == '':
            
            self.photometry_object = Photometry_Data()
            self.photometry_object.load_doric_data(self.doric_file_path)
            self.photometry_object.load_abet_data(self.abet_file_path)

            if self.iti_normalize == 1:
                self.photometry_object.abet_trial_definition(self.abet_trial_start_var,self.abet_trial_end_var,extra_prior_time=float(self.abet_trial_iti_var))

            if self.event_position_var == '':
                self.photometry_object.abet_search_event(start_event_id=self.event_id_var,start_event_group=self.event_group_var,extra_prior_time=float(self.event_prior_var),
                                                         extra_follow_time=float(self.event_follow_var),centered_event=True,start_event_item_name = self.event_name_var)
            else:
                self.photometry_object.abet_search_event(start_event_id=self.event_id_var,start_event_group=self.event_group_var,extra_prior_time=float(self.event_prior_var),
                                                         extra_follow_time=float(self.event_follow_var),centered_event=True,start_event_position = [self.event_position_var],
                                                         start_event_item_name = self.event_name_var)

            self.ttl_col = int(self.channel_ttl_var) - 1
            self.photometry_object.abet_doric_synchronize(self.ttl_col)

            self.control_col = int(self.channel_control_var) - 1
            self.active_col = int(self.channel_active_var) - 1

            self.photometry_object.doric_process(int(self.control_col),int(self.active_col),int(self.low_pass_var))
            
            
            if self.centered_z_var.get() == 0:
                self.photometry_object.trial_separator(normalize=True,whole_trial_normalize=False)     
            elif self.centered_z_var.get() == 1:
                self.photometry_object.trial_separator(normalize=True,whole_trial_normalize=True)


            if self.simple_var.get() == 1:
                self.photometry_object.write_data('Simple')
            if self.timed_var.get() == 1:
                self.photometry_object.write_data('Timed')
            if self.full_var.get() == 1:
                self.photometry_object.write_data('Full')


            self.confirmation_window = tk.Toplevel()
            self.confirmation_text = tk.Label(self.confirmation_window,text='Files have been generated')
            self.confirmation_text.grid(row=0,column=0)

            self.confirmation_button = tk.Button(self.confirmation_window,text='Continue',command=self.close_confirmation)
            self.confirmation_button.grid(row=1,column=0)

    def close_confirmation(self):
        self.confirmation_window.destroy()
        

        

    

        
        
        

Main_GUI = Photometry_GUI()
