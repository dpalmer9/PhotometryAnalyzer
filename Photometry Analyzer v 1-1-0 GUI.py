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
from tkinter import ttk
import numpy as np
from tkinter import filedialog
from scipy import fftpack
from scipy import signal
import pandas as pd


class Photometry_Data:
    def __init__(self):

        self.curr_cpu_core_count = os.cpu_count()
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
        abet_file = open(self.abet_file_path)
        abet_csv_reader = csv.reader(abet_file)
        abet_data_list = list()
        abet_name_list = list()
        colnames_found = False
        for row in abet_csv_reader:
            if colnames_found == False:
                if len(row) == 0:
                    continue
                if row[0] == 'Evnt_Time':
                    colnames_found = True
                    abet_name_list = [row[0],row[1],row[2],row[3],row[5],row[8]]
                else:
                    continue
            else:
                abet_data_list.append([row[0],row[1],row[2],row[3],row[5],row[8]])
        abet_file.close()
        abet_numpy = np.array(abet_data_list)
        self.abet_pandas = pd.DataFrame(data=abet_numpy,columns=abet_name_list)

    def load_doric_data(self,filepath,ch1_col,ch2_col,ttl_col):
        self.doric_file_path = filepath
        self.doric_loaded = True
        doric_file = open(self.doric_file_path)
        doric_csv_reader = csv.reader(doric_file)
        first_row_read = False
        second_row_read = False
        doric_name_list = list()
        doric_list = list()
        for row in doric_csv_reader:
            if first_row_read == False:
                first_row_read = True
                continue
            if second_row_read == False and first_row_read == True:
                doric_name_list = [row[0],row[ch1_col],row[ch2_col],row[ttl_col]]
                second_row_read = True
                continue
            else:
                doric_list.append([row[0],row[ch1_col],row[ch2_col],row[ttl_col]])
        doric_file.close()
        doric_numpy = np.array(doric_list)
        self.doric_pandas = pd.DataFrame(data=doric_numpy,columns=doric_name_list)
        self.doric_pandas.columns = ['Time','Control','Active','TTL']
        self.doric_pandas = self.doric_pandas.astype('float')
                
                

    def load_anymaze_data(self,filepath):
        self.anymaze_file_path = filepath
        self.anymaze_loaded = True

    def abet_trial_definition(self,start_event_group,end_event_group,extra_prior_time=0,extra_follow_time=0):
        if self.abet_loaded == False:
            return None


        filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'] == str(start_event_group)) | (self.abet_pandas['Item_Name'] == str(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
        
        if filtered_abet.iloc[0,3] != str(start_event_group):
            print('FAILED')
        
        trial_times = filtered_abet.Evnt_Time
        trial_times = trial_times.reset_index(drop=True)
        start_times = trial_times.iloc[::2]
        start_times = start_times.reset_index(drop=True)
        end_times = trial_times.iloc[1::2]
        end_times = end_times.reset_index(drop=True)
        self.trial_definition_times = pd.concat([start_times,end_times],axis=1)
        self.trial_definition_times.columns = ['Start_Time','End_Time']
        self.trial_definition_times = self.trial_definition_times.reset_index(drop=True)

    def abet_search_event(self,start_event_id='1',start_event_group='',start_event_item_name='',start_event_position=[''],
                          end_event_id='1',end_event_group='',end_event_item_name=list(''),end_event_position=[''],centered_event=False,
                          extra_prior_time=0,extra_follow_time=0):
        touch_event_names = ['Touch Up Event','Touch Down Event','Whisker - Clear Image by Position']
        if start_event_id in touch_event_names:
            filtered_abet = self.abet_pandas.loc[(self.abet_pandas['Evnt_Name'] == str(start_event_id)) & (self.abet_pandas['Group_ID'] == str(start_event_group)) & 
                                                 (self.abet_pandas['Item_Name'] == str(start_event_item_name)) & (self.abet_pandas['Arg1_Value'] == str(start_event_position)),:] 
    
        else:
            filtered_abet = self.abet_pandas.loc[(self.abet_pandas['Evnt_Name'] == str(start_event_id)) & (self.abet_pandas['Group_ID'] == str(start_event_group)) &
                                                (self.abet_pandas['Item_Name'] == str(start_event_item_name)),:]
            
        
        self.abet_event_times = filtered_abet.loc[:,'Evnt_Time']
        self.abet_event_times = self.abet_event_times.reset_index(drop=True)
        self.abet_event_times = pd.to_numeric(self.abet_event_times, errors='coerce')
        abet_start_times = self.abet_event_times - extra_prior_time
        abet_end_times = self.abet_event_times + extra_follow_time
        self.abet_event_times = pd.concat([abet_start_times,abet_end_times],axis=1)
        self.abet_event_times.columns = ['Start_Time','End_Time']

    def anymaze_search_event_OR(self,event_type='distance',distance_threshold=3.00,distance_event_tolerance=0.03,heading_error_threshold=20,centered_event=True,extra_prior_time=2.5,extra_follow_time=2.5):
        return



    def abet_doric_synchronize(self):
        if self.abet_loaded == False:
            return None
        if self.doric_loaded == False:
            return None
        
        doric_ttl_active = self.doric_pandas.loc[(self.doric_pandas['TTL'] > 3.00),]
        abet_ttl_active = self.abet_pandas.loc[(self.abet_pandas['Item_Name'] == 'TTL #1'),]

        doric_time = doric_ttl_active.iloc[0,0]
        doric_time = doric_time.astype(float)
        doric_time = np.asscalar(doric_time)
        abet_time = abet_ttl_active.iloc[0,0]
        abet_time = float(abet_time)

        self.abet_doric_sync_value = doric_time - abet_time
        
        self.doric_time = pd.to_numeric(self.doric_pandas['Time'])

        self.doric_pandas['Time'] = self.doric_time - self.abet_doric_sync_value
                                        

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

    def doric_process(self,filter_frequency=6):

        time_data = self.doric_pandas['Time'].to_numpy()
        f0_data = self.doric_pandas['Control'].to_numpy()
        f_data = self.doric_pandas['Active'].to_numpy()

        time_data = time_data.astype(float)
        f0_data = f0_data.astype(float)
        f_data = f_data.astype(float)

        self.sample_frequency = len(time_data) / time_data[(len(time_data) - 1)]
        filter_frequency_normalized = filter_frequency / (self.sample_frequency/2)
        butter_filter = signal.butter(N=2,Wn=filter_frequency,
                                           btype='lowpass',analog=False,
                                           output='sos',fs=self.sample_frequency)
        filtered_f0 = signal.sosfilt(butter_filter,f0_data)
        filtered_f = signal.sosfilt(butter_filter,f_data)
        
        f0_a_data = np.vstack([filtered_f0,np.ones(len(filtered_f0))]).T
        m,c = np.linalg.lstsq(f0_a_data,filtered_f,rcond=None)[0]
        f0_fitted = (filtered_f0.astype(np.float) * m) + c

        delta_f = (filtered_f.astype(float) - f0_fitted.astype(float)) / f0_fitted.astype(float)

        self.doric_pd = pd.DataFrame(time_data)
        self.doric_pd['DeltaF'] = delta_f
        self.doric_pd = self.doric_pd.rename(columns={0:'Time',1:'DeltaF'})

    def trial_separator(self,normalize=True,whole_trial_normalize=True,normalize_side = 'Left',trial_definition = False,trial_iti_pad=0):
        if self.abet_loaded == False and self.anymaze_loaded == False:
            return
        left_selection_list = ['Left','Before','L','l','left','before',1]
        right_selection_list = ['Right','right','R','r','After','after',2]

        trial_num = 1
        
        self.abet_time_list = self.abet_event_times
        
        length_time = self.abet_time_list.iloc[0,1]- self.abet_time_list.iloc[0,0]
        
        measurements_per_interval = length_time * self.sample_frequency
        if trial_definition == False:

            for index, row in self.abet_time_list.iterrows():
                start_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index,'Start_Time']).abs().idxmin()
                end_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index,'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index,'End_Time']:
                    end_index += 1

                while len(range(start_index,(end_index + 1))) < measurements_per_interval:
                    end_index += 1
                    
                while len(range(start_index,(end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                if whole_trial_normalize == False:
                    if normalize_side in left_selection_list:
                        norm_start_time = self.abet_time_list.loc[index,'Start_Time']
                        norm_end_time = self.abet_time_list.loc[index,'Start_Time'] + trial_iti_pad
                        iti_deltaf = trial_deltaf.loc[
                            trial_deltaf['Time'] < norm_end_time, 'DeltaF']
                    elif normalize_side in right_selection_list:
                        norm_start_time = self.abet_time_list.loc[index,'End_Time'] - trial_iti_pad
                        norm_end_time = self.abet_time_list.loc[index,'End_Time']
                        iti_deltaf = trial_deltaf.loc[
                            trial_deltaf['Time'] > norm_start_time, 'DeltaF']
                    z_mean = iti_deltaf.mean()
                    z_sd = iti_deltaf.std()
                else:
                    deltaf_split = trial_deltaf.loc[:, 'DeltaF']
                    z_mean = deltaf_split.mean()
                    z_sd = deltaf_split.std()
                trial_deltaf['zscore'] = (trial_deltaf['DeltaF'] - z_mean) / z_sd

                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})
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

                    self.partial_dataframe[colname_2] = trial_deltaf['zscore']
                    self.final_dataframe[colname_1] = trial_deltaf['Time']
                    self.final_dataframe[colname_2] = trial_deltaf['zscore']
                    trial_num += 1
                    
        elif trial_definition == True:
            for index, row in self.abet_time_list.iterrows():
                start_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index,'Start_Time']).abs().idxmin()
                end_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index,'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index,'End_Time']:
                    end_index += 1

                while len(range(start_index,(end_index + 1))) < measurements_per_interval:
                    end_index += 1
                    
                while len(range(start_index,(end_index + 1))) > measurements_per_interval:
                    end_index -= 1

                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                if whole_trial_normalize == False:
                    if normalize_side in self.left_selection_list:
                        trial_start_index = self.trial_definition_times['Start_Time'].sub(self.abet_time_list.loc[index,'Start_Time']).abs().idxmin()
                        trial_start_window = self.trial_definition_times.iloc[trial_start_index,0]
                        trial_iti_window = trial_start_window - trial_iti_pad
                        iti_data = self.doric_pd.loc[(self.doric_pd[''] >= trial_iti_window) & (self.doric_pd[''] <= trial_start_window),'DeltaF']
                    elif normalize_side in self.right_selection_list:
                        trial_end_index = self.trial_definition_times['End_Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()
                        trial_end_window = self.trial_definition_times.iloc[trial_end_index,0]
                        trial_iti_window = trial_end_window + trial_iti_pad
                        iti_data = self.doric_pd.loc[(self.doric_pd['Time'] >= trial_end_window) & (self.doric_pd['Time'] <= trial_iti_window),'DeltaF']

                    z_mean = iti_data.mean()
                    z_sd = iti_data.std()
                else:
                    deltaf_split = trial_deltaf.loc[:, 'DeltaF']
                    z_mean = deltaf_split.mean()
                    z_sd = deltaf_split.std()
                trial_deltaf.loc[:,'zscore'] = (trial_deltaf.loc[:,'DeltaF'] - z_mean) / z_sd

                colname_1 = 'Time Trial ' + str(trial_num)
                colname_2 = 'Z-Score Trial ' + str(trial_num)

                if trial_num == 1:
                    self.final_dataframe = trial_deltaf.loc[:, ('Time', 'zscore')]
                    self.final_dataframe = self.final_dataframe.reset_index(drop=True)
                    self.final_dataframe = self.final_dataframe.rename(
                        columns={'Time': colname_1, 'zscore': colname_2})

                    self.partial_dataframe = trial_deltaf.loc[:, 'zscore']
                    self.partial_dataframe = self.partial_dataframe.to_frame()
                    self.partial_dataframe = self.partial_dataframe.reset_index(drop=True)
                    self.partial_dataframe = self.partial_dataframe.rename(columns={'zscore': colname_2})
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

                    self.partial_dataframe[colname_2] = trial_deltaf['zscore']
                    self.final_dataframe[colname_1] = trial_deltaf['Time']
                    self.final_dataframe[colname_2] = trial_deltaf['zscore']
                    trial_num += 1

    def write_data(self,output_data,include_abet=False):
        processed_list = [1,'Full','full']
        partial_list = [2,'Simple','simple']
        final_list = [3,'Timed','timed']

        if self.abet_loaded == True:
            if include_abet == True:
                end_path = filedialog.asksaveasfilename(title='Save Output Data',
                                                        filetypes=(('Excel File', '*.xlsx'), ('all files', '*.')))

                abet_file = open(self.abet_file_path)
                abet_csv_reader = csv.reader(abet_file)
                colnames_found = False
                colnames = list()
                abet_raw_data = list()

                for row in abet_csv_reader:
                    if colnames_found == False:
                        if len(row) == 0:
                            continue

                        if row[0] == 'Evnt_Time':
                            colnames_found = True
                            colnames = row
                            continue
                        else:
                            continue
                    else:
                        abet_raw_data.append(row)

                self.abet_pd = pd.DataFrame(self.abet_raw_data,columns=self.colnames)

                if output_data in processed_list:
                    with pd.ExcelWriter(self.end_path) as writer:
                        self.doric_pd.to_excel(writer, sheet_name='Photometry Data',index=False)
                        self.abet_pd.to_excel(writer, sheet_name='ABET Trial Data',index=False)
                elif output_data in partial_list:
                    with pd.ExcelWriter(self.end_path) as writer:
                        self.partial_dataframe.to_excel(writer, sheet_name='Photometry Data',index=False)
                        self.abet_pd.to_excel(writer, sheet_name='ABET Trial Data',index=False)
                elif output_data in final_list:
                    with pd.ExcelWriter(self.end_path) as writer:
                        self.final_dataframe.to_excel(writer, sheet_name='Photometry Data',index=False)
                        self.abet_pd.to_excel(writer, sheet_name='ABET Trial Data',index=False)

                return

        current_time = datetime.now()
        current_time_string = current_time.strftime('%d-%m-%Y %H-%M-%S')

        file_path_string = self.main_folder_path + self.folder_symbol + 'Output' + self.folder_symbol +  output_data + self.current_time_string + '.csv'

        if output_data in processed_list:
            self.doric_pd.to_csv(self.file_path_string,index=False)
        elif output_data in partial_list:
            self.partial_dataframe.to_csv(self.file_path_string,index=False)
        elif output_data in final_list:
            self.final_dataframe.to_csv(self.file_path_string,index=False)

class Photometry_GUI:
    def __init__(self):
        if sys.platform == 'linux'or sys.platform == 'darwin':
            self.folder_symbol = '/'
        elif sys.platform == 'win32':
            self.folder_symbol = '\\'
        
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
        self.touch_event_names = ['Touch Up Event','Touch Down Event','Whisker - Clear Image by Position']
        self.position_numbers = ['']
        self.position_state = 'disabled'
        
        self.curr_dir = os.getcwd()
        self.config_path = self.curr_dir + self.folder_symbol + 'Photometry.cfg'
        self.config_file = open(self.config_path)
        self.configurations_list = self.config_file.readlines()
        self.config_file.close()
        self.configurations_list2 = list()
        
        for item in self.configurations_list:
            if '#' in item:
                continue
            item = item.replace('\n','')
            #item.replace(' ','')
            index_string = item.index('=')
            if index_string >= (len(item) - 1):
                if 'filepath' in item:
                    item = ''
                else:
                    item = 0
            else:
                item2 = item[(index_string + 2):len(item)]
                
            self.configurations_list2.append(item2)
        
        self.doric_file_path = self.configurations_list2[0]
        self.abet_file_path = self.configurations_list2[1]
        self.event_id_var = str(self.configurations_list2[2])
        self.event_name_var = str(self.configurations_list2[3])
        self.event_group_var = str(self.configurations_list2[4])
        self.event_position_var = str(self.configurations_list2[5])
        self.event_prior_var = str(self.configurations_list2[6])
        self.event_follow_var = str(self.configurations_list2[7])
        self.abet_trial_start_var = str(self.configurations_list2[8])
        self.abet_trial_end_var = str(self.configurations_list2[9])
        self.abet_trial_iti_var = str(self.configurations_list2[10])
        self.channel_control_var = str(self.configurations_list2[11])
        self.channel_active_var = str(self.configurations_list2[12])
        self.channel_ttl_var = str(self.configurations_list2[13])
        self.low_pass_var = str(self.configurations_list2[14])
        self.centered_z_var.set(int(self.configurations_list2[15]))
        self.title = tk.Label(self.root,text='Photometry Analyzer')
        self.title.grid(row=0,column=1)
        print(self.configurations_list2)

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
        self.abet_field.insert(END,self.abet_file_path)
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
        
        if self.doric_file_path != '':
            self.doric_file_load(path=self.doric_file_path)
        if self.abet_file_path != '':
            self.abet_file_load(path=self.abet_file_path)
        
        self.root.protocol("WM_DELETE_WINDOW", self.close_program)
        self.root.mainloop()
        
    def close_program(self):
        config_list = [self.doric_file_path,self.abet_file_path,self.event_id_var,self.event_name_var,self.event_group_var,
                            self.event_position_var,self.event_prior_var,self.event_follow_var,self.abet_trial_start_var,self.abet_trial_end_var,
                            self.abet_trial_iti_var,self.channel_control_var,self.channel_active_var,self.channel_ttl_var,
                            self.low_pass_var,self.centered_z_var.get()]
        config_index = 0
        configurations_list3 = list()
        for line in self.configurations_list:
            if '#' in line:
                configurations_list3.append(line)
                continue
            index_pos = line.index('=') + 2
            new_line = line[0:index_pos] + str(config_list[config_index]) + '\n'
            configurations_list3.append(new_line)
            config_index += 1
        self.config_file = open(self.config_path,'wt')
        for line in configurations_list3:
            self.config_file.write(line)
        self.config_file.close()
        self.root.destroy()
            
    def abet_setting_load(self):
        if self.event_id_var in self.abet_event_types:
            self.event_id_index = self.abet_event_types.index(self.event_id_var)
            self.abet_group_name = self.abet_pandas.loc[self.abet_pandas['Evnt_Name'] == self.event_id_var,'Item_Name']
            self.abet_group_name = self.abet_group_name.unique()
            self.abet_group_name = list(self.abet_group_name)
            self.abet_group_name = sorted(self.abet_group_name)
            if self.event_name_var in self.abet_group_name:
                self.event_name_index = self.abet_group_name.index(self.event_name_var)
                self.abet_group_numbers = self.abet_pandas.loc[(self.abet_pandas['Evnt_Name'] == self.event_id_var) & 
                                                               (self.abet_pandas['Item_Name'] == self.event_name_var),'Group_ID']
                self.abet_group_numbers = self.abet_group_numbers.unique()
                self.abet_group_numbers = list(self.abet_group_numbers)
                self.abet_group_numbers = sorted(self.abet_group_numbers)
                if self.event_group_var in self.abet_group_numbers:
                    self.event_group_index = self.abet_group_numbers.index(self.event_group_var)
                    if self.event_id_var in self.touch_event_names:
                        self.position_numbers = self.abet_pandas.loc[(self.abet_pandas['Evnt_Name'] == self.event_id_var) & 
                                                                     (self.abet_pandas['Item_Name'] == self.event_name_var) & 
                                                                     (self.abet_pandas['Group_ID'] == self.event_group_var),'Arg1_Value']
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
                self.abet_group_numbers = self.abet_pandas.loc[:,'Group_ID']
                self.abet_group_numbers = self.abet_group_numbers.unique()
                self.abet_group_numbers = list(self.abet_group_numbers)
                self.abet_group_numbers = sorted(self.abet_group_numbers)
        else:
            self.event_id_index = 0
            self.abet_group_name = self.abet_pandas.loc[:,'Item_Name']
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
        

    def doric_file_load(self,path=''):
        if path == '':
            self.doric_file_path = filedialog.askopenfilename(title='Select Doric File', filetypes=(('csv files','*.csv'),('all files','*.')))
        else:
            self.doric_file_path = path
            if os.path.isfile(path) != True:
                self.doric_file_path = ''
                self.doric_field.delete(0,END)
                self.doric_field.insert(END,str(self.doric_file_path))
                return
        self.doric_field.delete(0,END)
        self.doric_field.insert(END,str(self.doric_file_path))

        try:
            doric_file = open(self.doric_file_path)
            doric_csv_reader = csv.reader(doric_file)
            first_row_read = False
            second_row_read = False
            self.doric_name_list = list()
            for row in doric_csv_reader:
                if first_row_read == False:
                    first_row_read = True
                    continue
                if second_row_read == False and first_row_read == True:
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

    def abet_file_load(self,path=''):
        if path == '':
            self.abet_file_path = filedialog.askopenfilename(title='Select ABETII File', filetypes=(('csv files','*.csv'),('all files','*.')))
        else:
            self.abet_file_path = path
            if os.path.isfile(path) != True:
                self.abet_file_path = ''
                self.abet_field.delete(0,END)
                self.abet_field.insert(END,str(self.abet_file_path))
                return
        self.abet_field.delete(0,END)
        self.abet_field.insert(END,str(self.abet_file_path))
        try:
            abet_file = open(self.abet_file_path)
            abet_csv_reader = csv.reader(abet_file)
            abet_data_list = list()
            abet_name_list = list()
            colnames_found = False
            for row in abet_csv_reader:
                if colnames_found == False:
                    if len(row) == 0:
                        continue
                    if row[0] == 'Evnt_Time':
                        colnames_found = True
                        abet_name_list = row
                    else:
                        continue
                else:
                    abet_data_list.append(row)
            abet_file.close()
            abet_numpy = np.array(abet_data_list)
            self.abet_pandas = pd.DataFrame(data=abet_numpy,columns=abet_name_list)
            self.abet_event_types = self.abet_pandas.loc[:,'Evnt_Name']
            self.abet_event_types = self.abet_event_types.unique()
            self.abet_event_types = list(self.abet_event_types)
            self.abet_event_types = sorted(self.abet_event_types)
            if (len(self.abet_event_types) -1) < self.event_id_index:
                self.event_id_index = 0
                
            if self.event_id_var in self.abet_event_types:
                self.event_id_index = self.abet_event_types.index(self.event_id_var)
            else:
                self.event_id_index = 0
                
            self.abet_group_numbers = self.abet_pandas.loc[:,'Group_ID']
            self.abet_group_numbers = self.abet_group_numbers.unique()
            self.abet_group_numbers = list(self.abet_group_numbers)
            self.abet_group_numbers = sorted(self.abet_group_numbers)
            if (len(self.abet_group_numbers) - 1) < self.event_group_index:
                self.event_group_index = 0
                
            if self.event_group_var in self.abet_group_numbers:
                self.event_group_index = self.abet_group_numbers.index(self.event_group_var)
            else:
                self.event_group_index = 0
                
            self.abet_group_name = self.abet_pandas.loc[:,'Item_Name']
            self.abet_group_name = self.abet_group_name.unique()
            self.abet_group_name = list(self.abet_group_name)
            self.abet_group_name = sorted(self.abet_group_name)
            if (len(self.abet_group_name) - 1) < self.event_name_index:
                self.event_name_index = 0
            
            if self.event_name_var in self.abet_group_name:
                self.event_name_index = self.abet_group_name.index(self.event_name_var)
            else:
                self.event_name_index = 0
                
            
            self.abet_iti_group_name = self.abet_pandas.loc[(self.abet_pandas['Evnt_Name'] == 'Condition Event'),'Item_Name']
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
    def anymaze_file_load(self):
        self.anymaze_file_path = filedialog.askopenfilename(title='Select Anymaze File', filetypes=(('csv files','*.csv'),('all files','*.')))
        self.anymaze_field.delete(0,END)
        
        self.anymaze_field.insert(END,str(self.anymaze_file_path))

    def abet_event_name_check(self,event):
        self.abet_group_name = self.abet_pandas.loc[self.abet_pandas['Evnt_Name'] == str(self.event_id_type_entry.get()),'Item_Name']
        self.abet_group_name = self.abet_group_name.unique()
        self.abet_group_name = list(self.abet_group_name)
        self.abet_group_name = sorted(self.abet_group_name)
        self.event_name_entry['values'] = self.abet_group_name
        if str(self.event_id_type_entry.get()) in self.touch_event_names:
            self.event_position_entry.config(state='normal')
        else:
            self.event_position_entry.config(state='disabled')
        try:
            self.abet_group_numbers = self.abet_pandas.loc[self.abet_pandas['Evnt_Name'] == str(self.event_id_type_entry.get()),'Group_ID']
            self.abet_group_numbers = self.abet_group_numbers.unique()
            self.abet_group_numbers = list(self.abet_group_numbers)
            self.abet_group_numbers = sorted(self.abet_group_numbers)
            self.event_group_entry['values'] = self.abet_group_numbers
        except:
            self.abet_group_numbers = self.abet_pandas.loc[:,'Group_ID']
            self.abet_group_numbers = self.abet_group_numbers.unique()
            self.abet_group_numbers = list(self.abet_group_numbers)
            self.abet_group_numbers = sorted(self.abet_group_numbers)
            self.event_group_entry['values'] = self.abet_group_numbers
    def abet_item_name_check(self,event):
        self.abet_group_numbers = self.abet_pandas.loc[(self.abet_pandas['Evnt_Name'] == str(self.event_id_type_entry.get())) & 
                                                       (self.abet_pandas['Item_Name'] == str(self.event_name_entry.get())),'Group_ID']
        self.abet_group_numbers = self.abet_group_numbers.unique()
        self.abet_group_numbers = list(self.abet_group_numbers)
        self.abet_group_numbers = sorted(self.abet_group_numbers)
        self.event_group_entry['values'] = self.abet_group_numbers
        
    def abet_group_number_check(self,event):
        if str(self.event_id_type_entry.get()) in self.touch_event_names:
            self.position_numbers = self.abet_pandas.loc[(self.abet_pandas['Evnt_Name'] == str(self.event_id_type_entry.get())) & 
                                                         (self.abet_pandas['Item_Name'] == str(self.event_name_entry.get())) & 
                                                         (self.abet_pandas['Group_ID'] == str(self.event_group_entry.get())),'Arg1_Value']
            self.position_numbers = self.position_numbers.unique()
            self.position_numbers = list(self.position_numbers)
            self.position_numbers = sorted(self.position_numbers)
            self.event_position_entry['values'] = self.position_numbers
        else:
            return
        
    def abet_event_definition_gui(self):
        
        self.iti_zscoring = tk.IntVar()
        self.iti_zscoring.set(self.iti_normalize)
        
        self.abet_event_gui = tk.Toplevel()

        self.abet_event_title = tk.Label(self.abet_event_gui,text='ABET Event Definition')
        self.abet_event_title.grid(row=0,column=1)
        
        self.event_id_type_label = tk.Label(self.abet_event_gui,text='Event Type')
        self.event_id_type_label.grid(row=1,column=0)
        #self.event_id_type_entry = tk.Entry(self.abet_event_gui)
        self.event_id_type_entry = ttk.Combobox(self.abet_event_gui,values=self.abet_event_types)
        self.event_id_type_entry.grid(row=2,column=0)
        self.event_id_type_entry.bind("<<ComboboxSelected>>", self.abet_event_name_check)
        self.event_id_type_entry.current(self.event_id_index)
        #self.event_id_type_entry.insert(END,self.event_id_var)
        
        self.event_name_label = tk.Label(self.abet_event_gui,text='Name of Event')
        self.event_name_label.grid(row=1,column=1)
        #self.event_name_entry = tk.Entry(self.abet_event_gui)
        self.event_name_entry = ttk.Combobox(self.abet_event_gui,values=self.abet_group_name)
        self.event_name_entry.grid(row=2,column=1)
        self.event_name_entry.bind("<<ComboboxSelected>>", self.abet_item_name_check)
        self.event_name_entry.current(self.event_name_index)
        #self.event_name_entry.insert(END,self.event_name_var)

        self.event_group_label = tk.Label(self.abet_event_gui,text='Event Group #')
        self.event_group_label.grid(row=1,column=2)
        #self.event_group_entry = tk.Entry(self.abet_event_gui)
        self.event_group_entry = ttk.Combobox(self.abet_event_gui,values=self.abet_group_numbers)
        self.event_group_entry.grid(row=2,column=2)
        self.event_group_entry.current(self.event_group_index)
        self.event_group_entry.bind("<<ComboboxSelected>>", self.abet_group_number_check)
        #self.event_group_entry.insert(END,self.event_group_var)

        self.event_position_label = tk.Label(self.abet_event_gui,text='Event Position #')
        self.event_position_label.grid(row=3,column=0)
        #self.event_position_entry = tk.Entry(self.abet_event_gui)
        self.event_position_entry = ttk.Combobox(self.abet_event_gui,values=self.abet_group_numbers)
        self.event_position_entry.grid(row=4,column=0)
        self.event_position_entry.current(self.event_position_index)
        if str(self.event_id_type_entry.get()) in self.touch_event_names:
            self.event_position_entry.config(state='normal')
        else:
            self.event_position_entry.config(state='disabled')
        #self.event_position_entry.insert(END,self.event_position_var)


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
        #self.abet_trial_start_entry = tk.Entry(self.abet_event_gui)
        self.abet_trial_start_entry = ttk.Combobox(self.abet_event_gui,values=self.abet_iti_group_name)
        self.abet_trial_start_entry.grid(row=7,column=0)
        self.abet_trial_start_entry.current(self.abet_trial_start_index)
        #self.abet_trial_start_entry.insert(END,self.abet_trial_start_var)

        self.abet_trial_end_group_label = tk.Label(self.abet_event_gui,text='End Event Group #')
        self.abet_trial_end_group_label.grid(row=6,column=1)
        #self.abet_trial_end_entry = tk.Entry(self.abet_event_gui)
        self.abet_trial_end_entry = ttk.Combobox(self.abet_event_gui,values=self.abet_iti_group_name)
        self.abet_trial_end_entry.grid(row=7,column=1)
        self.abet_trial_end_entry.current(self.abet_trial_end_index)
        #self.abet_trial_end_entry.insert(END,self.abet_trial_end_var)

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
        self.event_id_index = int(self.event_id_type_entry.current())
        self.event_group_var = str(self.event_group_entry.get())
        self.event_group_index = int(self.event_group_entry.current())
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


    def settings_menu(self):
        self.settings_gui = tk.Toplevel()


        self.settings_title = tk.Label(self.settings_gui,text='Settings')
        self.settings_title.grid(row=0,column=1)

        self.channel_control_label = tk.Label(self.settings_gui,text='Control Channel Column Number: ')
        self.channel_control_label.grid(row=1,column=0)
        self.channel_control_entry = ttk.Combobox(self.settings_gui,values=self.doric_name_list)
        #self.channel_control_entry = tk.Entry(self.settings_gui)
        self.channel_control_entry.grid(row=1,column=2)
        self.channel_control_entry.current(self.channel_control_index)

        self.channel_active_label = tk.Label(self.settings_gui,text='Active Channel Column Number: ')
        self.channel_active_label.grid(row=2,column=0)
        self.channel_active_entry = ttk.Combobox(self.settings_gui,values=self.doric_name_list)
        #self.channel_active_entry = tk.Entry(self.settings_gui)
        self.channel_active_entry.grid(row=2,column=2)
        #self.channel_active_entry.insert(END,self.channel_active_var)
        self.channel_active_entry.current(self.channel_active_index)

        self.channel_ttl_label = tk.Label(self.settings_gui,text='TTL Channel Column Number: ')
        self.channel_ttl_label.grid(row=3,column=0)
        self.channel_ttl_entry = ttk.Combobox(self.settings_gui,values=self.doric_name_list)
        #self.channel_ttl_entry = tk.Entry(self.settings_gui)
        self.channel_ttl_entry.grid(row=3,column=2)
        #self.channel_ttl_entry.insert(END,self.channel_ttl_var)
        self.channel_ttl_entry.current(self.channel_ttl_index)

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
        self.channel_control_index = int(self.channel_control_entry.current())
        self.channel_active_index = int(self.channel_active_entry.current())
        self.channel_ttl_index = int(self.channel_ttl_entry.current())
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
            self.photometry_object.load_doric_data(self.doric_file_path,self.channel_control_index,self.channel_active_index,self.channel_ttl_index)
            self.photometry_object.load_abet_data(self.abet_file_path)

            if self.iti_normalize == 1:
                self.photometry_object.abet_trial_definition(self.abet_trial_start_var,self.abet_trial_end_var,extra_prior_time=float(self.abet_trial_iti_var))

            if self.event_position_var == '':
                self.photometry_object.abet_search_event(start_event_id=self.event_id_var,start_event_group=self.event_group_var,extra_prior_time=float(self.event_prior_var),
                                                         extra_follow_time=float(self.event_follow_var),centered_event=True,start_event_item_name = self.event_name_var)
            else:
                self.photometry_object.abet_search_event(start_event_id=self.event_id_var,start_event_group=self.event_group_var,extra_prior_time=float(self.event_prior_var),
                                                         extra_follow_time=float(self.event_follow_var),centered_event=True,start_event_position = self.event_position_var,
                                                         start_event_item_name = self.event_name_var)

            self.photometry_object.abet_doric_synchronize()


            self.photometry_object.doric_process(int(self.low_pass_var))
            
            
            if self.centered_z_var.get() == 0:
                self.photometry_object.trial_separator(normalize=True,whole_trial_normalize=False)     
            elif self.centered_z_var.get() == 1:
                self.photometry_object.trial_separator(normalize=True,whole_trial_normalize=True, trial_definition = True)


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
