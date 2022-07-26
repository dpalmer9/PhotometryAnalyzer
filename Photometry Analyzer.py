## Module Load Section ##
import os
import sys
import csv
import tkinter as tk
import time
import configparser
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
        event_time_colname = ['Evnt_Time','Event_Time']
        colnames_found = False
        for row in abet_csv_reader:
            if colnames_found == False:
                if len(row) == 0:
                    continue
                if row[0] == 'Animal ID':
                    self.animal_id = str(row[1])
                    continue
                if row[0] == 'Date/Time':
                    self.date = str(row[1])
                    self.date = self.date.replace(':','-')
                    self.date = self.date.replace('/','-')
                    continue
                if row[0] in event_time_colname:
                    colnames_found = True
                    self.time_var_name = row[0]
                    self.event_name_col = row[2]
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
                if row[ch1_col] == '' or row[ch2_col] == '' or row[ttl_col] == '':
                    continue
                doric_list.append([row[0],row[ch1_col],row[ch2_col],row[ttl_col]])
        doric_file.close()
        doric_numpy = np.array(doric_list)
        self.doric_pandas = pd.DataFrame(data=doric_numpy,columns=doric_name_list)
        self.doric_pandas.columns = ['Time','Control','Active','TTL']
        self.doric_pandas = self.doric_pandas.astype('float')
                
                

    def load_anymaze_data(self,filepath):
        self.anymaze_file_path = filepath
        self.anymaze_loaded = True
        anymaze_file = open(self.anymaze_file_path)
        anymaze_csv = csv.reader(anymaze_file)
        colname_found = False
        anymaze_data = list()
        anymaze_colnames = list()
        for row in anymaze_csv:
            if colname_found == False:
                anymaze_colnames = row
                colname_found = True
            else:
                anymaze_data.append(row)
        anymaze_file.close()
        anymaze_numpy = np.array(anymaze_data)
        self.anymaze_pandas = pd.DataFrame(data=anymaze_numpy,columns=anymaze_colnames)
        self.anymaze_pandas = self.anymaze_pandas.replace(r'^\s*$', np.nan, regex=True)
        self.anymaze_pandas = self.anymaze_pandas.astype('float')

    def abet_trial_definition(self,start_event_group,end_event_group,extra_prior_time=0,extra_follow_time=0):
        if self.abet_loaded == False:
            return None

        print(isinstance(start_event_group,list))
        if isinstance(start_event_group,list) and isinstance(end_event_group,list):
            #filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'].isin(start_event_group)) | (self.abet_pandas['Item_Name'].isin(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
            event_group_list = start_event_group + end_event_group
            filtered_abet = self.abet_pandas[self.abet_pandas.Item_Name.isin(event_group_list)]
        elif isinstance(start_event_group,list) and not(isinstance(end_event_group,list)):
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'].isin(start_event_group)) | (self.abet_pandas['Item_Name'] == str(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
        elif isinstance(end_event_group,list) and not(isinstance(start_event_group,list)):
             filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'] == str(start_event_group)) | (self.abet_pandas['Item_Name'].isin(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
        else:
            filtered_abet = self.abet_pandas.loc[((self.abet_pandas['Item_Name'] == str(start_event_group)) | (self.abet_pandas['Item_Name'] == str(end_event_group))) & (self.abet_pandas['Evnt_ID'] == '1')]
            
        filtered_abet = filtered_abet.reset_index(drop=True)
        if filtered_abet.iloc[0,3] != str(start_event_group):
            filtered_abet = filtered_abet.drop([0])
            print('FAILED')
        trial_times = filtered_abet.loc[:,self.time_var_name]
        trial_times = trial_times.reset_index(drop=True)
        start_times = trial_times.iloc[::2]
        start_times = start_times.reset_index(drop=True)
        start_times = pd.to_numeric(start_times,errors='coerce')
        end_times = trial_times.iloc[1::2]
        end_times = end_times.reset_index(drop=True)
        end_times = pd.to_numeric(end_times,errors='coerce')
        self.trial_definition_times = pd.concat([start_times,end_times],axis=1)
        self.trial_definition_times.columns = ['Start_Time','End_Time']
        self.trial_definition_times = self.trial_definition_times.reset_index(drop=True)
        

    def abet_search_event(self,start_event_id='1',start_event_group='',start_event_item_name='',start_event_position=[''],
                          filter_event_id='1',filter_event_group='',filter_event_item_name='',filter_event_position=[''],
                          filter_event=False,filter_before=True,centered_event=False,
                          extra_prior_time=0,extra_follow_time=0):
        touch_event_names = ['Touch Up Event','Touch Down Event','Whisker - Clear Image by Position']
        condition_event_names = ['Condition Event']
        variable_event_names = ['Variable Event']
        
        if start_event_id in touch_event_names:
            filtered_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(start_event_id)) & (self.abet_pandas['Group_ID'] == str(start_event_group)) & 
                                                 (self.abet_pandas['Item_Name'] == str(start_event_item_name)) & (self.abet_pandas['Arg1_Value'] == str(start_event_position)),:] 
    
        else:
            filtered_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(start_event_id)) & (self.abet_pandas['Group_ID'] == str(start_event_group)) &
                                                (self.abet_pandas['Item_Name'] == str(start_event_item_name)),:]
            
        if filter_event == True:
            if filter_event_id in condition_event_names:
                filter_event_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(filter_event_id)) & (self.abet_pandas['Group_ID'] == str(filter_event_group)),:]
            elif filter_event_id in variable_event_names:
                filter_event_abet = self.abet_pandas.loc[(self.abet_pandas[self.event_name_col] == str(filter_event_id)) & (self.abet_pandas['Item_Name'] == str(filter_event_item_name)),:]
        
        self.abet_event_times = filtered_abet.loc[:,self.time_var_name]
        self.abet_event_times = self.abet_event_times.reset_index(drop=True)
        self.abet_event_times = pd.to_numeric(self.abet_event_times, errors='coerce')
        
        if filter_event == True:
            if filter_event_id in condition_event_names:
                for index, value in self.abet_event_times.items():
                    sub_values = filter_event_abet.loc[:,self.time_var_name]
                    sub_values = sub_values.astype(dtype='float64')
                    sub_values = sub_values.sub(float(value))
                    if filter_before == True:
                        sub_values[sub_values > 0] = np.nan
                    else:
                        sub_values[sub_values < 0] = np.nan
                    sub_index = sub_values.abs().idxmin(skipna=True)
                    
                    filter_value = filter_event_abet.loc[sub_index,'Item_Name']
                    if filter_value != filter_event_item_name:
                        self.abet_event_times[index] = np.nan
                
                self.abet_event_times = self.abet_event_times.dropna()
                self.abet_event_times = self.abet_event_times.reset_index(drop=True)
            elif filter_event_id in variable_event_names:
                for index, value in self.abet_event_times.items():
                    sub_values = filter_event_abet.loc[:,self.time_var_name]
                    sub_values = sub_values.astype(dtype='float64')
                    sub_values = sub_values.sub(float(value))
                    if filter_before == True:
                        sub_values[sub_values > 0] = np.nan
                    else:
                        sub_values[sub_values < 0] = np.nan
                    sub_index = sub_values.abs().idxmin(skipna=True)
                    
                    filter_value = filter_event_abet.loc[sub_index,'Arg1_Value']
                    if filter_value != filter_event_position:
                        self.abet_event_times[index] = np.nan
                
        
        abet_start_times = self.abet_event_times - extra_prior_time
        abet_end_times = self.abet_event_times + extra_follow_time
        self.abet_event_times = pd.concat([abet_start_times,abet_end_times],axis=1)
        self.abet_event_times.columns = ['Start_Time','End_Time']
        self.event_name = start_event_item_name
        self.extra_follow = extra_follow_time
        self.extra_prior = extra_prior_time
        

    def anymaze_search_event_OR(self,event1_name,event1_operation,event1_value=0,event2_name='None',event2_operation='None',event2_value=0,event3_name='None',event3_operation='None',event3_value=0,
                                event_tolerance = 1.00,extra_prior_time=0,extra_follow_time=0,event_definition='Event Start'):
        def operation_search(event,operation,value=0):
            if operation == 'Active':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] == 1,:]
            elif operation == 'Inactive':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] == 0,:]
            elif operation == 'Less Than':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] < value,:]
            elif operation == 'Less Than or Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] <= value,:]
            elif operation == 'Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] == value,:]
            elif operation == 'Greater Than or Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] >= value,:]
            elif operation == 'Greater Than':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] > value,:]
            elif operation == 'Not Equal':
                search_data = self.anymaze_pandas.loc[self.anymaze_pandas[event] != value,:]

            search_index = search_data.index
            search_index = search_index.tolist()
            return search_index

        
        anymaze_boolean_list = ['Active','Inactive']
        anymaze_operation_list = ['Less Than', 'Less Than or Equal', 'Equal', 'Greater Than or Equal', 'Greater Than','Not Equal']

        if event1_name == 'None' and event2_name == 'None' and event3_name == 'None':
            return
        
        elif event2_name == 'None' and event3_name == 'None' and event1_name != 'None':
            event_index = operation_search(event1_name,event1_operation,event1_value)
        elif event3_name == 'None' and event2_name != 'None' and event1_name != 'None':
            event1_index = operation_search(event1_name,event1_operation,event1_value)
            event2_index = operation_search(event2_name,event2_operation,event2_value)
            event_index_hold = event1_index + event2_index
            event_index = list()
            for item in event_index_hold:
                if event_index_hold.count(item) >= 2:
                    if item not in event_index:
                        event_index.append(item)
            
        else:
            event1_index = operation_search(event1_name,event1_operation,event1_value)
            event2_index = operation_search(event2_name,event2_operation,event2_value)
            event3_index = operation_search(event2_name,event2_operation,event2_value)
            event_index_hold = event1_index + event2_index + event3_index
            event_index = list()
            for item in event_index_hold:
                if event_index_hold.count(item) >= 3:
                    if item not in event_index:
                        event_index.append(item)
            

        #event_index = event_index.sort()


        search_times = self.anymaze_pandas.loc[event_index,'Time']
        search_times = search_times.reset_index(drop=True)

        event_start_times = list()
        event_end_times = list()

        event_start_time = self.anymaze_pandas.loc[0,'Time']

        current_time = self.anymaze_pandas.loc[0,'Time']

        previous_time = self.anymaze_pandas.loc[0,'Time']

        event_end_time = 0

        for index,value in search_times.items():
            previous_time = current_time
            current_time = value
            if event_start_time == self.anymaze_pandas.loc[0,'Time']:
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
            for time in event_start_times:
                final_start_time = time - extra_prior_time
                if final_start_time <= 0:
                    continue
                final_start_times.append(final_start_time)
                final_end_time = time + extra_follow_time
                final_end_times.append(final_end_time)

        elif event_definition == "Event Center":
            center_times = list()
            for index in range(0,(len(event_start_times) -1)):
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
            for time in event_end_times:
                final_start_time = time - extra_prior_time
                if final_start_time <= 0:
                    continue
                final_start_times.append(final_start_time)
                final_end_time = time + extra_follow_time
                final_end_times.append(final_end_time)
        self.anymaze_event_times = pd.DataFrame(final_start_times)
        self.anymaze_event_times['End_Time'] = final_end_times
        self.anymaze_event_times.columns = ['Start_Time','End_Time']
        self.abet_event_times = self.anymaze_event_times
                
                
            
        
        


    def abet_doric_synchronize(self):
        if self.abet_loaded == False:
            return None
        if self.doric_loaded == False:
            return None
        try:
            doric_ttl_active = self.doric_pandas.loc[(self.doric_pandas['TTL'] > 1.00),]
        except:
            print('No TTL Signal Detected. Ending Analysis')
            return
        try:
            abet_ttl_active = self.abet_pandas.loc[(self.abet_pandas['Item_Name'] == 'TTL #1'),]
        except:
            print('ABET II File missing TTL Pulse Output')
            return

        doric_time = doric_ttl_active.iloc[0,0]
        doric_time = doric_time.astype(float)
        doric_time = np.asscalar(doric_time)
        abet_time = abet_ttl_active.iloc[0,0]
        abet_time = float(abet_time)

        self.abet_doric_sync_value = doric_time - abet_time
        
        self.doric_time = pd.to_numeric(self.doric_pandas['Time'])

        self.doric_pandas['Time'] = self.doric_time - self.abet_doric_sync_value
                                        

    def anymaze_doric_synchronize_OR(self):
        if self.anymaze_loaded == False:
            return None
        if self.doric_loaded == False:
            return None

        try:
            doric_ttl_active = self.doric_pandas.loc[(self.doric_pandas['TTL'] > 1.00),]
        except:
            print('No TTL Signal Detected. Ending Analysis')
            return

        try:
            anymaze_ttl_active = self.anymaze_pandas.loc[(self.anymaze_pandas['TTL Pulse active'] > 0),]
        except:
            print('Anymaze File missing TTL Pulse Output')

        doric_time = doric_ttl_active.iloc[0,0]
        doric_time = doric_time.astype(float)
        doric_time = np.asscalar(doric_time)
        anymaze_time = anymaze_ttl_active.iloc[0,0]
        anymaze_time = float(anymaze_time)

        self.anymaze_doric_sync_value = doric_time - anymaze_time
        
        self.doric_time = pd.to_numeric(self.doric_pandas['Time'])

        self.doric_pandas['Time'] = self.doric_time - self.anymaze_doric_sync_value

    def doric_process(self,filter_frequency=6):
        time_data = self.doric_pandas['Time'].to_numpy()
        f0_data = self.doric_pandas['Control'].to_numpy()
        f_data = self.doric_pandas['Active'].to_numpy()

        time_data = time_data.astype(float)
        f0_data = f0_data.astype(float)
        f_data = f_data.astype(float)

        self.sample_frequency = len(time_data) / (time_data[(len(time_data) - 1)] - time_data[0])
        filter_frequency_normalized = filter_frequency / (self.sample_frequency/2)
        butter_filter = signal.butter(N=2,Wn=filter_frequency,
                                           btype='lowpass',analog=False,
                                           output='sos',fs=self.sample_frequency)
        filtered_f0 = signal.sosfilt(butter_filter,f0_data)
        filtered_f = signal.sosfilt(butter_filter,f_data)
        

        filtered_poly = np.polyfit(filtered_f0,filtered_f,1)
        filtered_lobf = np.multiply(filtered_poly[0],filtered_f0) + filtered_poly[1]
        
        delta_f = filtered_f - filtered_lobf

        self.doric_pd = pd.DataFrame(time_data)
        self.doric_pd['DeltaF'] = delta_f
        self.doric_pd = self.doric_pd.rename(columns={0:'Time',1:'DeltaF'})

    def trial_separator(self,normalize=True,whole_trial_normalize=True,normalize_side = 'Left',trial_definition = False,trial_iti_pad=0,event_location='None'):
        if self.abet_loaded == False and self.anymaze_loaded == False:
            return
        left_selection_list = ['Left','Before','L','l','left','before',1]
        right_selection_list = ['Right','right','R','r','After','after',2]
        
        trial_definition_none_list = ['None',0,'0','No',False]
        trial_definition_ind_list = ['Individual',1,'1','Ind','Indv']
        trial_definition_overall_list = ['Overall',2,'2']

        trial_num = 1
        
        self.abet_time_list = self.abet_event_times
        
        
        length_time = self.abet_time_list.iloc[0,1]- self.abet_time_list.iloc[0,0]
        measurements_per_interval = length_time * self.sample_frequency
        if trial_definition in trial_definition_none_list:
            for index, row in self.abet_time_list.iterrows():

                try:
                    start_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index,'Start_Time']).abs().idxmin()
                except:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd['Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()
                except:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

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

                    self.partial_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.final_dataframe.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    trial_num += 1
                    
        elif trial_definition in trial_definition_ind_list:
            for index, row in self.abet_time_list.iterrows():
                try:
                    start_index = self.doric_pd.loc[:,'Time'].sub(self.abet_time_list.loc[index,'Start_Time']).abs().idxmin()
                except:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd.loc[:,'Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()
                except:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

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
                        trial_start_index_diff = self.trial_definition_times.loc[:,'Start_Time'].sub((self.abet_time_list.loc[index,'Start_Time'] + self.extra_prior))#.abs().idxmin()
                        #print(trial_start_index_diff)
                        trial_start_index_diff[trial_start_index_diff > 0] = np.nan
                        trial_start_index = trial_start_index_diff.abs().idxmin(skipna=True)
                        trial_start_window = self.trial_definition_times.iloc[trial_start_index,0]
                        print([(self.abet_time_list.loc[index,'Start_Time'] + self.extra_prior),trial_start_window])
                        trial_iti_window = trial_start_window - float(trial_iti_pad)
                        iti_data = self.doric_pd.loc[(self.doric_pd.loc[:,'Time'] >= trial_iti_window) & (self.doric_pd.loc[:,'Time'] <= trial_start_window),'DeltaF']
                    elif normalize_side in right_selection_list:
                        trial_end_index = self.trial_definition_times.loc[:,'End_Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()
                        trial_end_window = self.trial_definition_times.iloc[trial_end_index,0]
                        trial_iti_window = trial_end_window + trial_iti_pad
                        iti_data = self.doric_pd.loc[(self.doric_pd['Time'] >= trial_end_window) & (self.doric_pd['Time'] <= trial_iti_window),'DeltaF']

                    z_mean = iti_data.mean()
                    z_sd = iti_data.std()
                else:
                    deltaf_split = trial_deltaf.loc[:, 'DeltaF']
                    z_mean = deltaf_split.mean()
                    z_sd = deltaf_split.std()
                    
                trial_deltaf.loc[:,'zscore'] = trial_deltaf.loc[:,'DeltaF'].map(lambda x: ((x - z_mean)/z_sd))
                #trial_deltaf.loc[:,'zscore'] = (trial_deltaf.loc[:,'DeltaF'] - z_mean) / z_sd

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

                    self.partial_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.final_dataframe.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    trial_num += 1
                    
        elif trial_definition in trial_definition_overall_list:
            mod_trial_times = self.trial_definition_times
            mod_trial_times.iloc[-1,1] = np.nan
            mod_trial_times.iloc[0,0] = np.nan
            mod_trial_times['Start_Time'] = mod_trial_times['Start_Time'].shift(-1)
            mod_trial_times = mod_trial_times[:-1]
            for index, row in mod_trial_times.iterrows():
                try:
                    end_index = self.doric_pd.loc[:,'Time'].sub(mod_trial_times.loc[index,'Start_Time']).abs().idxmin()
                except:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    start_index = self.doric_pd.loc[:,'Time'].sub(mod_trial_times.loc[index,'End_Time']).abs().idxmin()
                except:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > mod_trial_times.loc[index,'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < mod_trial_times.loc[index,'End_Time']:
                    end_index += 1

                while len(range(start_index,(end_index + 1))) < measurements_per_interval:
                    end_index += 1
                    
                while len(range(start_index,(end_index + 1))) > measurements_per_interval:
                    end_index -= 1
                    
                iti_deltaf = self.doric_pd.iloc[start_index:end_index]
                iti_deltaf = iti_deltaf.loc[:,'DeltaF']
                if index == 0:
                    full_iti_deltaf = iti_deltaf
                else:
                    full_iti_deltaf = full_iti_deltaf.append(iti_deltaf)
            
            
            
            z_mean = full_iti_deltaf.mean()
            z_sd = full_iti_deltaf.std()
            
            for index, row in self.abet_time_list.iterrows():
                try:
                    start_index = self.doric_pd.loc[:,'Time'].sub(self.abet_time_list.loc[index,'Start_Time']).abs().idxmin()
                except:
                    print('Trial Start Out of Bounds, Skipping Event')
                    continue
                try:
                    end_index = self.doric_pd.loc[:,'Time'].sub(self.abet_time_list.loc[index,'End_Time']).abs().idxmin()
                except:
                    print('Trial End Out of Bounds, Skipping Event')
                    continue

                while self.doric_pd.iloc[start_index, 0] > self.abet_time_list.loc[index,'Start_Time']:
                    start_index -= 1

                while self.doric_pd.iloc[end_index, 0] < self.abet_time_list.loc[index,'End_Time']:
                    end_index += 1

                while len(range(start_index,(end_index + 1))) < measurements_per_interval:
                    end_index += 1
                    
                while len(range(start_index,(end_index + 1))) > measurements_per_interval:
                    end_index -= 1
                
                trial_deltaf = self.doric_pd.iloc[start_index:end_index]
                trial_deltaf.loc[:,'zscore'] = trial_deltaf.loc[:,'DeltaF'].map(lambda x: ((x - z_mean)/z_sd))
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

                    self.partial_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    self.final_dataframe.loc[:,colname_1] = trial_deltaf.loc[:,'Time']
                    self.final_dataframe.loc[:,colname_2] = trial_deltaf.loc[:,'zscore']
                    trial_num += 1
                    
                

    def write_data(self,output_data,include_abet=False,filename_override=''):
        processed_list = [1,'Full','full']
        partial_list = [3,'Simple','simple']
        final_list = [5,'Timed','timed']
        partialf_list = [2,'SimpleF','simplef']
        finalf_list = [4,'TimedF','timedf']

        if self.abet_loaded == True:
            if include_abet == True:
                #end_path = filedialog.asksaveasfilename(title='Save Output Data',
                                                        #filetypes=(('Excel File', '*.xlsx'), ('all files', '*.')))

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


        output_folder = self.main_folder_path + self.folder_symbol + 'Output'
        if (os.path.isdir(output_folder)) == False:
            os.mkdir(output_folder)
        if self.abet_loaded == True and self.anymaze_loaded == False:
            file_path_string = output_folder + self.folder_symbol +  output_data + '-' + self.animal_id + ' ' + self.date + ' ' + self.event_name + '.csv'
        else:
            current_time = datetime.now()
            current_time_string = current_time.strftime('%d-%m-%Y %H-%M-%S')
            file_path_string = output_folder + self.folder_symbol +  output_data + '-' + current_time_string + '.csv'
        
        if filename_override != '':
            file_path_string = output_folder + self.folder_symbol + filename_override

        print(file_path_string)
        if output_data in processed_list:
            self.doric_pd.to_csv(file_path_string,index=False)
        elif output_data in partial_list:
            self.partial_dataframe.to_csv(file_path_string,index=False)
        elif output_data in final_list:
            self.final_dataframe.to_csv(file_path_string,index=False)
        elif output_data in partialf_list:
            self.partial_deltaf.to_csv(file_path_string,index=False)
        elif output_data in finalf_list:
            self.final_deltaf.to_csv(file_path_string,index=False)

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
        self.touch_event_names = ['Touch Up Event','Touch Down Event','Whisker - Clear Image by Position']
        self.position_numbers = ['']
        self.position_state = 'disabled'
        self.event_time_colname = ['Evnt_Time','Event_Time']
        self.event_name_colname = ['Event_Name','Evnt_Name']

        self.anymaze_boolean_list = ['Active','Inactive']
        self.anymaze_operation_list = ['Less Than', 'Less Than or Equal', 'Equal', 'Greater Than or Equal', 'Greater Than','Not Equal']
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
        self.centered_z_var.set(int(self.config_file['Doric']['centered_z']))
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
        self.abet_field.insert(END,self.abet_file_path)
        self.abet_button = tk.Button(self.root,text='...',command=self.abet_file_load)
        self.abet_button.grid(row=2,column=2)

        self.anymaze_label = tk.Label(self.root,text='Anymaze Filepath:')
        self.anymaze_label.grid(row=3,column=0)
        self.anymaze_field = tk.Entry(self.root)
        self.anymaze_field.grid(row=3,column=1)
        self.anymaze_field.insert(END,self.anymaze_file_path)
        self.anymaze_button = tk.Button(self.root,text='...',command=self.anymaze_file_load)
        self.anymaze_button.grid(row=3,column=2)

        self.abet_event_button = tk.Button(self.root,text='ABET Events',command=self.abet_event_definition_gui)
        self.abet_event_button.grid(row=4,column=0)

        self.anymaze_event_button = tk.Button(self.root,text='Anymaze Events',command=self.anymaze_event_description_gui)
        self.anymaze_event_button.grid(row=4,column=1)

        self.settings_button = tk.Button(self.root,text='Settings',command=self.settings_menu)
        self.settings_button.grid(row=4,column=2)

        self.output_title = tk.Label(self.root,text='Output')
        self.output_title.grid(row=6,column=1)

        self.simplef_output_check = tk.Checkbutton(self.root,text='Simple Delta-F',variable=self.simplef_var)
        self.simplef_output_check.grid(row=7,column=1)
        self.simple_output_check = tk.Checkbutton(self.root,text='Simple Z',variable=self.simple_var)
        self.simple_output_check.grid(row=7,column=2)
        self.timedf_output_check = tk.Checkbutton(self.root,text='Timed Delta-F',variable=self.timedf_var)
        self.timedf_output_check.grid(row=8,column=0)
        self.timed_output_check = tk.Checkbutton(self.root,text='Timed Z',variable=self.timed_var)
        self.timed_output_check.grid(row=8,column=1)
        self.full_output_check = tk.Checkbutton(self.root,text='Full Output',variable=self.full_var)
        self.full_output_check.grid(row=7,column=0)

        self.run_button = tk.Button(self.root,text='Run',command=self.run_photometry_analysis)
        self.run_button.grid(row=9,column=1)
        
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
        
        with open(self.config_path,'w') as configfile:
            self.config_file.write(configfile)
        
        
        self.root.destroy()
            
    def abet_setting_load(self):
        if self.event_id_var in self.abet_event_types:
            self.event_id_index = self.abet_event_types.index(self.event_id_var)
            self.abet_group_name = self.abet_pandas.loc[self.abet_pandas[self.abet_event_name_col] == self.event_id_var,'Item_Name']
            self.abet_group_name = self.abet_group_name.unique()
            self.abet_group_name = list(self.abet_group_name)
            self.abet_group_name = sorted(self.abet_group_name)
            if self.event_name_var in self.abet_group_name:
                self.event_name_index = self.abet_group_name.index(self.event_name_var)
                self.abet_group_numbers = self.abet_pandas.loc[(self.abet_pandas[self.abet_event_name_col] == self.event_id_var) & 
                                                               (self.abet_pandas['Item_Name'] == self.event_name_var),'Group_ID']
                self.abet_group_numbers = self.abet_group_numbers.unique()
                self.abet_group_numbers = list(self.abet_group_numbers)
                self.abet_group_numbers = sorted(self.abet_group_numbers)
                if self.event_group_var in self.abet_group_numbers:
                    self.event_group_index = self.abet_group_numbers.index(self.event_group_var)
                    if self.event_id_var in self.touch_event_names:
                        self.position_numbers = self.abet_pandas.loc[(self.abet_pandas[self.abet_event_name_col] == self.event_id_var) & 
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
            except:
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
            except:
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
            except:
                self.anymaze_event3_column_index = self.anymaze_column_names.index('None')
                self.anymaze_event3_operation_state = 'disabled'
                self.anymaze_event3_value_state = 'disabled'
                self.anymaze_event3_operation_index = 0
                self.anymaze_event3_value_var = 0



        

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
            self.abet_pandas = pd.DataFrame(data=abet_numpy,columns=abet_name_list)
            self.abet_event_types = self.abet_pandas.loc[:,self.abet_event_name_col]
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
                
            
            self.abet_iti_group_name = self.abet_pandas.loc[(self.abet_pandas[self.abet_event_name_col] == 'Condition Event'),'Item_Name']
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
    def anymaze_file_load(self,path=''):
        if path == '':
            self.anymaze_file_path = filedialog.askopenfilename(title='Select Anymaze File', filetypes=(('csv files','*.csv'),('all files','*.')))
        else:
            self.anymaze_file_path = path
            if os.path.isfile(path) != True:
                self.anymaze_file_path = ''
                self.anymaze_field.delete(0,END)
                self.anymaze_field.insert(END,str(self.anymaze_file_path))
                return
        self.anymaze_field.delete(0,END)
        self.anymaze_field.insert(END,str(self.anymaze_file_path))

        try:
            anymaze_file = open(self.anymaze_file_path)
            anymaze_csv = csv.reader(anymaze_file)
            colname_found = False
            anymaze_data = list()
            for row in anymaze_csv:
                if colname_found == False:
                    self.anymaze_column_names = row
                    colname_found = True
                else:
                    anymaze_data.append(row)
            anymaze_file.close()
            self.anymaze_column_names.append('None')
            anymaze_numpy = np.array(anymaze_data)
            self.anymaze_pandas = pd.DataFrame(data=anymaze_numpy,columns=self.anymaze_column_names[0:(len(self.anymaze_column_names) -1)])
        except:
            self.anymaze_column_names = ['None']

        if path != '':
            self.anymaze_setting_load()

    def abet_event_name_check(self,event):
        self.abet_group_name = self.abet_pandas.loc[self.abet_pandas[self.abet_event_name_col] == str(self.event_id_type_entry.get()),'Item_Name']
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
            self.abet_group_numbers = self.abet_pandas.loc[self.abet_pandas[self.abet_event_name_col] == str(self.event_id_type_entry.get()),'Group_ID']
            self.abet_group_numbers = self.abet_group_numbers.unique()
            self.abet_group_numbers = list(self.abet_group_numbers)
            self.abet_group_numbers = sorted(self.abet_group_numbers)
            self.event_group_entry.config(values=self.abet_group_numbers)
            if self.abet_group_numbers == 1:
                self.event_group_entry.current(0)
                self.event_group_entry.set(self.abet_group_numbers[0])
        except:
            self.abet_group_numbers = self.abet_pandas.loc[:,'Group_ID']
            self.abet_group_numbers = self.abet_group_numbers.unique()
            self.abet_group_numbers = list(self.abet_group_numbers)
            self.abet_group_numbers = sorted(self.abet_group_numbers)
            self.event_group_entry['values'] = self.abet_group_numbers
            if self.abet_group_numbers == 1:
                self.event_group_entry.current(0)
                self.event_group_entry.set(self.abet_group_numbers[0])
            
        self.abet_event_gui.update()
    def abet_item_name_check(self,event):
        self.abet_group_numbers = self.abet_pandas.loc[(self.abet_pandas[self.abet_event_name_col] == str(self.event_id_type_entry.get())) & 
                                                       (self.abet_pandas['Item_Name'] == str(self.event_name_entry.get())),'Group_ID']
        self.abet_group_numbers = self.abet_group_numbers.unique()
        self.abet_group_numbers = list(self.abet_group_numbers)
        self.abet_group_numbers = sorted(self.abet_group_numbers)
        self.event_group_entry['values'] = self.abet_group_numbers
        if self.abet_group_numbers == 1:
                self.event_group_entry.current(0)
                self.event_group_entry.set(self.abet_group_numbers[0])
        self.abet_event_gui.update()
        
    def abet_group_number_check(self,event):
        if str(self.event_id_type_entry.get()) in self.touch_event_names:
            self.position_numbers = self.abet_pandas.loc[(self.abet_pandas[self.abet_event_name_col] == str(self.event_id_type_entry.get())) & 
                                                         (self.abet_pandas['Item_Name'] == str(self.event_name_entry.get())) & 
                                                         (self.abet_pandas['Group_ID'] == str(self.event_group_entry.get())),'Arg1_Value']
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

        self.anymaze_event_title = tk.Label(self.anymaze_event_gui,text='Anymaze Event Definition')
        self.anymaze_event_title.grid(row=0,column=1)

        self.anymaze_event_type_colname = tk.Label(self.anymaze_event_gui,text='Column')
        self.anymaze_event_type_colname.grid(row=1,column=0)

        self.anymaze_event_operation_colname = tk.Label(self.anymaze_event_gui,text='Function')
        self.anymaze_event_operation_colname.grid(row=1,column=1)

        self.anymaze_event_value_colname = tk.Label(self.anymaze_event_gui,text='Value')
        self.anymaze_event_value_colname.grid(row=1,column=2)

        self.anymaze_event1_colname = ttk.Combobox(self.anymaze_event_gui,values=self.anymaze_column_names)
        self.anymaze_event1_colname.grid(row=2,column=0)
        self.anymaze_event1_colname.bind("<<ComboboxSelected>>", self.anymaze_column_set_event1)
        try:
            self.anymaze_event1_column_index = self.anymaze_column_names.index(self.anymaze_event1_column_var)
            self.anymaze_event1_colname.current(self.anymaze_event1_column_index)
        except:
            self.anymaze_event1_colname.current(0)
        self.anymaze_event1_operation = ttk.Combobox(self.anymaze_event_gui,values=self.anymaze_event1_operation_list)
        self.anymaze_event1_operation.grid(row=2,column=1)
        self.anymaze_event1_operation.config(state=self.anymaze_event1_operation_state)
        try:
            self.anymaze_event1_operation_index = self.anymaze_event1_operation_list.index(self.anymaze_event1_operation_var)
            self.anymaze_event1_operation.current(self.anymaze_event1_operation_index)
        except:
            self.anymaze_event1_operation.current(0)
        self.anymaze_event1_value = tk.Entry(self.anymaze_event_gui)
        self.anymaze_event1_value.grid(row=2,column=2)
        self.anymaze_event1_value.config(state=self.anymaze_event1_value_state)
        if self.anymaze_event1_value_state != "disabled":
            self.anymaze_event1_value.insert(END,self.anymaze_event1_value_var)
        if self.anymaze_event1_column_var != 'None':
            self.anymaze_column_set_event1(event=self.anymaze_event1_column_var)
        self.anymaze_event1_operation.current(self.anymaze_event1_operation_index)

        self.anymaze_event2_colname = ttk.Combobox(self.anymaze_event_gui,values=self.anymaze_column_names)
        self.anymaze_event2_colname.grid(row=3,column=0)
        self.anymaze_event2_colname.bind("<<ComboboxSelected>>", self.anymaze_column_set_event2)
        try:
            self.anymaze_event2_column_index = self.anymaze_column_names.index(self.anymaze_event2_column_var)
            self.anymaze_event2_colname.current(self.anymaze_event2_column_index)
        except:
            self.anymaze_event2_colname.current(0)
        self.anymaze_event2_operation = ttk.Combobox(self.anymaze_event_gui,values=self.anymaze_event2_operation_list)
        self.anymaze_event2_operation.grid(row=3,column=1)
        self.anymaze_event2_operation.config(state=self.anymaze_event2_operation_state)
        try:
            self.anymaze_event2_operation_index = self.anymaze_event2_operation_list.index(self.anymaze_event2_operation_var)
            self.anymaze_event2_operation.current(self.anymaze_event2_operation_index)
        except:
            self.anymaze_event2_operation.current(0)
        self.anymaze_event2_value = tk.Entry(self.anymaze_event_gui)
        self.anymaze_event2_value.grid(row=3,column=2)
        self.anymaze_event2_value.config(state=self.anymaze_event2_value_state)
        if self.anymaze_event2_value_state != "disabled":
            self.anymaze_event2_value.insert(END,self.anymaze_event2_value_var)
        if self.anymaze_event2_column_var != 'None':
            self.anymaze_column_set_event2(event=self.anymaze_event2_column_var)
        self.anymaze_event2_operation.current(self.anymaze_event2_operation_index)

        self.anymaze_event3_colname = ttk.Combobox(self.anymaze_event_gui,values=self.anymaze_column_names)
        self.anymaze_event3_colname.grid(row=4,column=0)
        self.anymaze_event3_colname.bind("<<ComboboxSelected>>", self.anymaze_column_set_event3)
        try:
            self.anymaze_event3_column_index = self.anymaze_column_names.index(self.anymaze_event3_column_var)
            self.anymaze_event3_colname.current(self.anymaze_event3_column_index)
        except:
            self.anymaze_event3_colname.current(0)
        self.anymaze_event3_operation = ttk.Combobox(self.anymaze_event_gui,values=self.anymaze_event3_operation_list)
        self.anymaze_event3_operation.grid(row=4,column=1)
        self.anymaze_event3_operation.config(state=self.anymaze_event3_operation_state)
        try:
            self.anymaze_event3_operation_index = self.anymaze_event3_operation_list.index(self.anymaze_event3_operation_var)
            self.anymaze_event3_operation.current(self.anymaze_event3_operation_index)
        except:
            self.anymaze_event1_operation.current(0)
        self.anymaze_event3_value = tk.Entry(self.anymaze_event_gui)
        self.anymaze_event3_value.grid(row=4,column=2)
        self.anymaze_event3_value.config(state=self.anymaze_event3_value_state)
        if self.anymaze_event3_value_state != "disabled":
            self.anymaze_event3_value.insert(END,self.anymaze_event3_value_var)
        if self.anymaze_event3_column_var != 'None':
            self.anymaze_column_set_event3(event=self.anymaze_event3_column_var)
        self.anymaze_event3_operation.current(self.anymaze_event3_operation_index)

        self.anymaze_settings_label = tk.Label(self.anymaze_event_gui,text='Anymaze Settings')
        self.anymaze_settings_label.grid(row=5,column=1)

        self.anymaze_tolerance_label = tk.Label(self.anymaze_event_gui,text='Event Length Tolerance')
        self.anymaze_tolerance_label.grid(row=6,column=0)
        self.anymaze_tolerance_entry = tk.Entry(self.anymaze_event_gui)
        self.anymaze_tolerance_entry.grid(row=7,column=0)
        self.anymaze_tolerance_entry.insert(END,self.anymaze_tolerance_var)

        self.anymaze_extra_prior_label = tk.Label(self.anymaze_event_gui,text='Time Prior to Event (sec)')
        self.anymaze_extra_prior_label.grid(row=6,column=1)
        self.anymaze_extra_prior_entry = tk.Entry(self.anymaze_event_gui)
        self.anymaze_extra_prior_entry.grid(row=7,column=1)
        self.anymaze_extra_prior_entry.insert(END,self.anymaze_extra_prior_var)

        self.anymaze_extra_follow_label = tk.Label(self.anymaze_event_gui,text='Time Following Event (sec)')
        self.anymaze_extra_follow_label.grid(row=6,column=2)
        self.anymaze_extra_follow_entry = tk.Entry(self.anymaze_event_gui)
        self.anymaze_extra_follow_entry.grid(row=7,column=2)
        self.anymaze_extra_follow_entry.insert(END,self.anymaze_extra_follow_var)

        self.anymaze_event_location_label = tk.Label(self.anymaze_event_gui,text='Centering Location')
        self.anymaze_event_location_label.grid(row=8,column=0)
        self.anymaze_event_location_list = ['Event Start','Event Center','Event End']
        self.anymaze_event_location_entry = ttk.Combobox(self.anymaze_event_gui,values=self.anymaze_event_location_list)
        self.anymaze_event_location_entry.grid(row=9,column=0)
        self.anymaze_event_location_entry.current(self.anymaze_event_location_index)

        self.anymaze_event_pad_label = tk.Label(self.anymaze_event_gui,text='Centering Prior')
        self.anymaze_event_pad_label.grid(row=8,column=1)
        self.anymaze_event_pad_entry = tk.Entry(self.anymaze_event_gui)
        self.anymaze_event_pad_entry.grid(row=9,column=1)
        self.anymaze_event_pad_entry.insert(END,self.anymaze_event_pad_var)

        self.anymaze_finish_button = tk.Button(self.anymaze_event_gui,text='Finish',command=self.anymaze_event_commit)
        self.anymaze_finish_button.grid(row=10,column=1)


        
    def anymaze_column_set_event1(self,event):
        self.anymaze_event1_column_var = self.anymaze_event1_colname.get()
        if self.anymaze_event1_column_var == 'None':
            self.anymaze_event1_operation_state = 'disabled'
            self.anymaze_event1_value_state = 'disabled'
            self.anymaze_event1_operation.config(state=self.anymaze_event1_operation_state)
            self.anymaze_event1_value.config(state=self.anymaze_event1_value_state)
        else:
            self.anymaze_event1_operation_state = 'normal'
            self.anymaze_event1_operation.config(state=self.anymaze_event1_operation_state)
            anymaze_event1_options = self.anymaze_pandas.loc[:,self.anymaze_event1_column_var]
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

    def anymaze_column_set_event2(self,event):
        self.anymaze_event2_column_var = self.anymaze_event2_colname.get()
        if self.anymaze_event2_column_var == 'None':
            self.anymaze_event2_operation_state = 'disabled'
            self.anymaze_event2_value_state = 'disabled'
            self.anymaze_event2_operation.config(state=self.anymaze_event2_operation_state)
            self.anymaze_event2_value.config(state=self.anymaze_event2_value_state)
        else:
            self.anymaze_event2_operation_state = 'normal'
            self.anymaze_event2_operation.config(state=self.anymaze_event2_operation_state)
            anymaze_event2_options = self.anymaze_pandas.loc[:,self.anymaze_event2_column_var]
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

    def anymaze_column_set_event3(self,event):
        self.anymaze_event3_column_var = self.anymaze_event3_colname.get()
        if self.anymaze_event3_column_var == 'None':
            self.anymaze_event3_operation_state = 'disabled'
            self.anymaze_event3_value_state = 'disabled'
            self.anymaze_event3_operation.config(state=self.anymaze_event3_operation_state)
            self.anymaze_event3_value.config(state=self.anymaze_event3_value_state)
        else:
            self.anymaze_event3_operation_state = 'normal'
            self.anymaze_event3_operation.config(state=self.anymaze_event3_operation_state)
            anymaze_event3_options = self.anymaze_pandas.loc[:,self.anymaze_event3_column_var]
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
            if self.anymaze_event1_boolean == True:
                self.anymaze_event1_operation_index = self.anymaze_boolean_list.index(self.anymaze_event1_operation_var)
            else:
                self.anymaze_event1_operation_index = self.anymaze_operation_list.index(self.anymaze_event1_operation_var)
            self.anymaze_event1_value_var = self.anymaze_event1_value.get()

        if self.anymaze_event2_column_var == 'None':
            self.anymaze_event2_operation_var = 'None'
            self.anymaze_event2_operation_index = 0
            self.anymaze_event2_value_var = 'None'
        else:
            self.anymaze_event2_operation_var = self.anymaze_event2_operation.get()
            if self.anymaze_event2_boolean == True:
                self.anymaze_event2_operation_index = self.anymaze_boolean_list.index(self.anymaze_event2_operation_var)
            else:
                self.anymaze_event2_operation_index = self.anymaze_operation_list.index(self.anymaze_event2_operation_var)
            self.anymaze_event2_value_var = self.anymaze_event2_value.get()

        if self.anymaze_event3_column_var == 'None':
            self.anymaze_event3_operation_var = 'None'
            self.anymaze_event3_operation_index = 0
            self.anymaze_event3_value_var = 'None'
        else:
            self.anymaze_event3_operation_var = self.anymaze_event3_operation.get()
            if self.anymaze_event3_boolean == True:
                self.anymaze_event3_operation_index = self.anymaze_boolean_list.index(self.anymaze_event3_operation_var)
            else:
                self.anymaze_event3_operation_index = self.anymaze_operation_list.index(self.anymaze_event3_operation_var)
            self.anymaze_event3_value_var = self.anymaze_event3_value.get()


        self.anymaze_event_gui.destroy()

        

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
                self.photometry_object.trial_separator(normalize=True,whole_trial_normalize=True,trial_definition = True,
                                                       trial_iti_pad=float(self.abet_trial_iti_var))    
            elif self.centered_z_var.get() == 1:
                self.photometry_object.trial_separator(normalize=True,whole_trial_normalize=False, trial_definition = True,
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
            self.confirmation_text = tk.Label(self.confirmation_window,text='Files have been generated')
            self.confirmation_text.grid(row=0,column=0)

            self.confirmation_button = tk.Button(self.confirmation_window,text='Continue',command=self.close_confirmation)
            self.confirmation_button.grid(row=1,column=0)
            
        if self.anymaze_file_path != '' and self.doric_file_path != '':
            self.photometry_object = Photometry_Data()
            self.photometry_object.load_doric_data(self.doric_file_path,self.channel_control_index,self.channel_active_index,self.channel_ttl_index)
            self.photometry_object.load_anymaze_data(self.anymaze_file_path)

            self.photometry_object.anymaze_search_event_OR(event1_name=self.anymaze_event1_column_var,event1_operation=self.anymaze_event1_operation_var,event1_value=self.anymaze_event1_value_var,
                                                           event2_name=self.anymaze_event2_column_var,event2_operation=self.anymaze_event2_operation_var,event2_value=self.anymaze_event2_value_var,
                                                           event3_name=self.anymaze_event3_column_var,event3_operation=self.anymaze_event3_operation_var,event3_value=self.anymaze_event3_value_var,
                                                           event_tolerance = float(self.anymaze_tolerance_var),extra_prior_time=float(self.anymaze_extra_prior_var),
                                                           extra_follow_time=float(self.anymaze_extra_follow_var))

            self.photometry_object.anymaze_doric_synchronize_OR()


            self.photometry_object.doric_process(int(self.low_pass_var))
            
            
            if self.centered_z_var.get() == 1:
                self.photometry_object.trial_separator(normalize=True,whole_trial_normalize=True,trial_definition = False)
            elif self.centered_z_var.get() == 0:
                self.photometry_object.trial_separator(normalize=True,whole_trial_normalize=False, trial_definition = False,
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
            self.confirmation_text = tk.Label(self.confirmation_window,text='Files have been generated')
            self.confirmation_text.grid(row=0,column=0)

            self.confirmation_button = tk.Button(self.confirmation_window,text='Continue',command=self.close_confirmation)
            self.confirmation_button.grid(row=1,column=0)
            

    def close_confirmation(self):
        self.confirmation_window.destroy()
        

        

    

        
        
        

Main_GUI = Photometry_GUI()
