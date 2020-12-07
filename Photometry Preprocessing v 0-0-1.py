## Import ## - Packages you need numpy, pandas, scipy
### to install in cmd-
# python -m pip install --upgrade pip wheel setuptools
# python -m pip install numpy pandas scipy
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import END
import numpy as np
import pandas as pd
import csv
from scipy import signal

def open_file():
    filepath = filedialog.askopenfilename(title="Select Doric Photometry File",filetypes=(('csv files','*.csv'),('all files','*.')))
    doric_file = open(filepath)
    doric_csv_reader = csv.reader(doric_file)
    first_row_read = False
    second_row_read = False
    doric_name_list = list()
    for row in doric_csv_reader:
        if first_row_read == False:
            first_row_read = True
            continue
        if second_row_read == False and first_row_read == True:
            doric_name_list = row
            break
    doric_file.close()
    
    col_list = doric_name_list
    ch_ctrl_box['values'] = col_list
    ch_ctrl_box.config(state="normal")
    ch_act_box['values'] = col_list
    ch_act_box.config(state="normal")
    filter_entry.config(state="normal")
    filepath_text.insert(END,str(filepath))
    
def launch_process():
    filepath = str(filepath_text.get())
    ctrl_index = int(ch_ctrl_box.current())
    act_index = int(ch_act_box.current())
    try:
        filter_freq = int(filter_entry.get())
    except:
        return
    
    process_photometry_data(filepath, ctrl_index, act_index, filter_freq)

def process_photometry_data(filepath,ch1_col,ch2_col,filter_frequency):
            ## Open file in csv module
    doric_file_path = filepath # Filepath
    doric_file = open(doric_file_path)
    doric_csv_reader = csv.reader(doric_file) # CSV Reader
    
    ## Finds column names and parses data into separate storage
    first_row_read = False
    second_row_read = False
    doric_name_list = list()
    doric_list = list()
    for row in doric_csv_reader:
        if first_row_read == False:
            first_row_read = True
            continue
        if second_row_read == False and first_row_read == True:
            doric_name_list = [row[0],row[ch1_col],row[ch2_col]]
            second_row_read = True
            continue
        else:
            doric_list.append([row[0],row[ch1_col],row[ch2_col]])
    doric_file.close()
    
    
    ## Create Numpy and Pandas format
    doric_numpy = np.array(doric_list)
    doric_pandas = pd.DataFrame(data=doric_numpy,columns=doric_name_list)
    doric_pandas.columns = ['Time','Control','Active']
    doric_pandas = doric_pandas.astype('float')
    
    
    # Convert Numeric values to a numpy array
    time_data = doric_pandas['Time'].to_numpy()
    f0_data = doric_pandas['Control'].to_numpy()
    f_data = doric_pandas['Active'].to_numpy()
    
    
    # Convert data to a 64 bit float value
    time_data = time_data.astype(float)
    f0_data = f0_data.astype(float)
    f_data = f_data.astype(float)
    
    # Identify the sample frequency based on time
    sample_frequency = len(time_data) / time_data[(len(time_data) - 1)]
    
    ## Low Pass Butterworth Filter
    butter_filter = signal.butter(N=2,Wn=filter_frequency,
                                   btype='lowpass',analog=False,
                                   output='sos',fs=sample_frequency)
    filtered_f0 = signal.sosfilt(butter_filter,f0_data)
    filtered_f = signal.sosfilt(butter_filter,f_data)
    
    ## Least Mean Squares Regression
    f0_a_data = np.vstack([filtered_f0,np.ones(len(filtered_f0))]).T
    m,c = np.linalg.lstsq(f0_a_data,filtered_f,rcond=None)[0]
    f0_fitted = (filtered_f0.astype(np.float) * m) + c
    
    ## Create Delta F
    delta_f = (filtered_f.astype(float) - f0_fitted.astype(float)) / f0_fitted.astype(float)
    
    # Add Delta F data to Pandas format
    doric_pd = pd.DataFrame(time_data)
    doric_pd['DeltaF'] = delta_f
    doric_pd = doric_pd.rename(columns={0:'Time',1:'DeltaF'})
    
    save_path = filedialog.asksaveasfilename(defaultextension='.csv')
    doric_pd.to_csv(save_path,index=False)
    

col_list = ['']
root = tk.Tk()

menu_title = tk.Label(root,text="Doric Photometry Pre-Processor")
menu_title.grid(row=0,column=1)


filepath_label = tk.Label(root,text="Filepath:")
filepath_label.grid(row=1,column=0)

filepath_text = tk.Entry(root)
filepath_text.grid(row=1,column=1)

filepath_button = tk.Button(root,text="...",command=open_file)
filepath_button.grid(row=1,column=2)

ch_ctrl_label = tk.Label(root,text="Control Channel Column:")
ch_ctrl_label.grid(row=2,column=0)

ch_ctrl_box = ttk.Combobox(root,values=col_list)
ch_ctrl_box.grid(row=2,column=2)
ch_ctrl_box.config(state='disabled')

ch_act_label = tk.Label(root,text="Active Channel Column:")
ch_act_label.grid(row=3,column=0)

ch_act_box = ttk.Combobox(root,values=col_list)
ch_act_box.grid(row=3,column=2)
ch_act_box.config(state='disabled')

filter_label = tk.Label(root,text="Low Pass Filter Value (hz):")
filter_label.grid(row=4,column=0)

filter_entry = tk.Entry(root)
filter_entry.grid(row=4,column=2)
filter_entry.config(state='disabled')

process_button = tk.Button(root,text="Process",command=launch_process)
process_button.grid(row=5,column=1)

root.mainloop()
