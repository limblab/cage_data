#  xds_gui.py
#
#  GUI wrapper around the cage_data class defined in cage_data.py
#
#  This is used to bring data into XDS type data containers.


from logging import warning
from os import path
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from cage_data import cage_data  # needs to be in the same folder
import numpy as np
from scipy.stats import mode
from matplotlib import pyplot as plt


# define the class holding all of the information about the window etc


class xds_gui(tk.Frame):
    """XDS conversion GUI class

    requires a tkinter root as an input
    """
    def __init__(self, parent):
        super().__init__(parent) # refer to Python docs for info about class inheretance

        # initialize the xds
        self.xds = cage_data()


        # --------------------------------------------------------------
        # file selection -- will try to autopopulate metadata below
        self.file_frame = tk.LabelFrame(self, text='File Conversion')
        # .nev loading
        # this will also look for associated .nsx files
        self.nev_fn = tk.StringVar()
        self.nev_entry = ttk.Entry(self.file_frame, textvariable=self.nev_fn)
        self.nev_entry.grid(row=0, column=0, padx=30, pady=5, ipadx=100)
        self.nev_select = ttk.Button(self.file_frame, text='Select .nev', command=lambda:self.fn_update('nev'))
        self.nev_select.grid(row=0, column=1, padx=5, pady=5)

        # do we want force, EMG, etc
        self.emg_fn = tk.StringVar()
        self.emg_entry = ttk.Entry(self.file_frame, textvariable=self.emg_fn)
        self.emg_entry.grid(row=1, column=0, padx=30, pady=5, ipadx=100)
        self.emg_select = ttk.Button(self.file_frame, text='Select .rhd', command=lambda:self.fn_update('rhd'))
        self.emg_select.grid(row=1, column=1, padx=5, pady=5)

        # .mot files
        self.mot_fn = tk.StringVar()
        self.mot_entry = ttk.Entry(self.file_frame, textvariable=self.mot_fn)
        self.mot_entry.grid(row=2, column=0, padx=30, pady=5, ipadx=100)
        self.mot_select = ttk.Button(self.file_frame, text='Select .mot', command=lambda:self.fn_update('mot'))
        self.mot_select.grid(row=2, column=1, padx=5, pady=5)
        
        self.convert_button = ttk.Button(self.file_frame, text='Convert', command=self.convert_xds)
        self.convert_button.grid(row=3, column=0, padx=5, pady=5)
        
        self.file_frame.grid(row=0, column=0, padx=5, pady=5, columnspan=2)

        # binning data
        self.bin_frame = tk.LabelFrame(self, text='Bin Data')
        self.bin_label = ttk.Label(self.bin_frame, text='Bin Size (ms)')
        self.bin_label.grid(row=0, column=0, pady=5, padx=5)
        self.bin_size = tk.StringVar()
        self.bin_size_txt = ttk.Entry(self.bin_frame, state='disabled', textvariable=self.bin_size)
        self.bin_size_txt.grid(row=0, column=1, ipadx = 10, pady=5, padx=5)
        self.bin_button = ttk.Button(self.bin_frame, text='Bin Data', state='disabled', command=self.bin_data)
        self.bin_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.bin_frame.grid(row=1, column=0, padx=5, pady=5, columnspan=2)

        # plot data
        self.plot_frame = tk.LabelFrame(self, text='Plotting')
        
        self.raw_plot_button = ttk.Button(self.plot_frame, text='Plot Raw', state='disabled')
        self.raw_plot_button.grid(row=0, column=0, padx = 5, pady = 5)
        self.bin_plot_button = ttk.Button(self.plot_frame, text='Plot Binned', state='disabled')
        self.bin_plot_button.grid(row=0, column=1, padx = 5, pady = 5)


        self.plot_frame.grid(row=2, column=0, padx=5, pady=5, columnspan=2)


        self.save_button = ttk.Button(self, text='Save', command=self.save_xds, state='disabled')
        self.save_button.grid(row=3, column=1, padx=5, pady=5)

        self.grid()



    def fn_update(self, box):
        if box == 'nev':
            self.nev_fn.set(fd.askopenfilename(filetypes=[('Neural Event','*.nev')]))
        elif box == 'rhd':
            self.emg_fn.set(fd.askopenfilename(filetypes=[('Intan Digital', '*.rhd')]))
        elif box == 'mot':
            self.mot_fn.set(fd.askopenfilename(filetypes=[('OpenSim or SIMM files', '*.mot')]))

    def convert_xds(self):
        # get any of the stored filenames
        nev_fn = self.nev_fn.get() # get the stored nev filename
        emg_fn = self.emg_fn.get()
        mot_fn = self.mot_fn.get()

        self.xds.create(nev_fn, rhd_file=emg_fn, mot_file=mot_fn)

        # if it has .mot files, that will determine the bin size
        if self.xds.has_mot:
            self.bin_size.set(np.round(mode(np.diff(self.xds.mot_timestamps))[0][0] * 1000))
            self.bin_button['state'] = 'normal'
        else:  # otherwise allow the user to choose
            self.bin_size_txt['state'] = 'normal'
            self.bin_size.set(str(50)) # default 50 ms
            self.bin_button['state'] = 'normal'

        
        self.save_button['state'] = 'normal'
        

    def bin_data(self): # just runs the cage_data.bin_data function, turns off some stuff
        bin_sz = float(self.bin_size.get())  # bin size for the conversion process
        self.xds.bin_data(bin_size=bin_sz)

        self.bin_plot_button['state'] = 'normal' # turn on the plot buttons

    def plot_binned(self):
        fig_bin,fig_ax = plt.subplots() # plotting




    def save_xds(self):
        # just a wrapper for the save_to_pickle function
        nev_fn = self.nev_fn.get() # get the stored nev filename
        nev_path, nev_fn = path.split(nev_fn)
        nev_fn, _ = path.splitext(nev_fn)
        print(nev_fn)


        save_fn = fd.asksaveasfilename(master=self,\
            confirmoverwrite=True, filetypes=[('Pickle','.pkl')],\
            initialdir=nev_path, initialfile= nev_fn )
        save_path, save_fn = path.split(save_fn)
        self.xds.save_to_pickle(save_path, save_fn)





# initialize the window
if __name__ == "__main__":
    root = tk.Tk() # root tk 
    # root.geometry("500x250")
    frm = xds_gui(root) # create the base xds window

    # run the whole thing
    root.mainloop()


