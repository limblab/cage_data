#  cage_data_gui.py
#
#  GUI wrapper around the cage_data class defined in cage_data.py
#
#  This is used to bring data into cage_data type data containers.


from os import path
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from cage_data import cage_data  # needs to be in the same folder
import numpy as np
from scipy.stats import mode
from matplotlib import pyplot as plt
import pickle


# define the class holding all of the information about the window etc


class cage_data_gui(tk.Frame):
    """cage_data conversion GUI class

    requires a tkinter root as an input
    """
    def __init__(self, parent):
        super().__init__(parent) # refer to Python docs for info about class inheretance



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
        
        self.convert_button = ttk.Button(self.file_frame, text='Convert', command=self.convert_cage_data)
        self.convert_button.grid(row=3, column=0, padx=5, pady=5)
        
        self.file_frame.grid(row=0, column=0, padx=5, pady=5, columnspan=2)


        # --------------------------------------------------------------
        # open existing cage_data
        self.exist_frame = tk.LabelFrame(self, text='Open Pickled Data File')
        self.exist_fn = tk.StringVar()
        self.exist_entry = ttk.Entry(self.exist_frame, textvariable=self.exist_fn)
        self.exist_entry.grid(row=0, column=0, padx=30, pady=5, ipadx=100)
        self.exist_browse = ttk.Button(self.exist_frame, text='Browse Existing', command=lambda:self.fn_update('pkl'))
        self.exist_browse.grid(row=0, column=1, padx=5, pady=5)
        self.exist_open = ttk.Button(self.exist_frame, text='Open', command=self.open_existing)
        self.exist_open.grid(row=1, column=0, padx=5, pady=5)
        
        self.exist_frame.grid(row=1, column=0, padx=5, pady=5, columnspan=2)


        # --------------------------------------------------------------
        # binning data
        self.bin_frame = tk.LabelFrame(self, text='Bin Data')
        self.bin_label = ttk.Label(self.bin_frame, text='Bin Size (ms)')
        self.bin_label.grid(row=0, column=0, pady=5, padx=5)
        self.bin_size = tk.StringVar()
        self.bin_size_txt = ttk.Entry(self.bin_frame, state='disabled', textvariable=self.bin_size)
        self.bin_size_txt.grid(row=0, column=1, ipadx = 10, pady=5, padx=5)
        self.bin_button = ttk.Button(self.bin_frame, text='Bin Data', state='disabled', command=self.bin_data)
        self.bin_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.bin_frame.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

        # --------------------------------------------------------------
        # plot data
        self.plot_frame = tk.LabelFrame(self, text='Plotting')
        
        self.raw_plot_button = ttk.Button(self.plot_frame, text='Plot Raw', state='disabled')
        self.raw_plot_button.grid(row=0, column=0, padx = 5, pady = 5)
        self.bin_plot_button = ttk.Button(self.plot_frame, text='Plot Binned', state='disabled')
        self.bin_plot_button.grid(row=0, column=1, padx = 5, pady = 5)

        self.plot_frame.grid(row=3, column=0, padx=5, pady=5, columnspan=2)

        # --------------------------------------------------------------
        # build decoder
        self.predict_frame = tk.LabelFrame(self, text='Decoders')
        
        self.decoder_predict_button = ttk.Button(self.predict_frame, text='Build Decoder', state='disabled', command=self.build_decoder)
        self.decoder_predict_button.grid(row=0, column=0, padx = 5, pady = 5)
        self.xval_predict_button = ttk.Button(self.predict_frame, text='k-fold Xval', state='disabled')
        self.xval_predict_button.grid(row=0, column=1, padx = 5, pady = 5)

        self.predict_frame.grid(row=3, column=0, padx=5, pady=5, columnspan=2)
        
        # --------------------------------------------------------------
        self.save_button = ttk.Button(self, text='Save', command=self.save_cage_data, state='disabled')
        self.save_button.grid(row=4, column=1, padx=5, pady=5)
        self.clear_button = ttk.Button(self, text='Clear', command=self.reinit_cage_data)
        self.clear_button.grid(row=4, column=0, padx=5, pady=5)

        self.grid()


    # -------------------------
    def fn_update(self, box):
        if box == 'nev':
            self.nev_fn.set(fd.askopenfilename(filetypes=[('Neural Event','*.nev')]))
        elif box == 'rhd':
            self.emg_fn.set(fd.askopenfilename(filetypes=[('Intan Digital', '*.rhd')]))
        elif box == 'mot':
            self.mot_fn.set(fd.askopenfilename(filetypes=[('OpenSim or SIMM files', '*.mot')]))
        elif box == 'pkl':
            self.exist_fn.set(fd.askopenfilename(filetypes=[('Cage Data pickle', '*.pkl')]))


    # -------------------------
    def convert_cage_data(self):
        # get any of the stored filenames
        nev_fn = self.nev_fn.get() # get the stored nev filename
        emg_fn = self.emg_fn.get()
        mot_fn = self.mot_fn.get()

        # initialize the cage_data
        self.cage_data = cage_data()
        self.cage_data.create(nev_fn, rhd_file=emg_fn, mot_file=mot_fn)

        # deactivate everything -- we don't want to confuse which files we're working with
        self.nev_entry['state'] = 'disabled'
        self.nev_select['state'] = 'disabled'
        self.emg_entry['state'] = 'disabled'
        self.emg_select['state'] = 'disabled'
        self.mot_entry['state'] = 'disabled'
        self.mot_select['state'] = 'disabled'
        self.exist_browse['state'] = 'disabled'
        self.exist_entry['state'] = 'disabled'
        self.exist_open['state'] = 'disabled'
        self.convert_button['state'] = 'disabled'



        # if it has .mot files, that will determine the bin size
        if self.cage_data.has_mot:
            self.bin_size.set(np.round(mode(np.diff(self.cage_data.mot_timestamps))[0][0] * 1000))
            self.bin_button['state'] = 'normal'
        else:  # otherwise allow the user to choose
            self.bin_size_txt['state'] = 'normal'
            self.bin_size.set(str(50)) # default 50 ms
            self.bin_button['state'] = 'normal'

        
        self.save_button['state'] = 'normal'


    # -------------------------
    # open existing cage_data
    def open_existing(self):
        with open(self.exist_fn.get(), 'rb') as file:
            self.cage_data = pickle.load(file)

        # disable all of the opening methods -- don't want to confuse anything
        self.nev_entry['state'] = 'disabled'
        self.nev_select['state'] = 'disabled'
        self.emg_entry['state'] = 'disabled'
        self.emg_select['state'] = 'disabled'
        self.mot_entry['state'] = 'disabled'
        self.mot_select['state'] = 'disabled'
        self.exist_browse['state'] = 'disabled'
        self.exist_entry['state'] = 'disabled'
        self.exist_open['state'] = 'disabled'
        self.convert_button['state'] = 'disabled'

    # -------------------------
    def bin_data(self): # just runs the cage_data.bin_data function, turns off some stuff
        bin_sz = float(self.bin_size.get())/1000  # bin size for the conversion process
        self.cage_data.bin_data(bin_size=bin_sz)

        # enable buttons
        self.bin_plot_button['state'] = 'normal' # turn on the plot buttons
        self.decoder_predict_button['state'] = 'normal'
        self.xval_predict_button['state'] = 'normal'



    # -------------------------
    def plot_binned(self):
        '''
        Plot binned data -- will check to see what binned data we have,
        then create a standalone plot
        
        '''
        binned_list = self.cage_data.binned.keys()

        if not binned_list:
            self.warn = tk.messagebox.showwarning(master = self, title='No Binned Data', message='No binned data found in cage_data structure')

        # non-neuronal stuff
        other_binned = [item for item in binned_list if item not in ['spikes','timeframe']]

        # going to have at least spikes.
        # Potentially FSRs, EMGs, etc.
        bin_select = tk.Toplevel(self)
        neuron_box = tk.LabelFrame(bin_select, text = 'Neurons')
        # for 
        # neuron_text.



        # fig_bin,fig_ax = plt.subplots() # plotting

    # ------------------------
    def build_decoder(self):
        '''
        properly arranges the inputs for cage_data's decoder building methods.

        Currently will just plot the entirety of whichever signal type is chosen,
        but in the future should allow you to chose a subset of the signals
        based on their names.

        The method returns the decoders and the vafs, so we'll go ahead and 
        plot the VAFs in a nice way from here.
        '''

        options_window = tk.Toplevel(self) # pop up to chose the predicted signals

        # which data do you want to predict?
        signal_box = tk.LabelFrame(options_window)
        binned_list = [keys for keys in self.cage_data.binned.keys() if keys not in ['spikes','timeframe']]
        predict_radio = {}
        predict_selection = tk.Variable()
        row_locn = 0
        for binned in binned_list: # make a checkbox for each option
            predict_radio[binned] = ttk.Radiobutton(signal_box, text=binned, variable=predict_selection, value=binned)
            predict_radio[binned].grid(row=row_locn, column=0, pady=5)
            row_locn += 1 # for the next option
        signal_box.grid(row=0, column=0, padx=10, pady=10)

        # nonlinearity
        nl_box = tk.LabelFrame(options_window)
        nl_var = tk.Variable()
        nl_var.set(None)
        nl_none = ttk.Radiobutton(nl_box, text="None", variable=nl_var, value=None)
        nl_none.grid(row=0,column=0, padx=5, pady=5)
        nl_poly = ttk.Radiobutton(nl_box, text="Polynomial", variable=nl_var, value='poly')
        nl_poly.grid(row=0, column=1, padx=5, pady=5)
        nl_exp = ttk.Radiobutton(nl_box, text="Exponential", variable=nl_var, value='exp')
        nl_exp.grid(row=0, column=2, padx=5, pady=5)

        nl_box.grid(row=1, column=0, padx=10, pady=10)

        # num lags        
        lag_var = tk.Variable()
        lag_var.set('5')
        lag_entry = ttk.Entry(options_window, textvariable=lag_var)
        lag_entry.grid(row=2, column=0, pady=5)



        predict_button = ttk.Button(options_window, text='Build Decoder',\
            command=self.cage_data.filter_builder(out_type=predict_selection.get(),\
                n_lags=int(lag_entry.get()), nonlinearity=nl_var.get()))
        predict_button.grid(row=3, column=0, padx=5, pady=5)
        
        
            







    # -------------------------
    def save_cage_data(self):
        # just a wrapper for the save_to_pickle function
        nev_fn = self.nev_fn.get() # get the stored nev filename
        nev_path, nev_fn = path.split(nev_fn)
        nev_fn, _ = path.splitext(nev_fn)
        print(nev_fn)


        save_fn = fd.asksaveasfilename(master=self,\
            confirmoverwrite=True, filetypes=[('Pickle','.pkl')],\
            initialdir=nev_path, initialfile= nev_fn )
        save_path, save_fn = path.split(save_fn)
        self.cage_data.save_to_pickle(save_path, save_fn)

    
    # -------------------------
    def reinit_cage_data(self):
        # reinitialize file selection
        self.nev_fn.set('')
        self.nev_select['state'] = 'normal'
        self.emg_fn.set('')
        self.emg_select['state'] = 'normal'
        self.mot_fn.set('')
        self.mot_select['state'] = 'normal'
        self.convert_button['state'] = 'normal'
        self.exist_browse['state'] = 'normal'
        self.exist_entry['state'] = 'normal'
        self.exist_open['state'] = 'normal'
        
        # binning data
        self.bin_size.set('')
        self.bin_size_txt['state'] = 'disabled'
        self.bin_button['state'] = 'disabled'

        # plot data
        self.raw_plot_button['state'] = 'disabled'
        self.bin_plot_button['state'] = 'disabled'

        self.save_button['state'] = 'disabled'

        self.grid()


# ------------------------------------------
# initialize the window
if __name__ == "__main__":
    root = tk.Tk() # root tk 
    frm = cage_data_gui(root) # create the base cage_data window

    # run the whole thing
    root.mainloop()


