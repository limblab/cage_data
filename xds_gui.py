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


# define the class holding all of the information about the window etc


class xds_gui(tk.Frame):
    """XDS conversion GUI class

    requires a tkinter root as an input
    """
    def __init__(self, parent):
        super().__init__(parent) # refer to Python docs for info about class inheretance
        self.grid()

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

        # # binning data
        # self.bin_frame = tk.LabelFrame(self, text='Bin Data')
        


        # self.bin_frame.grid(row=1, column=0, padx=5, pady=5, columnspan=2)

        # plot data
        self.plot_frame = tk.LabelFrame(self, text='Plotting')
        
        self.raw_plot_button = ttk.Button(self.plot_frame, text='Plot Raw', state='disabled')
        self.raw_plot_button.grid(row=0, column=0, padx = 5, pady = 5)
        self.bin_plot_button = ttk.Button(self.plot_frame, text='Plot Binned', state='disabled')
        self.bin_plot_button.grid(row=0, column=1, padx = 5, pady = 5)


        self.plot_frame.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

        # # --------------------------------------------------------------
        # # metadata -- monkey name, task name
        # self.meta_frame = tk.LabelFrame(self, text='Meta Data')
        # self.monkey = tk.StringVar()
        # self.monkey_menu = tk.Menubutton(self.meta_frame, text='Monkey Name')
        # self.monkey_menu.grid()
        # self.monkey_menu.menu = tk.Menu(self.meta_frame)
        # self.monkey_menu.menu.add
        # # self.monkey_options = tk.Menu(self.meta_frame).grid()
        # # self.monkey_options.add_command(label='Pop', command=lambda:self.monkey.set('Pop'))
        # # self.monkey_menu = tk.Menubutton(self.meta_frame, text='Monkey Name', menu=self.monkey_options).grid(row=0, column=0, padx=10, pady=10)

        # self.meta_frame.grid(row=1, column=0, padx=5, pady=5)



        self.save_button = ttk.Button(self, text='Save', command=self.save_xds, state='disabled')
        self.save_button.grid(row=3, column=1, padx=5, pady=5)



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
        self.save_button.state('normal')
        



    def save_xds(self):
        # just a wrapper for the save_to_pickle function
        nev_fn = self.nev_fn.get() # get the stored nev filename
        nev_path, nev_fn = path.split(nev_fn)
        nev_fn, _ = path.splitext(nev_fn)


        save_fn = fd.asksaveasfilename(master=self,\
            confirmoverwrite=True, filetypes=(['Pickle','.pkl']),\
            initialdir=nev_path, initialfile= nev_fn.join('.pkl') )
        
        self.xds.save_to_pickle()





# initialize the window
if __name__ == "__main__":
    root = tk.Tk() # root tk 
    root.geometry("500x250")
    frm = xds_gui(root) # create the base xds window

    # run the whole thing
    root.mainloop()


