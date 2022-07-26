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
        self.file_frame = tk.LabelFrame(self, text='File Selection')
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
        
        
        
        self.file_frame.grid(row=0, column=0, padx=5, pady=5)


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




        self.convert_button = ttk.Button(self, text='Convert', command=self.convert_xds)
        self.convert_button.grid(row=2, column=0)




    def fn_update(self, box):
        if box == 'nev':
            self.nev_fn.set(fd.askopenfilename(filetypes=[('Neural Event','*.nev')]))
        elif box == 'rhd':
            self.emg_fn.set(fd.askopenfilename(filetypes=[('Intan Digital', '*.rhd')]))
        elif box == 'mot':
            self.mot_fn.set(fd.askopenfilename(filetypes=[('OpenSim or SIMM files', '*.mot')]))

    def convert_xds(self):
        nev_fn = self.nev_fn.get() # get the stored nev filename
        emg_fn = self.emg_fn.get()
        mot_fn = self.mot_fn.get() # not implemented in cage_data.py yet. will have to finish updating

        # split out the path name
        nev_path, nev_fn = path.split(nev_fn)
        emg_path, emg_fn = path.split(emg_fn)

        # check emg and nev path locations
        if nev_path!= emg_path:
            print('The nev and EMG paths are different. XDS converter will not handle that well...')

        self.xds.create(nev_path, nev_file=nev_fn, rhd_file=emg_fn)




# initialize the window
if __name__ == "__main__":
    root = tk.Tk() # root tk 
    root.geometry("500x250")
    frm = xds_gui(root) # create the base xds window

    # run the whole thing
    root.mainloop()


