#  xds_gui.py
#
#  GUI wrapper around the cage_data class defined in cage_data.py
#
#  This is used to bring data into XDS type data containers.



import tkinter as tk
from tkinter import ttk
from cage_data import cage_data  # needs to be in the same folder


# define the class holding all of the information about the window etc


class xds_gui(tk.Frame):
    """XDS conversion GUI class

    requires a tkinter root as an input
    """
    def __init__(self, parent):
        super().__init__(parent) # refer to Python docs for info about class inheretance
        #self.pack()
        self.grid()

        # .nev loading
        self.nev_check = ttk.Checkbutton(self).












# initialize the window

root = tk.Tk() # root tk 
frm = ttk.Frame(root, padding=15)
frm.grid()

# .nev loading

ttk.Label(frm, text='Hello World!').grid(column=0, row=0)
ttk.Button(frm, text='Quit', command=root.destroy).grid(column=1, row=0)



# run the whole thing
root.mainloop()

