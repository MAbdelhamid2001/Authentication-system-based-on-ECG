
import tkinter as tk
from tkinter import messagebox,ttk,filedialog
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
from help_functions import *
import warnings
warnings.filterwarnings('ignore')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)


def exe_GUI():

    top=tk.Tk()
    top.geometry('500x700')
    top.minsize(500,700) 
    top.maxsize(500,700)
    top.title('Authentication system GUI')
    top['background']='#856ff8'
    #opening file 
    #function definition for opening file dialog

    def openf():
        global file
        global to_test

        
        file = filedialog.askopenfilename(initialdir=r'C:\Users\mhmed\HCI lab codes\hci Project', title="select file",
                                         filetypes=(("Txt Files",".txt"), ("All Files","*.*")))
        #inserting data to text editor
        to_test=np.loadtxt(file)
        msg=messagebox.showinfo('welcome','File inserted successfully')
        f_name=re.findall('sub_\w',file)

        teditor.delete('1.0', tk.END)
        teditor.insert(tk.END,f_name)
        # print(data)
        

    #creating text editor
    teditor = tk.Text(top, width=16, height=2)
    teditor.place(x=200,y=50)
    # teditor.pack(pady=10)
    file_open = tk.Button(top, text="Open file", command= openf,width=12,height=2)
    file_open.pack()
    file_open.place(x=90,y=50)

    ###f2
    f2_l=tk.Label(top,text="Preprocessing",width=12,height=2)
    f2_l.place(x=90,y=100)


    f2_var=tk.StringVar()
    f2=ttk.Combobox(top,width=18,height=5,textvariable=f2_var)
    f2['values']=('Wavelet','Fiducial','AC/DCT')
    f2.current(0)
    f2.place(x=200,y=100)

    def prep_type(type_):
        if type_==1:
            return 'wavelet'
        elif type_==2:
            return 'fiducial_features'
        else:
            return 'AC/DCT'
    def get_sub(idx):
        return f'sub_{idx+1}'



    def retrieve_f2():
        f2_v=f2.get()
        print(f2_v)
        return f2_v
    Button = tk.Button(top, text = "Submit", command = retrieve_f2,width=8,height=2)
    Button.place(x=350,y=100)


    ##Run
    def plot(signal,test,type_):
        fig=Figure(figsize=(4,4),dpi=100)
        ax1=fig.add_subplot(2,1,1)
        ax1.set_title('signal',fontdict={'fontsize':12})
        ax1.plot(signal)

        ax2=fig.add_subplot(2,1,2)
        if type_==2:
            ax2.set_title(f'After preprocessing using {prep_type(type_)}',fontdict={'fontsize':12})
            ax2.plot(test[0])

        else:
            
            ax2.set_title(f'After preprocessing using {prep_type(type_)}',fontdict={'fontsize':12})
            ax2.plot(test)

        fig.tight_layout()

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig,
                                master = top)  
        # canvas.draw()
    
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().place(x=50,y=300)
    

    def callback_Run():
        msg=messagebox.showinfo('Running!','Wait Please ')
        print("Code is Running")

        Output.delete('1.0', tk.END)
################################
        f2_v= retrieve_f2()
        type_=f2_v

        if type_=='Wavelet':
            type_=1
        elif type_=='Fiducial':
            type_=2
        else:
            type_=3


        actual =re.findall('sub_\w',file)
        print("actual",actual[0])
        print('test data',to_test)

        test=preprocessing(to_test,type_)
        test=np.array(test)
        print('test shape',test.shape)
        plot(to_test,test,type_)
#######################

        # ('Wavelet','Fiducial','AC/DCT')
        if type_==1:
            loaded_model=pickle.load(open('LDA_classifier_wavelet.pkl','rb'))
        # fiducial 
        elif type_==2:
            loaded_model=pickle.load(open('LDA_classifier_fiducial.pkl','rb'))
        else:
            loaded_model=pickle.load(open('LDA_classifier_ACDCT.pkl','rb'))


        if type_ !=2:
            test=np.expand_dims(test,axis=0)
        print('test shape after ',test.shape)

        pred=loaded_model.predict(test)


        if actual[0] == get_sub(pred[0]):
            res='Subject identified --> Access Allowed'
        else:
            res='Subject is unidentified --> Access Denied'


        Output.insert(tk.END,res)
        print("Code is Running")

        

    B=tk.Button(top,text='Predict',height=2,width=10,command=callback_Run)
    B.place(x=225,y=150)

    Output = tk.Text(top, height = 2,
                  width = 35,
                  bg = "light cyan")
    Output.place(x=120,y=200)


    top.mainloop()

# In[42]:


def main():
    exe_GUI()

if __name__=='__main__':
    main()

