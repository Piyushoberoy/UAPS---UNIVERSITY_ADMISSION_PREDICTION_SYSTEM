# --------------------------------------------------------------------LAYOUT MODULES------------------------------------------------------------------------
import tkinter as tk
from PIL import ImageTk, Image
import joblib
# ------------------------------------------------------------PREDICTION & ANALYSATION MODULES--------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

path = r'K:\PROGRAMS\PYTHON\UAPS\UAPS---UNIVERSITY_ADMISSION_PREDICTION_SYSTEM\\'
model1 = joblib.load(path+'UAPS1.pkl')
model2 = joblib.load(path+'UAPS2.pkl')

class features:
    def __init__(self, frame):
        self.main_frame = frame
        self.frame = tk.Frame(self.main_frame)
        self.frame.place(relx=.5, rely=.5, anchor=tk.CENTER)

        self.sub_frame = tk.Frame(self.main_frame)
        self.sub_frame.place(relx=.5, rely=.5, anchor=tk.CENTER)

        self.font_style = "Gamorand"
        self.font_size = "18"
        self.bg_color = "white"
        self.fg_color = "black"

        self._gre_label = tk.Label(self.frame, text="GRE Score")
        self._gre = tk.DoubleVar()
        self.gre = tk.Entry(self.frame, textvariable=self._gre, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)
        self.gre.bind('<FocusOut>', self.validate_gre)

        self._toefl_label = tk.Label(self.frame, text="TOEFL Score")
        self._toefl = tk.DoubleVar()
        self.toefl = tk.Entry(self.frame, textvariable=self._toefl, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)
        self.toefl.bind('<FocusOut>', self.validate_toefl)

        self._u_rating_label = tk.Label(self.frame, text="University Rating")
        self._u_rating = tk.DoubleVar()
        self.u_rating = tk.Entry(self.frame, textvariable=self._u_rating, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)
        self.u_rating.bind('<FocusOut>', self.validate_u_rating)

        self._sop_label = tk.Label(self.frame, text="Strength of SOP")
        self._sop = tk.DoubleVar()
        self.sop = tk.Entry(self.frame, textvariable=self._sop, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)
        self.sop.bind('<FocusOut>', self.validate_sop)

        self._lor_label = tk.Label(self.frame, text="Strength of LOR")
        self._lor = tk.DoubleVar()
        self.lor = tk.Entry(self.frame, textvariable=self._lor, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)
        self.lor.bind('<FocusOut>', self.validate_lor)

        self._cgpa_label = tk.Label(self.frame, text="CGPA")
        self._cgpa = tk.DoubleVar()
        self.cgpa = tk.Entry(self.frame, textvariable=self._cgpa, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)
        self.cgpa.bind('<FocusOut>', self.validate_cgpa)

        self._research_label = tk.Label(self.frame, text="Research Experience")
        self._research = tk.IntVar()
        self.research = tk.Entry(self.frame, textvariable=self._research, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)
        self.research.bind('<FocusOut>', self.validate_research)

        self.check = tk.Button(self.frame, text="Check", command=self.show_result_form1, font=(
            self.font_style, self.font_size))
        self.check.grid(row=6, column=1, columnspan=4)
        self.home_btn = tk.Button(self.frame, text="Home", command=self.home, font=(
            self.font_style, self.font_size))
        self.home_btn.grid(row=7, column=1, columnspan=4)
    
    # --------------------------------------------------------------VALIDATION---------------------------------------------------------------

    def validate_gre(self, event):
        gre = float(self.gre.get())
        if gre > 340:
            self._gre.set(340.0)
    
    def validate_toefl(self, event):
        toefl = float(self.toefl.get())
        if toefl < 0:
            self._toefl.set(0.0)
        elif toefl > 120:
            self._toefl.set(120.0)
            
    def validate_u_rating(self, event):
        u_rating = float(self.u_rating.get())
        if u_rating < 0:
            self._u_rating.set(0.0)
        elif u_rating > 5:
            self._u_rating.set(5.0)
    
    def validate_sop(self, event):
        sop = float(self.sop.get())
        if sop < 0:
            self._sop.set(0.0)
        elif sop > 5:
            self._sop.set(5.0)
        
    def validate_lor(self, event):
        lor = float(self.lor.get())
        if lor < 0:
            self._lor.set(0.0)
        elif lor > 5:
            self._lor.set(5.0)

    
    def validate_cgpa(self, event):
        cgpa = float(self.cgpa.get())
        if cgpa < 0:
            self._cgpa.set(0.0)
        elif cgpa > 10:
            self._cgpa.set(10.0)

            
    def validate_research(self, event):
        research = float(self.u_rating.get())
        if research < 0:
            self._research.set(0.0)
        elif research > 1:
            self._research.set(1.0)


    # --------------------------------------------------------------END VALIDATION------------------------------------------------------------


    def home(self):
        self.frame.destroy()
        self.sub_frame.destroy()
        self.home_class = Home(self.main_frame)

    def form1(self):
        self._gre_label.grid(row=1, column=0)
        self.gre.grid(row=1, column=1)
        self._toefl_label.grid(row=1, column=2)
        self.toefl.grid(row=1, column=3)

        self._u_rating_label.grid(row=2, column=0)
        self.u_rating.grid(row=2, column=1)
        self._sop_label.grid(row=2, column=2)
        self.sop.grid(row=2, column=3)

        self._lor_label.grid(row=3, column=0)
        self.lor.grid(row=3, column=1)
        self._cgpa_label.grid(row=3, column=2)
        self.cgpa.grid(row=3, column=3)

        self._research_label.grid(row=4, column=0)
        self.research.grid(row=4, column=1)

    def show_result_form1(self):
        _result = tk.StringVar()
        _result.set("")
        result = tk.Label(self.frame, textvariable=_result)
        
        try:
            _ans = np.array(model1.predict([[self._gre.get(), self._toefl.get(), self._u_rating.get(
            ), self._sop.get(), self._lor.get(), self._cgpa.get(), self._research.get()]]))
            _ans = round(_ans.flatten()[0]*100, 2)

            if (_ans >= 0):
                if(_ans > 100):
                    _ans = 100.00
                pass
            else:
                _ans = 0
            _result.set("      Chance of admission: "+str(_ans)+" %      ")
        except:
            _result.set("ERROR! Enter only valid numbers.")
            
            
        result.grid(row=5, column=1, columnspan=4)

    def form2(self):
        self.check.configure(command=self.show_result_form2)
        self._u_rating_label.grid(row=1, column=0)
        self.u_rating.grid(row=1, column=1, columnspan=4)

    def show_result_form2(self):
        try:
            _gre, _toefl, _sod, _lor, _cgpa, _research = np.array(model2.predict([[self._u_rating.get()]])).flatten()
            if (_research<=0):
                _research = 0
        except:
            _gre, _toefl, _sod, _lor, _cgpa, _research = 0, 0, 0, 0, 0, 0
        
        self._u_rating_label.grid(row=1, column=0, columnspan=2)
        self._gre_label.grid(row=2, column=0)
        self.gre.grid(row=2, column=1)
        self._gre.set(round(_gre, 2))

        self._toefl_label.grid(row=2, column=2)
        self.toefl.grid(row=2, column=3)
        self._toefl.set(round(_toefl, 2))

        self._sop_label.grid(row=3, column=0)
        self.sop.grid(row=3, column=1)
        self._sop.set(round(_sod, 1))

        self._lor_label.grid(row=3, column=2)
        self.lor.grid(row=3, column=3)
        self._lor.set(round(_lor, 1))

        self._cgpa_label.grid(row=4, column=0)
        self.cgpa.grid(row=4, column=1)
        self._cgpa.set(round(_cgpa, 2))

        self._research_label.grid(row=4, column=2)
        self.research.grid(row=4, column=3)
        self._research.set(round(_research, 0))


class Home:
    def __init__(self, main_window):
        self.main_window = main_window
        self.font_style = "Times New Roman"
        self.font_size = "18"
        self.bg_color = "white"
        self.fg_color = "black"
        self.middle_frame()

    # --------------------------------------------------------------TOP FRAME---------------------------------------------------------------
    def top_frame(self):
        self._top_frame = tk.Frame(self.main_window)
        self._top_frame.pack(side=tk.TOP)
        self.logo = tk.Label(self._top_frame, text="University Graduates admission prediction system",
                        compound='top', font=("Algerian", 24))
        self.logo.pack()

    # ------------------------------------------------------------MIDDLE FRAME--------------------------------------------------------------
    def middle_frame(self):
        self._middle_frame = tk.Frame(self.main_window)
        self._middle_frame.place(relx=.5, rely=.5, anchor=tk.CENTER)

        _main_preict_ = tk.Button(self._middle_frame, text="Predict chance of admission in university",
                                  command=self.main_predict, font=(self.font_style, self.font_size))
        _main_preict_.pack()

        _sub_predict_ = tk.Button(self._middle_frame, text="Predict how much scores needed to get into specified rating university",
                                  command=self.sub_predict, font=(self.font_style, self.font_size))
        _sub_predict_.pack()

    def main_predict(self):
        self._middle_frame.destroy()
        _main_predict = features(self.main_window)
        _main_predict.form1()

    def sub_predict(self):
        self._middle_frame.destroy()
        _sub_predict = features(self.main_window)
        _sub_predict.form2()

    # ------------------------------------------------------------BOTTOM FRAME--------------------------------------------------------------

    def bottom_frame(self):
        self._bottom_frame = tk.Frame(self.main_window)
        self._bottom_frame.pack(side=tk.BOTTOM)

        
        self.origin = tk.Label(self._bottom_frame, text="Made with",
                          compound='right', font=("Garamond", 16))
        self.origin.pack()
        self.made_in = tk.Label(self._bottom_frame, text="In India",
                           compound='right', font=("Garamond", 16))
        self.made_in.pack()

        
        self.made_by = tk.Label(self._bottom_frame, text="By Piyush",
                           compound='right', font=("Garamond", 16))
        self.made_by.pack()

def image(img):
    path = r"K:\PROGRAMS\PYTHON\UAPS\UAPS---UNIVERSITY_ADMISSION_PREDICTION_SYSTEM\PHOTOS"+"\\"
    im = path+str(img)
    return ImageTk.PhotoImage(Image.open(im))


window = tk.Tk()

window.title("University Graduates Admission Prediction System")
window.geometry("1028x720")

main_window = Home(window)

main_window.top_frame()
lg = image("Logo.png")
main_window.logo.configure(image=lg)

main_window.bottom_frame()
heart = image("Heart.png")
Indian_flag = image("Indian_flag.png")
smile = image("Smile.png")
main_window.origin.configure(image=heart)
main_window.made_in.configure(image=Indian_flag)
main_window.made_by.configure(image=smile)

window.mainloop()
