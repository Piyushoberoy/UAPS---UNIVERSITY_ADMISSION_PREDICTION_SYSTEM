# --------------------------------------------------------------------LAYOUT MODULES------------------------------------------------------------------------
import tkinter as tk
import tkinter.messagebox as tk_msg
from PIL import ImageTk, Image

# ------------------------------------------------------------PREDICTION & ANALYSATION MODULES--------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

file = pd.read_csv("K:\\TRAINING PROJECT\\admission_data.csv")
file.rename(columns={'Chance of Admit ': 'Chance of Admit',
                     'LOR ': 'LOR'}, inplace=True)


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

        self._toefl_label = tk.Label(self.frame, text="TOEFL Score")
        self._toefl = tk.DoubleVar()
        self.toefl = tk.Entry(self.frame, textvariable=self._toefl, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)

        self._u_rating_label = tk.Label(self.frame, text="University Rating")
        self._u_rating = tk.DoubleVar()
        self.u_rating = tk.Entry(self.frame, textvariable=self._u_rating, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)

        self._sop_label = tk.Label(self.frame, text="SOP")
        self._sop = tk.DoubleVar()
        self.sop = tk.Entry(self.frame, textvariable=self._sop, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)

        self._lor_label = tk.Label(self.frame, text="LOR")
        self._lor = tk.DoubleVar()
        self.lor = tk.Entry(self.frame, textvariable=self._lor, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)

        self._cgpa_label = tk.Label(self.frame, text="CGPA")
        self._cgpa = tk.DoubleVar()
        self.cgpa = tk.Entry(self.frame, textvariable=self._cgpa, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)

        self._research_label = tk.Label(self.frame, text="Research")
        self._research = tk.IntVar()
        self.research = tk.Entry(self.frame, textvariable=self._research, font=(
            self.font_style, self.font_size), bg=self.bg_color, fg=self.fg_color)

        self.check = tk.Button(self.frame, text="Check", command=self.show_result_form1, font=(
            self.font_style, self.font_size))
        self.check.grid(row=6, column=1, columnspan=4)
        self.home_btn = tk.Button(self.frame, text="Home", command=self.home, font=(
            self.font_style, self.font_size))
        self.home_btn.grid(row=7, column=1, columnspan=4)

        # ----------------------------------------------------------------MODEL CREATION & PREDICTION-----------------------------------------------------------------
        self.kf = KFold(n_splits=200)
        # --------------------------------------------------------------------------MODEL 1---------------------------------------------------------------------------
        mod_file = file.drop(["Chance of Admit"], axis="columns")
        Chance_of_admit = pd.DataFrame(file["Chance of Admit"])

        self.f1_X_train, self.f1_X_test, self.f1_y_train, self.f1_y_test = self.train_test_case(
            mod_file, Chance_of_admit, self.get_score1)

        # --------------------------------------------------------------------------MODEL 2---------------------------------------------------------------------------
        U_rating = file["University Rating"]
        mod_file_1 = file.drop(
            ["Chance of Admit", "University Rating"], axis="columns")
        self.f2_X_train, self.f2_X_test, self.f2_y_train, self.f2_y_test = self.train_test_case(
            U_rating, mod_file_1, self.get_score2)

        # --------------------------------------------------------------END MODEL CREATION & PREDICTION---------------------------------------------------------------

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

    def train_test_case(self, input, output, score_calculator):
        score = [0]
        f_X_train, f_X_test, f_y_train, f_y_test = 0, 0, 0, 0
        for train_index, test_index in self.kf.split(input):
            X_train, X_test, y_train, y_test = input.loc[train_index], input.loc[
                test_index], output.loc[train_index], output.loc[test_index]
            res = round(score_calculator(LinearRegression(),
                                         X_train, X_test, y_train, y_test)*100, 2)
            if res > max(score):
                f_X_train, f_X_test, f_y_train, f_y_test = X_train, X_test, y_train, y_test
            score.append(res)
        return f_X_train, f_X_test, f_y_train, f_y_test

    def get_score1(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)

    def show_result_form1(self):
        _result = tk.StringVar()
        _result.set("")
        result = tk.Label(self.frame, textvariable=_result)

        # ----------------------------------------------------------------MODEL CREATION & PREDICTION-----------------------------------------------------------------
        model = LinearRegression()
        model.fit(self.f1_X_train, self.f1_y_train)
        # --------------------------------------------------------------END MODEL CREATION & PREDICTION---------------------------------------------------------------

        try:
            _ans = np.array(model.predict([[self._gre.get(), self._toefl.get(), self._u_rating.get(
            ), self._sop.get(), self._lor.get(), self._cgpa.get(), self._research.get()]]))
            _ans = round(_ans.flatten()[0]*100, 2)

            if (_ans >= 0):
                pass
            else:
                _ans = 0
            _result.set("Chance of admission: "+str(_ans)+" %")
        except:
            _result.set("ERROR! Enter only valid numbers.")

        result.grid(row=5, column=1, columnspan=4)

    def get_score2(self, model, X_train, X_test, y_train, y_test):
        model.fit(np.array(X_train).reshape(-1, 1), y_train)
        return model.score(np.array(X_test).reshape(-1, 1), y_test)

    def form2(self):
        self.check.configure(command=self.show_result_form2)
        self._u_rating_label.grid(row=1, column=0)
        self.u_rating.grid(row=1, column=1, columnspan=4)

    def show_result_form2(self):
        # ----------------------------------------------------------------MODEL CREATION & PREDICTION-----------------------------------------------------------------
        model = LinearRegression()
        model.fit(pd.DataFrame(self.f2_X_train), self.f2_y_train)

        print(round(model.score(pd.DataFrame(self.f2_X_test), self.f2_y_test)*100, 2))
        try:
            _gre, _toefl, _sod, _lor, _cgpa, _research = np.array(model.predict([[self._u_rating.get()]])).flatten()
            if (_research<=0):
                _research = 0
        except:
            _gre, _toefl, _sod, _lor, _cgpa, _research = 0, 0, 0, 0, 0, 0
        # --------------------------------------------------------------END MODEL CREATION & PREDICTION---------------------------------------------------------------
        
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

    # def custom_form(self):
    #     _ch_GRE = tk.IntVar()
    #     ch_GRE = tk.Checkbutton(self.sub_frame, text = "GRE Score", variable = _ch_GRE, font=(
    #         self.font_style, str(int(self.font_size)-6)))
    #     ch_GRE.grid(row=1, column=0, sticky="w")

    #     _ch_TOEFL = tk.IntVar()
    #     ch_TOEFL = tk.Checkbutton(self.sub_frame, text = "TOEFL Score", variable = _ch_TOEFL, font=(
    #         self.font_style, str(int(self.font_size)-6)))
    #     ch_TOEFL.grid(row=1, column=1, sticky="w")

    #     _ch_u_rating = tk.IntVar()
    #     ch_u_rating = tk.Checkbutton(self.sub_frame, text = "University Rating", variable = _ch_u_rating, font=(
    #         self.font_style, str(int(self.font_size)-6)))
    #     ch_u_rating.grid(row=2, column=0, sticky="w")

    #     _ch_SOP = tk.IntVar()
    #     ch_SOP = tk.Checkbutton(self.sub_frame, text = "SOP", variable = _ch_SOP, font=(
    #         self.font_style, str(int(self.font_size)-6)))
    #     ch_SOP.grid(row=2, column=1, sticky="w")

    #     _ch_LOR = tk.IntVar()
    #     ch_LOR = tk.Checkbutton(self.sub_frame, text = "LOR", variable = _ch_LOR, font=(
    #         self.font_style, str(int(self.font_size)-6)))
    #     ch_LOR.grid(row=3, column=0, sticky="w")

    #     _ch_CGPA = tk.IntVar()
    #     ch_CGPA = tk.Checkbutton(self.sub_frame, text = "CGPA", variable = _ch_CGPA, font=(
    #         self.font_style, str(int(self.font_size)-6)))
    #     ch_CGPA.grid(row=3, column=1, sticky="w")

    #     _ch_research = tk.IntVar()
    #     ch_research = tk.Checkbutton(self.sub_frame, text = "Research", variable = _ch_research, font=(
    #         self.font_style, str(int(self.font_size)-6)))
    #     ch_research.grid(row=4, column=0, columnspan=4)

    #     self.check.destroy()
    #     self.home_btn.destroy()

    #     self.check = tk.Button(self.sub_frame, text="GO", command=lambda:self.select_model(_ch_GRE.get(), _ch_TOEFL.get(), _ch_u_rating.get(), _ch_SOP.get(), _ch_LOR.get(), _ch_CGPA.get(), _ch_research.get()), font=(self.font_style, self.font_size))
    #     self.check.grid(row=6, column=0, columnspan=4)
    #     self.home_btn = tk.Button(self.sub_frame, text="Home", command=self.home, font=(self.font_style, self.font_size))
    #     self.home_btn.grid(row=7, column=0, columnspan=4)


    # def select_model(self, p1, p2, p3, p4, p5, p6, p7):
    #     self.sub_frame.destroy()
    #     print(p1, p2, p3, p4, p5, p6, p7)
    #     self.form1()
    #     self.check = tk.Button(self.frame, text="Check", command=self.show_result_custom_form, font=(self.font_style, self.font_size))
    #     self.check.grid(row=6, column=0, columnspan=4)
        
    #     self.home_btn = tk.Button(self.frame, text="Home", command=self.home, font=(self.font_style, self.font_size))
    #     self.home_btn.grid(row=7, column=0, columnspan=4)

    # def show_result_custom_form(self):
    #     pass


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

        # _custom_predict_ = tk.Button(self._middle_frame, text="Custom prediction",
        #                              command=self.custom_predict, font=(self.font_style, self.font_size))
        # _custom_predict_.pack()

    def main_predict(self):
        self._middle_frame.destroy()
        _main_predict = features(self.main_window)
        _main_predict.form1()

    def sub_predict(self):
        self._middle_frame.destroy()
        _sub_predict = features(self.main_window)
        _sub_predict.form2()

    # def custom_predict(self):
    #     self._middle_frame.destroy()
    #     _custom_predict = features(self.main_window)
    #     _custom_predict.custom_form()

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
    path = "K:\TRAINING PROJECT\PHOTOS"+"\\"
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
