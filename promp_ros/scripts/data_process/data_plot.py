from datetime import time
from numpy.lib.function_base import insert
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
# from bioinfokit.analys import stat
import statsmodels
import numpy as np
import rospkg

r = rospkg.RosPack()
data_path = r.get_path('promp_ros')

df = pd.read_excel(data_path+'/data/time.xlsx', usecols=[0,1,2,3,4])
# print(df)
hr_kf = df[df['Human-Robot']=='With KF']
hr_no_kf = df[df['Human-Robot']=='Without KF']
pa_kf = df[df['Plug Alignment']=='With KF']
pa_nokf = df[df['Plug Alignment']=='Without KF']
insertion_p = df[df['Insertion']=='Passive']
insertion_a = df[df['Insertion']=='Active']


def plot_result():
    box_w = 0.5
    figure = plt.figure(1, figsize=(8, 5))
    ## color
    col = 'k'
    
    ## Human robot (kf)
    ax1 = figure.add_subplot(131)
    sns.boxplot(x='Human-Robot', y='Time (s)', data=df,  width=box_w, ax=ax1)
    d1 = hr_kf['Time (s)']
    d2 = hr_no_kf['Time (s)']
    ## Anova test
    fv, pv = stats.f_oneway(d1, d2)
    print(pv)
    y = 1.01*max(max(d1), max(d2))
    ## specify decimals in scientific notation
    text = f"p={pv:.2e}"
    ## space between line and box
    h = 0.1
    plt.plot([0, 0, 1, 1],[y, y+h, y+h, y], lw=1.5, c=col)
    plt.text(0.5, y+h, text, ha="center", va="bottom", color=col)
    
    ## Plug alignment (kf)
    ax2 = figure.add_subplot(132)
    sns.boxplot(x='Plug Alignment', y='Time (s)', data=df,  width=box_w, ax=ax2)
    d1 = pa_kf['Time (s)']
    d2 = pa_nokf['Time (s)']
    fv, pv = stats.f_oneway(d1, d2)
    print(pv)
    y = 1.01*max(max(d1), max(d2))
    text = f"p={pv:.2e}"
    ## space between line and box
    h = 0.1
    plt.plot([0, 0, 1, 1],[y, y+h, y+h, y], lw=1.5, c=col)
    plt.text(0.5, y+h, text, ha="center", va="bottom", color=col)
    
    ## Insertion (p/a)
    ax3 = figure.add_subplot(133)
    sns.boxplot(x='Insertion', y='Time (s)', order=[ 'Active', 'Passive'], data=df, width=box_w, ax=ax3)
    d1 = insertion_p['Time (s)']
    d2 = insertion_a['Time (s)']
    fv, pv = stats.f_oneway(d1, d2)
    print(pv)
    y = 1.01*max(max(d1), max(d2))
    text = f"p={pv:.2e}"
    ## space between line and box
    h = 0.1
    plt.plot([0, 0, 1, 1],[y, y+h, y+h, y], lw=1.5, c=col)
    plt.text(0.5, y+h, text, ha="center", va="bottom", color=col)
    
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    # plot_comparison("Arrival Position Error (m)", 0.02)
    # plot_comparison("Arrival Orientation Error (rad)", 0.03)
    # plot_comparison("Navigation Time (s)", 15)
    # plot_comparison("Total Commands", 2.5)
    anova_analysis()
    plot_result()
    pass



