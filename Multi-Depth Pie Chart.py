import sys
print (sys.path)
import  sys,os
sys.path.insert(0, os.path.join(os.path.abspath(sys.path[0]), "C:/Users/adm/Documents/Python Scripts/PycharmProjects/pythonProject", "lib"))
from pychartdir import *



from pygooglechart import PieChart3D
def python_pie3D():
    chart = PieChart3D(250,250)
    chart.add_data([398,294,840,462])
    chart.set_pie_labels(" Lithuania Bulgaria Ukraine Romania".split())# making labels for the slices
    chart.download('revenue_east_europe.png')