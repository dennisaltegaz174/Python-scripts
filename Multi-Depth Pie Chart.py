import  sys,os
sys.path.insert(0, os.path.join(os.path.abspath(sys.path[0]),"C:/Program Files/JetBrains/PyCharm Community Edition 2021.2.2","lib"))
from pychartdir import *

python-c "import sys; print(sys.path)"
import num


from pygooglechart import PieChart3D
def python_pie3D():
    chart = PieChart3D(250,250)
    chart.add_data([398,294,840,462])
    chart.set_pie_labels(" Lithuania Bulgaria Ukraine Romania".split())# making labels for the slices
    chart.download('revenue_east_europe.png')