import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mpl_toolkits import mplot3d

a = np.linspace([0,0.0345,0], [0, 0.04, 0.04], 100)

axes=plt.axes(projection="3d")
axes.plot(*a.T)
plt.show()