import scipy.io
import numpy as np
import os
from numpy  import array
vdata = []

for file in os.listdir("ECG/NSR/"):

  data = scipy.io.loadmat("ECG/NSR/"+file)
  datae = data['val']
  
  vdata.append(datae[0])
  print ",".join([str(x) for x in datae[0]] )  

np.savetxt("NSR.csv", vdata, fmt="%0d",delimiter=',')
