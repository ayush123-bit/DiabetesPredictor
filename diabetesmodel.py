# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
model=pickle.load(open("./trainedmodel.sav",'rb'))
input_data=(4,110,92,0,0,37.6,0.191,30)
input_numpy=np.asarray(input_data)
input_reshaped=input_numpy.reshape(1,-1)

prediction=model.predict(input_reshaped)
print(prediction)
if(prediction[0]==1):
  print("Diabetic")
else:
  print("Not Diabetic")  