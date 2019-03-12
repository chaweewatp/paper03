import pandas as pd
import numpy as np
from firebase import firebase
firebase = firebase.FirebaseApplication('https://thesis-10ad5.firebaseio.com', None)

#retieve data
result = firebase.get('/THP/0', None)



print('finish')