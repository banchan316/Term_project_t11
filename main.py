import numpy as np
import pandas as pd
from preprocessing import preprocessing

df = pd.read_csv("US_Accidents_March23_sampled_500k.csv")
preprocessing(df)