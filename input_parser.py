import pandas as pd

df = pd.read_csv('det.csv', names=['frame', 'id', 'bb_left','bb_top', 'bb_width','bb_height','cnf',
         'x', 'y', 'z'])

df.head()