
#* create data and file
import pandas as pd
data = pd.DataFrame(X) #X is feature
data['label'] = Y  # Y is labels
#%Save to CSV
data.to_csv('data.csv', index=False)

#*create link to download
from IPython.display import FileLink
FileLink('data.csv')
