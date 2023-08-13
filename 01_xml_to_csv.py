#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import xml.etree.ElementTree as xet 
from glob import glob 

path = glob('./dataset/*.xml')
print(path)


# In[3]:


import pandas as pd
import xml.etree.ElementTree as xet 
from glob import glob 

path = glob('./Users/shiva/OneDrive/Desktop/anpr/dataset/*.xml')
print(path)


# In[4]:


import pandas as pd
import xml.etree.ElementTree as xet 
from glob import glob 

path = glob('./Users/shiva/OneDrive/Desktop/anpr/dataset/*.xml')
print(path)


# In[5]:


import pandas as pd
import xml.etree.ElementTree as xet 
from glob import glob 

path = glob('C:/Users/shiva/OneDrive/Desktop/anpr/XML/*.xml')
print(path)


# In[7]:


labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
for filename in path:

     #filename path[0]
     info = xet.parse(filename)
     root = info.getroot()
     member_object = root.find('object')
     labels_info = member_object.find('bndbox') 
     xmin = int(labels_info.find('xmin').text)
     xmax = int(labels_info.find('xmax').text)
     ymin = int(labels_info.find('ymin').text)
     ymax = int(labels_info.find('ymax').text)
     #print (xwin, cx, yn, max)
     labels_dict['filepath'].append(filename)
     labels_dict["xmin"].append(xmin)
     labels_dict["xmax"].append(xmax)
     labels_dict["ymin"].append(ymin)
     labels_dict["ymax"].append(ymax)


# In[8]:


labels_dict = {
    'filepath': [],
    'xmin': [],
    'xmax': [],
    'ymin': [],
    'ymax': []
}

for filename in path:
    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox') 
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)
    
    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

# Create a DataFrame from the labels_dict
labels_df = pd.DataFrame(labels_dict)

# Display the DataFrame
print(labels_df)


# In[10]:


csv_filename = 'bounding_boxes.csv'
labels_df.to_csv(csv_filename, index=False)

print(f'Bounding box data saved to {csv_filename}')


# In[ ]:




