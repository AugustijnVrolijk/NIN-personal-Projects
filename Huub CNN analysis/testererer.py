import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pickle
"""
vals = np.arange(1,100)

size = vals.shape[0]

mask = np.random.choice(vals,size,replace=False)

cuttoff = 20
a = mask[:cuttoff]
b = mask[cuttoff:]



class imageData(Dataset):
    def __init__(self, input):
        self.input = input

    def __len__(self):
        return len(self.input)
        
    def __getitem__(self, idx):
        print(idx)
        return self.input[idx]

test = imageData(vals)

test_dataLoader = DataLoader(test,batch_size=5, shuffle=True)


for i in range(1000):
    tempVal = next(iter(test_dataLoader))
    print(tempVal)
    print(type(tempVal))
    print(tempVal.shape)"
"""

gen_path = r"C:\Users\augus\NIN Stuff\data\Huub data"
output_filename = os.path.join(gen_path,'Test cross validation scores')  


gridfile = open(output_filename,"rb")
gridpickleFile = pickle.load(gridfile)

label1 = "trainingEpoch_loss"
label2 = "validationEpoch_loss"
y1 = gridpickleFile[label1]
y2 = gridpickleFile[label2]

length = len(y2)
x = np.arange(length)

plt.plot(x, y1, label=label1)  # Plot the first line (y1 vs x)
plt.plot(x, y2, label=label2)  # Plot the second line (y2 vs x)

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train/validation loss with best features taken from: Huub_NaturalImages_darkReared_VISa_baseline\nmixed3b, lr:0.01, weight_decay:0.005, smooth_weight:0.05')

# Add a legend to differentiate the two lines
plt.legend()

# Show the plot
plt.show()