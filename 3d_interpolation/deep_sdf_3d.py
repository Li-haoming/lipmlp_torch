import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import mnist2sdf
import deep_sdf_decoder_3d as deepsdf
from mpl_toolkits import mplot3d

data_1 = np.load('shape_1.npz')
pos_1 = data_1['pos'][:25000]
neg_1 = data_1['neg'][:25000]
i1 = torch.tensor(np.concatenate((pos_1,neg_1),axis=0))

data_2 = np.load('shape_2.npz')
pos_2 = data_2['pos'][:25000]
neg_2 = data_2['neg'][:25000]
i2 = torch.tensor(np.concatenate((pos_2,neg_2),axis=0))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image = [i1,i2]

t_list = []
t0 = 0
t1 = 1
t_list.append(t0)
t_list.append(t1)

epochs = 100
num_images = 2
batch_size = 100
lr = 0.0001

model = deepsdf.Decoder(latent_size=1, dims=[256,256,256,256,256], use_tanh=True)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_list = []
for epoch in range(epochs):
    for p in range(num_images):
        for i in range(500):
            t = t_list[p]
            t = torch.ones(100,1)*t
            
            position = image[p][i*100:(i+1)*100,:3].to(device)
            
            x = torch.cat([t,position], dim=1)
            targets = image[p][i*100:(i+1)*100,3].reshape(100,1).to(device)
            sdf = model(x)
            
            loss = F.mse_loss(sdf, targets)
            loss_numpy = loss.cpu().detach().numpy()
            loss_list.append(loss_numpy)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()        
            if (i+1) % 250 == 0:
                print ("Epoch[{}/{}], Image[{}/{}], Step [{}/{}], Loss: {:.7f}" 
                       .format(epoch+1, epochs, p, num_images, i+1, 500, loss.item()))

torch.save(model.state_dict(),'deepsdf_3d_params_1.pth')

plt.xlabel("number of updates")
plt.ylabel("average train loss")
plt.title("Train Loss")            
plt.plot(np.array(range(epochs*num_images*500)), loss_list)
plt.savefig("interpolation_3d_training_loss.jpg")
plt.show()  

