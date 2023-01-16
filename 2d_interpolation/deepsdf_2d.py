import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import mnist2sdf
import deep_sdf_decoder as deepsdf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = mnist2sdf.MNISTSDFDataset('train', size=(28, 28))
val_dataset = mnist2sdf.MNISTSDFDataset('val', size=(28, 28))

image = []
i0 = train_dataset.__getitem__(1)["all"]
i1 = train_dataset.__getitem__(23)["all"]

image.append(i0)
image.append(i1)

t_list = []
t0 = 0
t1 = 1
t_list.append(t0)
t_list.append(t1)

epochs = 1000
num_images = 2
batch_size = 28
lr = 0.0001

model = deepsdf.Decoder(latent_size=1, dims=[64,64,64,64,64])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_list = []
for epoch in range(epochs):
    for p in range(num_images):
        for i in range(batch_size):
            t = t_list[p]
            t = torch.ones(28,1)*t
            position = image[p][0][i*28:(i+1)*28].to(device)
            x = torch.cat([t,position], dim=1)
            targets = image[p][1][i*28:(i+1)*28].to(device)
            sdf = model(x)
            loss = F.mse_loss(sdf, targets)*100
            loss_numpy = loss.cpu().detach().numpy()
            loss_list.append(loss_numpy)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()        
            if (i+1) % 14 == 0:
                print ("Epoch[{}/{}], Image[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(epoch+1, epochs, p, num_images, i+1, 28, loss.item()))

torch.save(model.state_dict(),'deepsdf_params_1.pth')

plt.xlabel("number of updates")
plt.ylabel("average train loss")
plt.title("Train Loss")            
plt.plot(np.array(range(epochs*num_images*batch_size)), loss_list)
plt.show()  

with torch.no_grad():
    val_t_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  
    position = image[0][0]
    output = []
    for i in range(11):
        t = val_t_list[i]
        t = torch.ones(784,1)*t
        x = torch.cat([t,position], dim=1)
        sdf = model(x)
        output.append(sdf)

for i in range(11):
    plt.imshow(output[i].reshape(28,28))
    plt.savefig("interpolation-{}.png".format(i))
    plt.show()
  
