import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import mnist2sdf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = mnist2sdf.MNISTSDFDataset('train', size=(28, 28))
val_dataset = mnist2sdf.MNISTSDFDataset('val', size=(28, 28))

train_data = train_dataset.img_dataset
val_data = val_dataset.img_dataset

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(34, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, 1)
        
    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        h3 = F.leaky_relu(self.fc3(h2))
        z = torch.sigmoid(F.leaky_relu(self.fc4(h3)))
        return z
 
    def decode(self, z):
        h4 = F.leaky_relu(self.fc5(z))
        h5 = F.leaky_relu(self.fc6(h4))
        h6 = F.leaky_relu(self.fc7(h5))
        x_out = self.fc8(h6)
        return x_out
    
    def forward(self, x, position):
        position = position * 100
        z = self.encode(x)
        new_z = torch.cat((z,position),1)
        sdf = self.decode(new_z)
        return sdf
    
epochs = 1
num_images = 60000
batch_size = 28
lr = 0.0001

model = AE()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_list = []
val_loss_list = []
for epoch in range(epochs):
    for p in range(num_images):
        for i in range(batch_size):
            train_dict = train_dataset.__getitem__(p)
            x = train_dict["all"][1].reshape(1, 784).repeat(28, 1).to(device)
            position = train_dict["all"][0][i*28:(i+1)*28].to(device)
            targets = train_dict["all"][1][i*28:(i+1)*28].to(device)
            sdf = model(x, position)
            loss = F.mse_loss(sdf, targets)*100
            loss_numpy = loss.cpu().detach().numpy()
            loss_list.append(loss_numpy)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()        
            if (i+1) % 14 == 0:
                print ("Epoch[{}/{}], Image[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(epoch+1, epochs, p, num_images, i+1, 28, loss.item()))
        val_dict = val_dataset.__getitem__(0)
        val_x = val_dict["all"][1].reshape(1, 784).repeat(784, 1).to(device)
        val_position = val_dict["all"][0].to(device)
        val_targets = val_dict["all"][1].to(device)
        val_sdf = model(val_x, val_position)
        val_loss = F.mse_loss(val_sdf, val_targets)*100 
        val_loss_numpy = val_loss.cpu().detach().numpy()
        val_loss_list.append(val_loss_numpy)
        print("Val Loss: ", val_loss)
torch.save(model.state_dict(),'ae_params_1.pth')                
with torch.no_grad():
  
  samples = torch.zeros(10, 784).to(device)
  output = torch.zeros(10, 784).to(device)
  for i in range(10):
      val_dict = val_dataset.__getitem__(i)
      samples[i] = val_dict["all"][1].reshape(1, 784)
      val_x = samples[i].repeat(784, 1).to(device)
      val_position = val_dict["all"][0].to(device)
      val_sdf = model(val_x, val_position)
      output[i] = val_sdf.reshape(1, 784)

for i in range(10):
    plt.figure(i)
    plt.subplot(1,2,1)
    plt.title("Reconstructed Image")
    plt.imshow(output[i].cpu().reshape(28,28))
    plt.subplot(1,2,2)
    plt.title("Original Image")
    plt.imshow(samples[i].cpu().reshape(28,28))
    plt.savefig("reconstructed_original_imgs-{}.png".format(i))
    plt.show()
  
plt.subplot(1,2,1)
plt.xlabel("number of updates")
plt.ylabel("average train loss")
plt.title("Train Loss")            
plt.plot(np.array(range(epochs*num_images*batch_size)), loss_list)
plt.subplot(1,2,2)
plt.xlabel("number of 100*updates")
plt.ylabel("average val loss")
plt.title("Validation Loss")            
plt.plot(np.array(range(epochs*num_images)), val_loss_list)
plt.savefig("train_val_loss.png")
plt.show()

def fgsm_attack(model, loss, image, position, targets, eps=0.05) :
    
    image.requires_grad = True
            
    output = model(image, position)
    
    model.zero_grad()
    cost = loss(output, targets)
    cost.backward()
    
    attack_image = image + eps*image.grad.sign()
    attack_image = torch.clamp(attack_image, 0, 1)
    
    return attack_image

def getAdvMNIST(model, loss, image, image0, position, position0, e=0.05):
    #image0.requires_grad = True
    position = position*100
    position0 = position0*100
    t0 = model.encode(image0)
    t = model.encode(image)
    embedding0 = torch.cat((t0,position0),1)
    J = loss(image, model.decode(embedding0))
    t0.retain_grad()
    model.zero_grad()
    J.backward()
    d = t0.grad
    t_adv = t + e * torch.sign(d)
    embedding_adv = torch.cat((t_adv,position),1)
    image_adv = model.decode(embedding_adv).reshape(1, 784)
    
    return image_adv

loss = torch.nn.MSELoss()
model.eval()

val_adv = torch.zeros(100, 784)
image0 = train_dataset.__getitem__(0)["all"][1].reshape(1, 784).repeat(784, 1)
position0 = train_dataset.__getitem__(0)["all"][0]
for i in range(100):
    val_dict = val_dataset.__getitem__(i)
    image = val_dict["all"][1].reshape(1, 784).repeat(784, 1)
    position = val_dict["all"][0]
    targets = val_dict["all"][1]
    val_adv[i] = getAdvMNIST(model, loss, image, image0, position, position0, e=0.05)
val_adv = val_adv.detach().numpy()

for i in range(10):
      plt.figure(i)
      plt.subplot(1,2,1)
      plt.title("AE initial")
      plt.imshow(output[i].cpu().reshape(28,28))
      plt.subplot(1,2,2)
      plt.title("AE adv.")
      plt.imshow(val_adv[i].reshape(28,28))
      plt.savefig("AE_initial_adv-{}.png".format(i))
      plt.show()
