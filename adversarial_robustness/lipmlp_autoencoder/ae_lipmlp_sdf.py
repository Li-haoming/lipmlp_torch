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

class AE_lipmlp(nn.Module):
    def __init__(self):
        super(AE_lipmlp, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(34, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, 1)
      
    def weight_normalization(self, layer, x):
        layer_param = layer.state_dict()
        W = layer_param["weight"]
        #b = layer_param["bias"]
        c = torch.max(torch.sum(torch.abs(W), dim=1))
        softplus_c = F.softplus(c)
        absrowsum = torch.sum(torch.abs(W), dim=1)
        shape = (softplus_c/absrowsum).shape
        scale = torch.minimum(torch.ones(shape), softplus_c/absrowsum)
        W = W * scale[:,None]
        x = layer(x)
        return x
    
    def get_lipschitz_loss(self):
        """
        This function computes the Lipschitz regularization
        """
        w1 = self.fc1.state_dict()["weight"]
        w2 = self.fc2.state_dict()["weight"]
        w3 = self.fc3.state_dict()["weight"]
        w4 = self.fc4.state_dict()["weight"]
        w5 = self.fc5.state_dict()["weight"]
        w6 = self.fc6.state_dict()["weight"]
        w7 = self.fc7.state_dict()["weight"]
        w8 = self.fc8.state_dict()["weight"]
        
        softplus_c1 = F.softplus(torch.max(torch.sum(torch.abs(w1), dim=1)))
        softplus_c2 = F.softplus(torch.max(torch.sum(torch.abs(w2), dim=1)))
        softplus_c3 = F.softplus(torch.max(torch.sum(torch.abs(w3), dim=1)))
        softplus_c4 = F.softplus(torch.max(torch.sum(torch.abs(w4), dim=1)))
        softplus_c5 = F.softplus(torch.max(torch.sum(torch.abs(w5), dim=1)))
        softplus_c6 = F.softplus(torch.max(torch.sum(torch.abs(w6), dim=1)))
        softplus_c7 = F.softplus(torch.max(torch.sum(torch.abs(w7), dim=1)))
        softplus_c8 = F.softplus(torch.max(torch.sum(torch.abs(w8), dim=1)))
        
        loss_lip = softplus_c1*softplus_c2*softplus_c3*softplus_c4*softplus_c5*softplus_c6*softplus_c7*softplus_c8
        
        return loss_lip
    
    def encode(self, x):
        
        h1 = self.weight_normalization(self.fc1, x)
        h1 = F.leaky_relu(h1)
        
        h2 = self.weight_normalization(self.fc2, h1)
        h2 = F.leaky_relu(h2)
        
        h3 = self.weight_normalization(self.fc3, h2)
        h3 = F.leaky_relu(h3)
        
        z = self.weight_normalization(self.fc4, h3)
        z = torch.sigmoid(F.leaky_relu(z))
        
        return z
 
    def decode(self, z):
        
        h4 = self.weight_normalization(self.fc5, z)
        h4 = F.leaky_relu(h4)
        
        h5 = self.weight_normalization(self.fc6, h4)
        h5 = F.leaky_relu(h5)
        
        h6 = self.weight_normalization(self.fc7, h5)
        h6 = F.leaky_relu(h6)
        
        x_out = self.weight_normalization(self.fc8, h6)
        
        return x_out
    
    def forward(self, x, position):
        position = position * 100
        z = self.encode(x)
        new_z = torch.cat((z,position),1)
        sdf = self.decode(new_z)
        return sdf

epochs = 1
num_images = 100
batch_size = 28
lr = 0.0001

model = AE_lipmlp()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_list = []
val_loss_list = []
loss_lip_list = []
val_loss_lip_list = []
loss_mse_list = []
val_loss_mse_list = []
for epoch in range(epochs):
    for p in range(num_images):
        for i in range(batch_size):
            train_dict = train_dataset.__getitem__(p)
            x = train_dict["all"][1].reshape(1, 784).repeat(28, 1).to(device)
            position = train_dict["all"][0][i*28:(i+1)*28].to(device)
            targets = train_dict["all"][1][i*28:(i+1)*28].to(device)
            
            sdf = model(x, position)
            
            loss_mse = F.mse_loss(sdf, targets)
            loss_mse_np = loss_mse.cpu().detach().numpy()
            loss_mse_list.append(loss_mse_np)
            
            loss_lip = model.get_lipschitz_loss()
            loss_lip_np = loss_lip.cpu().detach().numpy()
            loss_lip_list.append(loss_lip_np)
            
            loss = (0.000001 * loss_lip + loss_mse)*100
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
        
        val_loss_mse = F.mse_loss(val_sdf, val_targets)
        val_loss_mse_np = val_loss_mse.cpu().detach().numpy()
        val_loss_mse_list.append(val_loss_mse_np)
        
        val_loss_lip = model.get_lipschitz_loss()
        val_loss_lip_np = val_loss_lip.cpu().detach().numpy()
        val_loss_lip_list.append(val_loss_lip_np)
        
        val_loss = (val_loss_mse + 0.000001*val_loss_lip)*100
        val_loss_numpy = val_loss.cpu().detach().numpy()
        val_loss_list.append(val_loss_numpy)
        
        print("Val Loss: ", val_loss)
torch.save(model.state_dict(),'ae_lipmlp_params_1.pth')  

plt.subplot(1,2,1)
plt.xlabel("number of updates")
plt.ylabel("average train loss")
plt.title("Train MSE Loss")            
plt.plot(np.array(range(epochs*num_images*batch_size)), loss_mse_list)
plt.subplot(1,2,2)
plt.xlabel("number of updates")
plt.ylabel("average train loss")
plt.title("Train Lipschitz Loss")            
plt.plot(np.array(range(epochs*num_images*batch_size)), loss_lip_list)
plt.savefig("mse_lip_loss_lipae.png")
plt.show()

plt.xlabel("number of updates")
plt.ylabel("average train loss")
plt.title("Train Loss")            
plt.plot(np.array(range(epochs*num_images*batch_size)), loss_list)
plt.savefig("train_loss_lipae.png")
plt.show()
              
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
    plt.savefig("lipAE_reconstructed_original-{}.png".format(i))
    plt.show()
  
plt.subplot(1,2,1)
plt.xlabel("number of 100*updates")
plt.ylabel("average validation loss")
plt.title("Validation MSE Loss")            
plt.plot(np.array(range(epochs*num_images)), val_loss_mse_list)
plt.subplot(1,2,2)
plt.xlabel("number of 100*updates")
plt.ylabel("average val loss")
plt.title("Validation Lipschitz Loss")            
plt.plot(np.array(range(epochs*num_images)), val_loss_lip_list)
plt.savefig("val_mse_lip_loss_lipae.png")
plt.show()

plt.xlabel("number of updates")
plt.ylabel("average train loss")
plt.title("Validation Loss")            
plt.plot(np.array(range(epochs*num_images*batch_size)), loss_list)
plt.savefig("val_loss_lipae.png")
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
      plt.savefig("lipAE_initial_adv-{}.png".format(i))
      plt.show()
