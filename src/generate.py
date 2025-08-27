import torch
import matplotlib.pyplot as plt


class Generator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def generate(self, num):
        self.model.eval()
        label_onehot = torch.zeros(1, self.model.num_classes, device=self.device)
        label_onehot[0, num] = 1
        z = torch.randn(1, self.model.latent_dim, device=self.device)
        with torch.no_grad():
            img = self.model.decode(z, label_onehot).cpu().squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
