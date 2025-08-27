import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import math

class Trainer:
    def __init__(self, model, dataloader, device='cuda', lr=1e-3, checkpoint_dir='../outputs/checkpoints'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter('../outputs/logs')
        self.dataset = dataloader.dataset

    def loss_fn(self, recon_x, x, mu, logvar, class_pred=None, true_labels=None, beta=1.0):

        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        classification_loss = 0
        if class_pred is not None and true_labels is not None:
            classification_loss = F.cross_entropy(class_pred, true_labels, reduction='sum')

        total_loss = recon_loss + beta * kl_loss + 0.1 * classification_loss
        return total_loss, recon_loss, kl_loss, classification_loss

    def train(self, epochs=20, checkpoint_interval=20, steps_per_epoch=None, start_epoch=1):
        self.model.train()

        print(f"Начинаем обучение на {epochs} эпох (с {start_epoch}-й)")
        print(f"Размер датасета: {len(self.dataset)} изображений")
        print(f"Размер батча: {self.dataloader.batch_size}")

        for epoch in range(start_epoch, epochs + 1):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_class_loss = 0
            num_batches = 0

            beta = 0.5 * (1 - math.cos(math.pi * min(1.0, epoch / epochs)))

            for batch_idx, (x, labels) in enumerate(self.dataloader):
                if steps_per_epoch and batch_idx >= steps_per_epoch:
                    break

                x = x.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                labels_onehot = torch.zeros(x.size(0), self.model.num_classes, device=self.device)
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

                self.optimizer.zero_grad()
                recon, mu, logvar, class_pred = self.model(x, labels_onehot)

                loss, recon_loss, kl_loss, class_loss = self.loss_fn(
                    recon, x, mu, logvar, class_pred, labels, beta=beta
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_class_loss += class_loss.item() if class_loss else 0
                num_batches += 1

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch:3d}, Batch {batch_idx:4d}: Loss={loss.item() / x.size(0):.4f}, β={beta:.3f}")

            avg_total_loss = total_loss / len(self.dataset)
            avg_recon_loss = total_recon_loss / len(self.dataset)
            avg_kl_loss = total_kl_loss / len(self.dataset)
            avg_class_loss = total_class_loss / len(self.dataset)

            print(
                f"Epoch {epoch:3d}: Total={avg_total_loss:.4f} | Recon={avg_recon_loss:.4f} | KL={avg_kl_loss:.4f} | Class={avg_class_loss:.4f} | β={beta:.3f}")

            if self.writer:
                self.writer.add_scalar('Loss/Total', avg_total_loss, epoch)
                self.writer.add_scalar('Loss/Reconstruction', avg_recon_loss, epoch)
                self.writer.add_scalar('Loss/KL', avg_kl_loss, epoch)
                self.writer.add_scalar('Loss/Classification', avg_class_loss, epoch)
                self.writer.add_scalar('Beta', beta, epoch)

            if epoch % checkpoint_interval == 0:
                self.save_checkpoint(epoch, avg_total_loss)

    def save_checkpoint(self, epoch, loss, is_final=False):
        if is_final:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_final.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'model_config': {
                'latent_dim': self.model.latent_dim,
                'num_classes': self.model.num_classes
            }
        }, checkpoint_path)

        print(f"Checkpoint сохранен: {checkpoint_path}")

    def compute_beta(epoch, total_warmup_epochs=50):
        progress = min(1.0, epoch / total_warmup_epochs)
        return 0.5 * (1 - math.cos(math.pi * progress))

    def validate(self, val_dataloader=None):
        if val_dataloader is None:
            return

        self.model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for x, labels in val_dataloader:
                x = x.to(self.device)
                labels = labels.to(self.device)

                labels_onehot = torch.zeros(x.size(0), self.model.num_classes, device=self.device)
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

                recon, mu, logvar, class_pred = self.model(x, labels_onehot)
                loss, _, _, _ = self.loss_fn(recon, x, mu, logvar, class_pred, labels)

                total_val_loss += loss.item()

                # Точность классификации
                pred_labels = torch.argmax(class_pred, dim=1)
                correct_predictions += (pred_labels == labels).sum().item()
                total_samples += labels.size(0)

        avg_val_loss = total_val_loss / total_samples
        accuracy = correct_predictions / total_samples

        print(f"Validation: Loss={avg_val_loss:.4f}, Accuracy={accuracy:.4f}")
        self.model.train()

        return avg_val_loss, accuracy

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} не найден")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"Checkpoint загружен: {checkpoint_path}, начиная с эпохи {start_epoch}")
        return start_epoch