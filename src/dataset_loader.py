import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class DigitDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        print(f"Загружаем данные из: {data_dir}")

        for digit in range(10):
            digit_folder = os.path.join(data_dir, str(digit))
            if os.path.exists(digit_folder):
                image_files = [f for f in os.listdir(digit_folder)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for img_file in image_files:
                    img_path = os.path.join(digit_folder, img_file)
                    # ВАЖНО: label = digit (номер папки)
                    self.samples.append((img_path, digit))

                print(f"  Папка '{digit}': {len(image_files)} изображений -> label {digit}")

        print(f"Всего загружено: {len(self.samples)} изображений")

        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        print("Распределение по классам:")
        for digit in sorted(class_counts.keys()):
            print(f"  Класс {digit}: {class_counts[digit]} изображений")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Ошибка загрузки {img_path}: {e}")
            image = Image.new('L', (64, 64), 0)

        if self.transform:
            image = self.transform(image)

        return image, label


class MNISTDataset:
    def __init__(self, data_dir, batch_size=1024):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),

        ])

        self.dataset = DigitDataset(data_dir, transform=self.transform)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

    def get_loader(self):
        return self.dataloader

    def get_class_info(self):
        """Возвращает информацию о классах"""
        class_counts = {}
        for _, label in self.dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts


class MNISTDatasetWithAugmentation(MNISTDataset):
    def __init__(self, data_dir, batch_size=1024, use_augmentation=True):
        if use_augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])

        self.dataset = DigitDataset(data_dir, transform=self.transform)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )


def test_dataset_loading(data_dir):
    print("=== ТЕСТ ЗАГРУЗКИ ДАННЫХ ===")

    dataset = MNISTDataset(data_dir, batch_size=32)
    loader = dataset.get_loader()

    for batch_idx, (images, labels) in enumerate(loader):
        print(f"Батч {batch_idx}:")
        print(f"  Размер изображений: {images.shape}")
        print(f"  Размер лейблов: {labels.shape}")
        print(f"  Уникальные лейблы в батче: {torch.unique(labels).tolist()}")
        print(f"  Диапазон значений пикселей: [{images.min():.3f}, {images.max():.3f}]")
        break

    print("Информация о классах:")
    class_info = dataset.get_class_info()
    for digit in sorted(class_info.keys()):
        print(f"  Класс {digit}: {class_info[digit]} изображений")

    print("=== ТЕСТ ЗАВЕРШЕН ===")


if __name__ == "__main__":
    data_path = r"D:\abdulaziz\MyAI\dataset\MNIST - JPG - testing"
    test_dataset_loading(data_path)