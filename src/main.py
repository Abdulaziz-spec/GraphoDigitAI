import sys

import torch

from dataset_loader import MNISTDataset, MNISTDatasetWithAugmentation
from gui import DigitGUI
from model import ConditionalVAE
from trainer import Trainer

if __name__ == "__main__":
    print("=== Запуск main.py ===")

    data_path = r"D:\abdulaziz\MyAI\dataset\MNIST - JPG - testing"

    """
    Path to your dataset THIS CODE CAN ANALYSE ONLY .jpg OTHER CAN'T !!!!!!!!!!!!!!!!!!!!
    """

    print(f"Используем датасет: {data_path}")

    print("Введите параметры обучения:")
    batch_size = int(input("  Размер батча (рекомендуется 512-1024): "))

    device_input = input("  Использовать GPU? (y/n): ").lower()
    device = 'cuda' if device_input == 'y' and torch.cuda.is_available() else 'cpu'
    print(f"  Используем device: {device}")

    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // 1024 ** 3} GB")

    epochs = int(input("  Количество эпох: "))
    checkpoint_interval = int(input("  Через сколько эпох сохранять чекпоинт: "))

    steps_per_epoch_input = input("  Количество шагов в эпохе (Enter = весь датасет): ")
    steps_per_epoch = int(steps_per_epoch_input) if steps_per_epoch_input else None

    use_augmentation = input("  Использовать аугментацию данных? (y/n): ").lower() == 'y'

    print("=== Подготовка DataLoader ===")
    if use_augmentation:
        print("  Используем датасет с аугментацией")
        dataset = MNISTDatasetWithAugmentation(data_path, batch_size=batch_size, use_augmentation=True)
    else:
        print("  Используем базовый датасет")
        dataset = MNISTDataset(data_path, batch_size=batch_size)

    data_loader = dataset.get_loader()
    print(f"  Датасет загружен, всего {len(data_loader.dataset)} изображений")

    class_info = dataset.get_class_info()
    print("  Распределение по классам:")
    for digit in sorted(class_info.keys()):
        print(f"    Класс {digit}: {class_info[digit]} изображений")

    print("=== Инициализация модели ===")
    latent_dim = int(input("  Размерность латентного пространства (рекомендуется 16-32): "))
    learning_rate = float(input("  Learning rate (рекомендуется 1e-3): "))

    model = ConditionalVAE(latent_dim=latent_dim, num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Общее количество параметров: {total_params:,}")
    print(f"  Обучаемых параметров: {trainable_params:,}")

    print("=== Начало тренировки ===")
    trainer = Trainer(model, data_loader, device=device, lr=learning_rate)

    checkpoint_path = input("Введите путь к чекпоинту для продолжения (Enter = пропустить): ")
    start_epoch = 1
    if checkpoint_path:
        start_epoch = trainer.load_checkpoint(checkpoint_path)

    try:
        trainer.train(
            epochs=epochs,
            checkpoint_interval=checkpoint_interval,
            steps_per_epoch=steps_per_epoch,
            start_epoch=start_epoch
        )
        print("=== Тренировка завершена успешно ===")
    except KeyboardInterrupt:
        print("\n=== Тренировка прервана пользователем ===")
        trainer.save_checkpoint(start_epoch - 1, 0, is_final=False)
    except Exception as e:
        print(f"=== Ошибка во время тренировки: {e} ===")

    print("=== Запуск GUI для тестирования модели ===")
    start_gui = input("Запустить GUI? (y/n): ").lower()
    if start_gui == 'y':
        gui = DigitGUI(model, device=device)
        gui.run()
    else:
        print("GUI не запущен. Завершение программы.")

    print("=== Программа завершена ===")


def test_dataset_only():
    data_path = r"D:\abdulaziz\MyAI\dataset\MNIST - JPG - testing"
    print("=== ТЕСТИРОВАНИЕ ДАТАСЕТА ===")

    basic_dataset = MNISTDataset(data_path, batch_size=32)
    basic_loader = basic_dataset.get_loader()
    for images, labels in basic_loader:
        print(f"   Размер батча: {images.shape}")
        print(f"   Диапазон пикселей: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   Уникальные лейблы: {torch.unique(labels).tolist()}")
        break

    aug_dataset = MNISTDatasetWithAugmentation(data_path, batch_size=32, use_augmentation=True)
    aug_loader = aug_dataset.get_loader()
    for images, labels in aug_loader:
        print(f"   Размер батча: {images.shape}")
        print(f"   Диапазон пикселей: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   Уникальные лейблы: {torch.unique(labels).tolist()}")
        break

    print("=== ТЕСТ ЗАВЕРШЕН ===")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "test":
    test_dataset_only()