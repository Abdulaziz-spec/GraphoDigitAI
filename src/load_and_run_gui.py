import torch
from model import ConditionalVAE
from gui import DigitGUI
from pystyle import Colors, Colorate, Center
from colorama import init

init(autoreset=True)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_path = r"D:\abdulaziz\MyAI\outputs\checkpoints\model_epoch_1350.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_config' in checkpoint:
        latent_dim = checkpoint['model_config']['latent_dim']
        num_classes = checkpoint['model_config']['num_classes']
        print(Colorate.Horizontal(Colors.green_to_blue,
                                  f"[INFO] Модель из checkpoint: latent_dim={latent_dim}, num_classes={num_classes}"))
    else:
        fc_mu_weight_shape = checkpoint['model_state_dict']['fc_mu.weight'].shape
        latent_dim = fc_mu_weight_shape[0]
        num_classes = 10
        print(Colorate.Horizontal(Colors.yellow_to_red,
                                  f"[WARNING] Определили параметры из checkpoint: latent_dim={latent_dim}, num_classes={num_classes}"))

    model = ConditionalVAE(latent_dim=latent_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(Colorate.Horizontal(Colors.green_to_blue,
                              f"[SUCCESS] Модель успешно загружена с эпохи {checkpoint.get('epoch', 'Unknown')}"))
    print(Colorate.Horizontal(Colors.green_to_blue,
                              f"[INFO] Параметры модели: latent_dim={latent_dim}, num_classes={num_classes}"))

    print(Colorate.Horizontal(Colors.cyan_to_blue, Center.XCenter("Выберите режим работы:")))
    print(Colorate.Horizontal(Colors.cyan_to_blue, Center.XCenter("1 — Draw & Guess (рисовать)(beta)")))
    print(Colorate.Horizontal(Colors.cyan_to_blue, Center.XCenter("2 — Text -> Image (только генерация)")))

    mode = input("Введите 1 или 2: ").strip()

    if mode not in ['1', '2']:
        print(Colorate.Horizontal(Colors.yellow_to_red, "[WARNING] Неверный ввод. Используется режим 1."))
        mode = '1'

    gui = DigitGUI(model, device=device)
    gui.switch_mode(mode)
    gui.run()
