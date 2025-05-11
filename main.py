"""
Микро Stable Diffusion - Полная реализация с теорией и комментариями
"""

# Теория: 
# Диффузионные модели работают по принципу двухэтапного процесса:
# 1. Прямой процесс (диффузия) - постепенно добавляем шум к изображению
# 2. Обратный процесс - обучаем нейросеть восстанавливать изображение из шума
# Мы реализуем упрощённую версию для работы на слабом железе
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
import requests
import time
import torchvision.transforms as transforms

from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageTk
from tqdm import tqdm
from datetime import datetime
from io import BytesIO

def train_model_with_mixed_precision(model, dataloader, diffusion, criterion, optimizer, device, target_loss=0.02, max_epochs=150):
    """
    Обучает модель с использованием mixed precision (FP16)
    
    :param model: нейросеть (TinyUNet или улучшенная версия)
    :param dataloader: загрузчик датасета
    :param diffusion: процесс диффузии
    :param criterion: функция потерь (MSE)
    :param optimizer: оптимизатор
    :param device: устройство (cuda/cpu)
    :param target_loss: целевое значение loss (~98% точности)
    :param max_epochs: максимальное число эпох
    """
    print("🚀 Запуск обучения с Mixed Precision (FP16)")
    
        # Определяем допустимые расширения для изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # Получаем список файлов и папок в указанной директории
    files = os.listdir(output_dir)

    # Фильтруем только файлы с изображениями
    image_count = sum(1 for f in files if os.path.isfile(os.path.join(output_dir, f)) and os.path.splitext(f)[1].lower() in image_extensions)

    batches_per_epoch = image_count // 4
    total_batches = batches_per_epoch * max_epochs
    total_time_sec = total_batches * 0.4
    print(f"📊 Прогноз времени:")
    print(f"  • {image_count} изображений")
    print(f"  • {max_epochs} эпох")
    print(f"  • {4} изображений за проход")
    print(f"  • ~0.4 секунд на батч")
    print(f"⏱️ Общее время: {total_time_sec:.2f} секунд (~{total_time_sec//60} мин)")

    # Инициализация GradScaler для FP16
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')
    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(max_epochs):
        total_loss = 0
        model.train()  # Переключаем модель в режим обучения
        
        # Вывод информации о текущей эпохе
        print(f"🔁 Начинается Эпоха {epoch + 1} / {max_epochs}")
        
        for images, _ in dataloader:
            images = images.to(device)
            t = torch.randint(0, diffusion.T, (images.shape[0],)).to(device)

            with autocast():  # Включаем автоматическое управление точностью
                x_t, noise = diffusion.forward(images, t)
                predicted_noise = model(x_t, t)
                loss = criterion(predicted_noise, noise)

            # Обновляем веса модели
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"⏱️ Обучение заняло {total_duration:.2f} секунд")

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Эпоха {epoch + 1} завершена | Средний Loss: {avg_loss:.4f}")

        # Сохраняем лучшую модель
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "tiny_diffusion.pth")
            print(f"💾 Модель сохранена с Loss: {best_loss:.4f}")

        # Проверяем достижение целевого качества
        if avg_loss < target_loss:
            print(f"🎯 Целевой loss ({target_loss}) достигнут на эпохе {epoch + 1}")
            break

    print("🎉 Обучение завершено!")

def download_images_from_ddgs_results(results, output_dir="dataset/street_art", image_size=(128, 128), max_count=2000):
    """
    Скачивает и ресайзит изображения из результатов DDGS.images()
    Продолжает попытки, пока не будет скачано max_count изображений.
    """
    os.makedirs(output_dir, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Определяем допустимые расширения для изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # Получаем список файлов и папок в указанной директории
    files = os.listdir(output_dir)

    # Фильтруем только файлы с изображениями
    image_count = sum(1 for f in files if os.path.isfile(os.path.join(output_dir, f)) and os.path.splitext(f)[1].lower() in image_extensions)
    
    downloaded = image_count
    attempts = 0
    max_attempts_per_image = 1  # Максимум попыток на одно изображение

    print(f"🔍 Пытаюсь скачать {max_count} изображений...")

    while downloaded < max_count:
        for idx, item in enumerate(results):
            if downloaded >= max_count:
                break

            img_url = item["image"]
            success = False
            attempt = 0

            while not success and attempt < max_attempts_per_image:
                try:
                    response = requests.get(img_url, headers=headers, timeout=5)
                    response.raise_for_status()

                    # Загружаем изображение
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    img = img.resize(image_size)

                    # Генерируем имя файла
                    filename = f"{downloaded:04d}_{item['title'][:50].replace(' ', '_')}.jpg"
                    safe_filename = "".join(c for c in filename if c.isalnum() or c in ("_", ".", "-"))
                    save_path = os.path.join(output_dir, safe_filename)

                    # Сохраняем
                    img.save(save_path)
                    downloaded += 1
                    print(f"[OK] Скачано: {safe_filename}")
                    success = True

                except Exception as e:
                    attempt += 1
                    print(f"[Ошибка] Не удалось скачать: {img_url} | Причина: {e} | Попытка {attempt}/{max_attempts_per_image}")

            attempts += 1
            time.sleep(1)  # Небольшая пауза между запросами

    print(f"✅ Успешно скачано {downloaded} изображений в '{output_dir}'")

# 1. Архитектура модели
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = self.conv_block(3, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Decoder
        self.decoder3 = self.upconv_block(64, 32)
        self.decoder2 = self.upconv_block(32 + 32, 16)  # skip connection
        self.final = nn.Sequential(
            nn.ConvTranspose2d(16 + 16, 3, kernel_size=2, stride=2),
            nn.Tanh()
        )  # Чтобы выход был [-1, 1]

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, t):
        # print(f"Input shape: {x.shape}")  # [batch, 3, 128, 128]

        # Encoder
        e1 = self.encoder1(x)
        # print(f"e1 shape: {e1.shape}")  # [batch, 16, 64, 64]

        e2 = self.encoder2(e1)
        # print(f"e2 shape: {e2.shape}")  # [batch, 32, 32, 32]

        e3 = self.encoder3(e2)
        # print(f"e3 shape: {e3.shape}")  # [batch, 64, 16, 16]

        # Middle
        m = self.middle(e3)
        # print(f"Middle shape: {m.shape}")  # [batch, 64, 16, 16]

        # Decoder 3
        d3 = self.decoder3(m)
        # print(f"d3 shape: {d3.shape}")  # [batch, 32, 32, 32]

        # Skip connection
        # print(f"e2 shape: {e2.shape}")  # [batch, 32, 32, 32]
        d3 = torch.cat([d3, e2], dim=1)
        # print(f"After cat(d3, e2): {d3.shape}")  # [batch, 64, 32, 32]

        # Decoder 2
        d2 = self.decoder2(d3)
        # print(f"d2 shape: {d2.shape}")  # [batch, 16, 64, 64]

        # Skip connection
        # print(f"e1 shape: {e1.shape}")  # [batch, 16, 64, 64]
        d2 = torch.cat([d2, e1], dim=1)
        # print(f"After cat(d2, e1): {d2.shape}")  # [batch, 32, 64, 64]

        # Final layer to get back to input size
        output = self.final(d2)
        # print(f"Final output shape: {output.shape}")  # [batch, 3, 128, 128]

        return output

# 2. Текстовый энкодер
class TextEncoder:
    """
    Преобразует текстовые запросы в числовые векторы
    
    Используем TF-IDF для простоты реализации
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=128)
        
    def fit(self, texts):
        """Обучение на текстовых описаниях"""
        self.vectorizer.fit(texts)
        
    def encode(self, text):
        """Преобразование текста в вектор"""
        return torch.tensor(self.vectorizer.transform([text]).toarray()).float()

# 3. Процесс диффузии
class DiffusionProcess:
    """
    Реализация диффузионного процесса
    
    Теория:
    Прямой процесс диффузии: q(x_t | x_{t-1}) = N(x_t; sqrt(α_t)x_{t-1}, (1-α_t)I)
    Обратный процесс: p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ^2I)
    """
    def __init__(self, T=150):
        self.T = T
        self.betas = torch.linspace(0.0001, 0.02, T)  # Шум на каждом шаге
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)  # Кумулятивное произведение для общего шума

    def forward(self, x, t):
        """
        Прямой процесс: добавляем шум к изображению
        x: исходное изображение
        t: временной шаг
        """
        noise = torch.randn_like(x)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hats[t]).view(-1, 1, 1, 1)  # Расширяем размерность
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hats[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample(self, model, n, size):
        """
        Обратный процесс: генерация из шума
        model: обученная модель
        n: количество изображений
        size: размер изображения
        """
        device = next(model.parameters()).device
        x = torch.randn(n, 3, *size).to(device)

        for t in reversed(range(self.T)):
            noise_pred = model(x, t)
            alpha = self.alphas[t]
            alpha_hat = self.alpha_hats[t]

            x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * noise_pred)
            if t > 0:
                x += torch.sqrt(self.betas[t]) * torch.randn_like(x)

        return x.clamp(-1, 1)

# 4. Подготовка датасета
class ArtDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)  # ← Здесь используется Compose(), а не модуль
        
        return image, "street art"
    
    def __len__(self):
        return len(self.files)

# 5. Обучение модели
def train_model(target_loss=0.1, max_epochs=100):
    """
    Обучает модель, пока loss не станет меньше target_loss или не закончатся эпохи.
    
    Параметры:
    - target_loss: целевое значение потерь (например, 0.02 ≈ 98% "точности")
    - max_epochs: максимальное число эпох обучения
    """
    print("🚀 Запуск обучения...")

    # Трансформации изображений
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ArtDataset("dataset/street_art", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Инициализация модели и компонентов
    model = TinyUNet()
    diffusion = DiffusionProcess(T=100)
    text_encoder = TextEncoder()
    text_encoder.fit(["street art", "graffiti", "urban art"])

    # Устройство для вычислений
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Загрузка существующей модели, если она есть
    if os.path.exists("tiny_diffusion.pth"):
        model.load_state_dict(torch.load("tiny_diffusion.pth"))
        print("✅ Продолжаем обучение существующей модели")

    train_model_with_mixed_precision(model, dataloader, diffusion, criterion, optimizer, device, target_loss=target_loss, max_epochs=max_epochs)

    print("🎉 Обучение завершено!")

# 6. Графический интерфейс
class DiffusionGUI(ctk.CTk):
    """
    Графический интерфейс для генерации изображений
    
    Возможности:
    - Ввод текстового запроса
    - Генерация изображения
    - Отображение результата
    """
    def __init__(self):
        super().__init__()
        
        # Проверка наличия обученной модели
        if not os.path.exists("tiny_diffusion.pth"):
            print("Модель не найдена! Запуск обучения...")
            train_model()
        
        # Параметры окна
        self.title("Микро Stable Diffusion")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Компоненты интерфейса
        self.create_widgets()
        
        # Инициализация модели
        self.init_model()
    
    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Заголовок
        self.title_label = ctk.CTkLabel(
            self, 
            text="Микро Stable Diffusion", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=20)
        
        # Поле ввода
        self.prompt_entry = ctk.CTkEntry(
            self, 
            placeholder_text="Введите описание...",
            width=400
        )
        self.prompt_entry.pack(pady=10)
        
        # Кнопка генерации
        self.generate_button = ctk.CTkButton(
            self, 
            text="Сгенерировать",
            command=self.generate_image
        )
        self.generate_button.pack(pady=10)
        
        # Холст для изображения
        self.canvas = ctk.CTkCanvas(
            self, 
            width=512, 
            height=512,
            bg="#2b2b2b"
        )
        self.canvas.pack(pady=20)
    
    def init_model(self):
        """Инициализация модели и компонентов"""
        self.model = TinyUNet()
        self.model.load_state_dict(torch.load("tiny_diffusion.pth"))
        self.diffusion = DiffusionProcess()
        
        # Перевод модели в режим оценки
        self.model.eval()
        
        # Определение устройства
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def generate_image(self):
        prompt = self.prompt_entry.get()
        print(f"Генерация изображения для запроса: {prompt}")
        with torch.no_grad():
            generated = self.diffusion.sample(self.model, 1, (64, 64))
        image = transforms.ToPILImage()(generated[0].cpu() * 0.5 + 0.5)
        display_image = image.resize((512, 512))

        os.makedirs("generated", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("generated", f"{timestamp}_{prompt.replace(' ', '_')}.png")
        image.save(filename)

        self.tk_image = ImageTk.PhotoImage(display_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.image = self.tk_image
        print("Изображение сгенерировано!")


if __name__ == "__main__":
    output_dir = "dataset/street_art"
    # Шаг 1: Поиск изображений через DuckDuckGo
    # results = DDGS().images(
    #     keywords="street art graffiti urban art",
    #     region="wt-wt",
    #     safesearch="on"
    # )

    # Шаг 2: Скачивание найденных изображений
    # download_images_from_ddgs_results(
    #     results,
    #     output_dir=output_dir,
    #     image_size=(128, 128),
    #     max_count=2000
    # )

    # Шаг 3: Создаём трансформации правильно
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ArtDataset(output_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Шаг 4: Инициализация модели
    model = TinyUNet()
    diffusion = DiffusionProcess(T=100)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Шаг 5: Обучение с Mixed Precision
    # train_model_with_mixed_precision(model, dataloader, diffusion, criterion, optimizer, device, target_loss=0.02, max_epochs=150)

    # Шаг 6: Запуск GUI
    app = DiffusionGUI()
    app.mainloop()