import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk, ImageFont
from skimage.morphology import skeletonize
from torchvision import transforms


class ModernDigitGUI:
    def __init__(self, model, device='cuda', model_dir=r"D:\abdulaziz\MyAI\outputs\checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir

        self.photo_refs = []
        self.photo_orig = []

        self.generated_size = 120
        self.canvas_size = 300
        self.brush_size = 20
        self.num_variants = 3
        self.variants_per_row = 6
        self.last_text_input = ""
        self.last_prediction = None
        self.eraser_mode = False
        self.confidence_threshold = 0.5
        self.show_vectors = False
        self.line_refs = []

        self.colors = {
            'bg_dark': '#0d1117',
            'bg_medium': '#161b22',
            'bg_light': '#21262d',
            'accent_blue': '#1f6feb',
            'accent_green': '#2ea043',
            'accent_purple': '#8b5cf6',
            'accent_orange': '#fb8500',
            'text_primary': '#f0f6fc',
            'text_secondary': '#8b949e',
            'border': '#30363d',
            'success': '#2ea043',
            'warning': '#fb8500',
            'error': '#f85149'
        }

        self.setup_ui()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("🤖 Neural Network Digit Generator")
        self.root.geometry("1400x900")
        self.root.configure(bg=self.colors['bg_dark'])
        self.root.resizable(True, True)

        self.setup_styles()

        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_rowconfigure(0, weight=1)
        self.create_control_panel()
        self.create_main_area()

        self.create_status_bar()

        self.switch_mode('1')

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Neural.TButton',
                        font=('Segoe UI', 10, 'bold'),
                        foreground=self.colors['text_primary'],
                        background=self.colors['accent_blue'],
                        borderwidth=0,
                        focuscolor='none',
                        relief='flat')
        style.map('Neural.TButton',
                  background=[('active', self.colors['accent_purple']),
                              ('pressed', self.colors['bg_medium'])])

        style.configure('Neural.TCombobox',
                        fieldbackground=self.colors['bg_light'],
                        background=self.colors['bg_light'],
                        foreground=self.colors['text_primary'],
                        borderwidth=1,
                        relief='solid')

    def create_control_panel(self):

        self.control_frame = tk.Frame(self.root, bg=self.colors['bg_medium'], width=350)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.control_frame.grid_propagate(False)

        title_frame = tk.Frame(self.control_frame, bg=self.colors['bg_medium'])
        title_frame.pack(fill='x', padx=20, pady=20)

        tk.Label(title_frame, text="🧠 NEURAL CONTROL",
                 font=('Segoe UI', 16, 'bold'),
                 bg=self.colors['bg_medium'],
                 fg=self.colors['accent_blue']).pack()

        mode_frame = tk.LabelFrame(self.control_frame, text="🎯 Operation Mode",
                                   font=('Segoe UI', 11, 'bold'),
                                   bg=self.colors['bg_medium'],
                                   fg=self.colors['text_primary'])
        mode_frame.pack(fill='x', padx=20, pady=10)

        self.mode_var = tk.StringVar(value='2')
        modes = [('1', '✏️ Draw & Recognize'), ('2', '🎨 Text to Image')]

        for value, text in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=value,
                           command=lambda: self.switch_mode(self.mode_var.get()),
                           font=('Segoe UI', 10),
                           bg=self.colors['bg_medium'],
                           fg=self.colors['text_primary'],
                           selectcolor=self.colors['accent_green'],
                           activebackground=self.colors['bg_light']).pack(anchor='w', padx=10, pady=5)

        input_frame = tk.LabelFrame(self.control_frame, text="📝 Input",
                                    font=('Segoe UI', 11, 'bold'),
                                    bg=self.colors['bg_medium'],
                                    fg=self.colors['text_primary'])
        input_frame.pack(fill='x', padx=20, pady=10)

        self.entry_digit = tk.Entry(input_frame, font=('Segoe UI', 12),
                                    bg=self.colors['bg_light'],
                                    fg=self.colors['text_primary'],
                                    insertbackground=self.colors['text_primary'],
                                    relief='flat', bd=5)
        self.entry_digit.pack(fill='x', padx=10, pady=10)
        self.entry_digit.bind('<Return>', lambda e: self.predict())

        params_frame = tk.LabelFrame(self.control_frame, text="⚙️ Generation Parameters",
                                     font=('Segoe UI', 11, 'bold'),
                                     bg=self.colors['bg_medium'],
                                     fg=self.colors['text_primary'])
        params_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(params_frame, text="Variants per digit:",
                 font=('Segoe UI', 10),
                 bg=self.colors['bg_medium'],
                 fg=self.colors['text_secondary']).pack(anchor='w', padx=10, pady=(10, 0))

        self.variants_scale = tk.Scale(params_frame, from_=1, to=30, orient='horizontal',
                                       bg=self.colors['bg_medium'],
                                       fg=self.colors['text_primary'],
                                       highlightthickness=0,
                                       troughcolor=self.colors['bg_light'])
        self.variants_scale.set(3)
        self.variants_scale.pack(fill='x', padx=10, pady=5)

        tk.Label(params_frame, text="Brush size:",
                 font=('Segoe UI', 10),
                 bg=self.colors['bg_medium'],
                 fg=self.colors['text_secondary']).pack(anchor='w', padx=10, pady=(10, 0))

        self.brush_scale = tk.Scale(params_frame, from_=5, to=50, orient='horizontal',
                                    bg=self.colors['bg_medium'],
                                    fg=self.colors['text_primary'],
                                    highlightthickness=0,
                                    troughcolor=self.colors['bg_light'])
        self.brush_scale.set(20)
        self.brush_scale.pack(fill='x', padx=10, pady=5)

        model_frame = tk.LabelFrame(self.control_frame, text="🤖 Model Selection",
                                    font=('Segoe UI', 11, 'bold'),
                                    bg=self.colors['bg_medium'],
                                    fg=self.colors['text_primary'])
        model_frame.pack(fill='x', padx=20, pady=10)

        def extract_epoch(f):
            m = re.search(r'epoch_(\d+)', f)
            return int(m.group(1)) if m else 0

        try:
            self.model_files = sorted([f for f in os.listdir(self.model_dir) if f.endswith(".pth")], key=extract_epoch)
            self.model_map = {f: os.path.join(self.model_dir, f) for f in self.model_files}

            self.model_var = tk.StringVar(value=self.model_files[-1] if self.model_files else "No models found")
            self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                            values=self.model_files,
                                            style='Neural.TCombobox',
                                            font=('Segoe UI', 10))
            self.model_combo.pack(fill='x', padx=10, pady=10)
        except:
            tk.Label(model_frame, text="No checkpoints folder found",
                     font=('Segoe UI', 10),
                     bg=self.colors['bg_medium'],
                     fg=self.colors['error']).pack(padx=10, pady=10)

        actions_frame = tk.Frame(self.control_frame, bg=self.colors['bg_medium'])
        actions_frame.pack(fill='x', padx=20, pady=20)

        buttons = [
            ("🚀 Generate/Predict", self.predict, self.colors['accent_green']),
            ("🔄 Load Model", self.load_model, self.colors['accent_blue']),
            ("♻  Refresh Models", self.refresh_model_list, self.colors['accent_blue']),
            ("🗑️ Clear All", self.clear_all, self.colors['error']),
            ("➕ More Variants", self.add_variants, self.colors['accent_orange']),
            ("💾 Export Images", self.export_sequence, self.colors['accent_purple']),
            ("🔧 Debug Info", self.show_debug_info, self.colors['text_secondary'])
        ]

        for text, command, color in buttons:
            btn = tk.Button(actions_frame, text=text, command=command,
                            font=('Segoe UI', 10, 'bold'),
                            bg=color, fg=self.colors['text_primary'],
                            relief='flat', bd=0, pady=8)
            btn.pack(fill='x', pady=3)

            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=self.colors['bg_light']))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))

    def create_main_area(self):

        self.main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        header_frame = tk.Frame(self.main_frame, bg=self.colors['bg_light'], height=80)
        header_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        header_frame.grid_propagate(False)

        self.result_frame = tk.Frame(header_frame, bg=self.colors['bg_light'])
        self.result_frame.pack(side='left', fill='both', expand=True, padx=20, pady=10)

        tk.Label(self.result_frame, text="🎯 Recognition Result:",
                 font=('Segoe UI', 12, 'bold'),
                 bg=self.colors['bg_light'],
                 fg=self.colors['text_secondary']).pack(anchor='w')

        self.label_main_digit = tk.Label(self.result_frame, text="—",
                                         font=('Segoe UI', 24, 'bold'),
                                         bg=self.colors['bg_light'],
                                         fg=self.colors['accent_green'])
        self.label_main_digit.pack(anchor='w')

        stats_frame = tk.Frame(header_frame, bg=self.colors['bg_light'])
        stats_frame.pack(side='right', fill='y', padx=20, pady=10)

        self.stats_label = tk.Label(stats_frame, text="📊 Model Stats:\nLatent Dim: —\nDevice: —",
                                    font=('Segoe UI', 10),
                                    bg=self.colors['bg_light'],
                                    fg=self.colors['text_secondary'],
                                    justify='left')
        self.stats_label.pack()

        content_frame = tk.Frame(self.main_frame, bg=self.colors['bg_dark'])
        content_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=2)
        content_frame.grid_rowconfigure(0, weight=1)

        self.draw_frame = tk.LabelFrame(content_frame, text="✏️ Drawing Canvas",
                                        font=('Segoe UI', 12, 'bold'),
                                        bg=self.colors['bg_medium'],
                                        fg=self.colors['text_primary'])
        self.draw_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        canvas_container = tk.Frame(self.draw_frame, bg=self.colors['bg_medium'])
        canvas_container.pack(expand=True, fill='both', padx=20, pady=20)

        self.canvas = tk.Canvas(canvas_container,
                                width=self.canvas_size,
                                height=self.canvas_size,
                                bg='white',
                                highlightthickness=2,
                                highlightbackground=self.colors['accent_blue'])
        self.canvas.pack(expand=True)

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.paint)

        tools_frame = tk.Frame(self.draw_frame, bg=self.colors['bg_medium'])
        tools_frame.pack(fill='x', padx=20, pady=10)

        self.eraser_btn = tk.Button(tools_frame, text="🧽 Eraser",
                                    command=self.toggle_eraser,
                                    font=('Segoe UI', 10),
                                    bg=self.colors['bg_light'],
                                    fg=self.colors['text_primary'])
        self.eraser_btn.pack(side='left', padx=5)

        tk.Button(tools_frame, text="🗑️ Clear",
                  command=self.clear_canvas,
                  font=('Segoe UI', 10),
                  bg=self.colors['error'],
                  fg=self.colors['text_primary']).pack(side='left', padx=5)

        self.skeleton_btn = tk.Button(tools_frame, text="🦴 Skeleton",
                                      command=self.show_skeleton_analysis,
                                      font=('Segoe UI', 10),
                                      bg=self.colors['accent_purple'],
                                      fg=self.colors['text_primary'])
        self.skeleton_btn.pack(side='left', padx=5)

        self.vectors_btn = tk.Button(tools_frame, text="📐 Vectors",
                                     command=self.toggle_vectors,
                                     font=('Segoe UI', 10),
                                     bg=self.colors['accent_orange'],
                                     fg=self.colors['text_primary'])
        self.vectors_btn.pack(side='left', padx=5)

        self.show_vectors = False

        self.gen_frame = tk.LabelFrame(content_frame, text="🎨 Generated Images",
                                       font=('Segoe UI', 12, 'bold'),
                                       bg=self.colors['bg_medium'],
                                       fg=self.colors['text_primary'])
        self.gen_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        canvas_frame = tk.Frame(self.gen_frame, bg=self.colors['bg_medium'])
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.generated_canvas = tk.Canvas(canvas_frame,
                                          bg=self.colors['bg_light'],
                                          highlightthickness=1,
                                          highlightbackground=self.colors['border'])
        self.scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.generated_canvas.yview)
        self.generated_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.generated_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.update_model_stats()

    def create_status_bar(self):
        self.status_frame = tk.Frame(self.root, bg=self.colors['bg_medium'], height=30)
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.status_frame.grid_propagate(False)

        self.status_label = tk.Label(self.status_frame, text="✅ Ready",
                                     font=('Segoe UI', 10),
                                     bg=self.colors['bg_medium'],
                                     fg=self.colors['text_secondary'])
        self.status_label.pack(side='left', padx=10, pady=5)

        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate')
        self.progress.pack(side='right', padx=10, pady=5)

    def update_status(self, text, show_progress=False):
        self.status_label.config(text=text)
        if show_progress:
            self.progress.start()
        else:
            self.progress.stop()
        self.root.update_idletasks()

    def update_model_stats(self):
        try:
            latent_dim = getattr(self.model, 'latent_dim', 'Unknown')
            num_classes = getattr(self.model, 'num_classes', 'Unknown')
            device = str(self.device).upper()
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            stats_text = f"🤖 MODEL INFO:\n"
            stats_text += f"Architecture: {type(self.model).__name__}\n"
            stats_text += f"Latent Dim: {latent_dim}\n"
            stats_text += f"Classes: {num_classes}\n"
            stats_text += f"Device: {device}\n"
            stats_text += f"Total Params: {total_params:,}\n"
            stats_text += f"Trainable: {trainable_params:,}\n"

            if device == 'CUDA' and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() // 1024 ** 2
                cached = torch.cuda.memory_reserved() // 1024 ** 2
                stats_text += f"GPU Mem: {allocated}MB/{cached}MB"

            self.stats_label.config(text=stats_text)
        except Exception as e:
            self.stats_label.config(text=f"📊 Model Stats:\n❌ Error: {str(e)}")

    def load_model(self):
        if not hasattr(self, 'model_combo'):
            messagebox.showerror("Error", "No models available")
            return

        filename = self.model_var.get()
        if filename not in self.model_map:
            messagebox.showerror("Error", f"File not found: {filename}")
            return

        self.update_status("🔄 Loading model...", True)

        try:
            path = self.model_map[filename]
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            model_config = checkpoint.get('model_config', {})
            current_latent = getattr(self.model, 'latent_dim', None)
            checkpoint_latent = model_config.get('latent_dim', None)

            if checkpoint_latent and current_latent and checkpoint_latent != current_latent:
                recreate = messagebox.askyesno(
                    "Model Compatibility",
                    f"Model architecture mismatch!\n"
                    f"Current latent dim: {current_latent}\n"
                    f"Checkpoint latent dim: {checkpoint_latent}\n\n"
                    f"Recreate model with correct parameters?"
                )
                if recreate:
                    from model import ConditionalVAE
                    self.model = ConditionalVAE(
                        latent_dim=checkpoint_latent,
                        num_classes=model_config.get('num_classes', 10)
                    ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            epoch = checkpoint.get('epoch', 'Unknown')
            loss = checkpoint.get('loss', 0)
            self.update_status(f"✅ Model loaded from epoch {epoch}")
            self.update_model_stats()

            messagebox.showinfo(
                "Success",
                f"Model loaded successfully!\nEpoch: {epoch}\nLoss: {loss:.4f}\nLatent Dim: {checkpoint_latent}\nDevice: {self.device}"
            )
        except Exception as e:
            self.update_status("❌ Failed to load model")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()

    def switch_mode(self, mode):
        self.mode = mode
        self.clear_all()

        if mode == '1':
            self.draw_frame.grid()
            self.entry_digit.config(state='disabled')
            self.update_status("✏️ Drawing mode activated")
        else:
            self.draw_frame.grid_remove()
            self.entry_digit.config(state='normal')
            self.entry_digit.focus()
            self.update_status("📝 Text mode activated")

    def paint(self, event):
        brush_size = self.brush_scale.get()
        x1, y1 = (event.x - brush_size // 2), (event.y - brush_size // 2)
        x2, y2 = (event.x + brush_size // 2), (event.y + brush_size // 2)

        if self.eraser_mode:
            color = 'white'
            fill_color = 255
        else:
            color = 'black'
            fill_color = 0

        self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)
        self.draw.ellipse([x1, y1, x2, y2], fill=fill_color)

    def toggle_eraser(self):
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.eraser_btn.config(text="✏️ Pencil", bg=self.colors['accent_orange'])
        else:
            self.eraser_btn.config(text="🧽 Eraser", bg=self.colors['bg_light'])

    def predict(self):
        if self.mode == '1':
            self.predict_drawing()
        else:
            self.predict_text()

    def preprocess_image_for_recognition(self, img):

        img_np = np.array(img)

        img_inverted = 255 - img_np

        kernel = np.ones((2, 2), np.uint8)
        img_cleaned = cv2.morphologyEx(img_inverted, cv2.MORPH_CLOSE, kernel)
        img_cleaned = cv2.morphologyEx(img_cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(img_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:

            largest_contour = max(contours, key=cv2.contourArea)

            mask = np.zeros_like(img_cleaned)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

            img_masked = cv2.bitwise_and(img_cleaned, mask)

            x, y, w, h = cv2.boundingRect(largest_contour)

            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_masked.shape[1] - x, w + 2 * padding)
            h = min(img_masked.shape[0] - y, h + 2 * padding)

            digit_roi = img_masked[y:y + h, x:x + w]

            max_dim = max(w, h)
            square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)

            start_x = (max_dim - w) // 2
            start_y = (max_dim - h) // 2
            square_img[start_y:start_y + h, start_x:start_x + w] = digit_roi

            processed_img = Image.fromarray(square_img)
        else:

            processed_img = Image.fromarray(img_inverted)

        return processed_img

    def predict_drawing(self):
        self.update_status("🤖 Analyzing drawing...", True)
        try:
            img_array = np.array(self.image)
            if np.all(img_array >= 250):
                messagebox.showwarning("Warning", "Please draw a digit first!")
                self.update_status("⚠️ No drawing detected")
                return

            processed_images = self._preprocess_drawing_multi(self.image)
            best_prediction = None
            best_confidence = 0
            all_results = []

            for i, processed_img in enumerate(processed_images):
                result = self._predict_single_image(processed_img)
                all_results.append(result)
                if result['confidence'] > best_confidence:
                    best_confidence = result['confidence']
                    best_prediction = result

            if best_prediction:
                self._display_prediction_results(best_prediction, all_results)
                self.generate_variants_from_latent(
                    best_prediction['digit'],
                    best_prediction['latent'],
                    best_prediction['labels']
                )
                self.last_prediction = best_prediction['digit']
                self.update_status(f"✅ Best prediction: {best_prediction['digit']} ({best_confidence * 100:.1f}%)")
            else:
                messagebox.showwarning("Warning", "Could not recognize the drawing")
                self.update_status("❌ Recognition failed")

        except Exception as e:
            self.update_status("❌ Recognition failed")
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _preprocess_drawing_multi(self, image):

        processed_images = []
        img_array = np.array(image)

        processed_images.append(self.preprocess_image_for_recognition(image))

        img_inv = 255 - img_array
        kernel = np.ones((3, 3), np.uint8)
        img_clean = cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel)
        img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_OPEN, kernel)
        processed_images.append(Image.fromarray(img_clean))

        img_blur = cv2.GaussianBlur(img_array, (3, 3), 0)
        processed_images.append(self.preprocess_image_for_recognition(Image.fromarray(img_blur)))

        return processed_images

    def _predict_single_image(self, processed_img):
        model_input_size = 64
        img_resized = processed_img.resize((model_input_size, model_input_size), Image.LANCZOS)
        img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.model.eval()
            labels = torch.zeros(1, self.model.num_classes, device=self.device)

            mu, logvar = self.model.encode(img_tensor, labels)
            z = self.model.reparameterize(mu, logvar)

            if hasattr(self.model, 'classifier'):
                class_pred = self.model.classifier(z)
                probs = torch.softmax(class_pred, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                top_predictions = torch.topk(probs[0], 3)
            else:
                similarities = []
                for digit in range(self.model.num_classes):
                    digit_label = torch.zeros(1, self.model.num_classes, device=self.device)
                    digit_label[0, digit] = 1
                    generated = self.model.decode(z, digit_label)
                    similarity = -torch.nn.functional.mse_loss(img_tensor, generated).item()
                    similarities.append((digit, similarity))

                similarities.sort(key=lambda x: x[1], reverse=True)
                pred_class = similarities[0][0]
                confidence = max(0, min(1, (similarities[0][1] + 1) / 2))
                top_predictions = (torch.tensor([s[1] for s in similarities[:3]]),
                                   torch.tensor([s[0] for s in similarities[:3]]))

        return {
            'digit': pred_class,
            'confidence': confidence,
            'latent': z,
            'labels': labels,
            'top_predictions': top_predictions,
            'processed_image': processed_img
        }

    def _display_prediction_results(self, best_result, all_results):
        result_text = f"🎯 Best Prediction: {best_result['digit']}\n"
        result_text += f"🔥 Confidence: {best_result['confidence'] * 100:.1f}%\n\n"

        if hasattr(best_result['top_predictions'][0], 'tolist'):
            values, indices = best_result['top_predictions']
            result_text += "🏆 Top 3:\n"
            for i in range(min(3, len(indices))):
                digit = indices[i].item()
                conf = values[i].item() if hasattr(values[i], 'item') else values[i]
                result_text += f"  {i + 1}. Digit {digit}: {conf * 100:.1f}%\n"

        result_text += f"\n📊 Tested {len(all_results)} preprocessing variants"
        self.label_main_digit.config(text=result_text)

    def generate_variants_from_latent(self, digit, z, labels):
        self.update_status(f"🎨 Generating variants for digit {digit}...", True)
        try:
            self.generated_canvas.delete("all")
            self.photo_refs = []

            num_variants = self.variants_scale.get()
            y_offset = 10
            x_offset = 10
            row_height = 0
            max_width = 0

            digit_labels = torch.zeros(1, self.model.num_classes, device=self.device)
            digit_labels[0, digit] = 1

            for v in range(num_variants):
                with torch.no_grad():

                    if v == 0:

                        z_variant = z
                    else:

                        noise_scale = 0.1
                        noise = torch.randn_like(z) * noise_scale
                        z_variant = z + noise

                    gen_img = self.model.decode(z_variant, digit_labels)
                    pil_img = self.tensor_to_pil(gen_img)
                    photo = ImageTk.PhotoImage(pil_img)

                if x_offset + self.generated_size > 600:
                    x_offset = 10
                    y_offset += row_height + 10
                    row_height = 0

                self.generated_canvas.create_rectangle(
                    x_offset - 2, y_offset - 2,
                    x_offset + self.generated_size + 2,
                    y_offset + self.generated_size + 2,
                    outline=self.colors['border'], width=2
                )

                self.generated_canvas.create_image(x_offset, y_offset, anchor='nw', image=photo)

                label_text = f"#{v + 1}" if v > 0 else "Original"
                self.generated_canvas.create_text(
                    x_offset + 5, y_offset + 5,
                    text=label_text,
                    font=('Segoe UI', 8, 'bold'),
                    fill=self.colors['accent_blue']
                )

                self.photo_refs.append(photo)
                x_offset += self.generated_size + 10
                row_height = max(row_height, self.generated_size)
                max_width = max(max_width, x_offset)

            y_offset += row_height + 20
            self.generated_canvas.configure(scrollregion=(0, 0, max_width, y_offset))
            self.update_status(f"✅ Generated {num_variants} variants for digit {digit}")

        except Exception as e:
            self.update_status("❌ Variant generation failed")
            messagebox.showerror("Error", f"Variant generation failed: {str(e)}")

    def show_skeleton_analysis(self):
        try:

            img_array = np.array(self.image)
            if np.all(img_array == 255):
                messagebox.showwarning("Warning", "Please draw something first!")
                return

            self.update_status("🦴 Performing skeleton analysis...", True)

            skeleton_window = tk.Toplevel(self.root)
            skeleton_window.title("🦴 Skeleton Analysis")
            skeleton_window.geometry("800x600")
            skeleton_window.configure(bg=self.colors['bg_dark'])

            main_frame = tk.Frame(skeleton_window, bg=self.colors['bg_dark'])
            main_frame.pack(fill='both', expand=True, padx=10, pady=10)

            title_label = tk.Label(main_frame,
                                   text="🦴 SKELETON & VECTOR ANALYSIS",
                                   font=('Segoe UI', 16, 'bold'),
                                   bg=self.colors['bg_dark'],
                                   fg=self.colors['accent_blue'])
            title_label.pack(pady=10)

            images_frame = tk.Frame(main_frame, bg=self.colors['bg_dark'])
            images_frame.pack(fill='both', expand=True)

            orig_frame = tk.LabelFrame(images_frame,
                                       text="📝 Original Drawing",
                                       font=('Segoe UI', 12, 'bold'),
                                       bg=self.colors['bg_medium'],
                                       fg=self.colors['text_primary'])
            orig_frame.pack(side='left', fill='both', expand=True, padx=5)

            orig_img = self.image.resize((250, 250))
            orig_photo = ImageTk.PhotoImage(orig_img)
            orig_canvas = tk.Canvas(orig_frame, width=260, height=260,
                                    bg='white', highlightthickness=1,
                                    highlightbackground=self.colors['border'])
            orig_canvas.pack(padx=10, pady=10)
            orig_canvas.create_image(5, 5, anchor='nw', image=orig_photo)

            skel_frame = tk.LabelFrame(images_frame,
                                       text="🦴 Skeletonized",
                                       font=('Segoe UI', 12, 'bold'),
                                       bg=self.colors['bg_medium'],
                                       fg=self.colors['text_primary'])
            skel_frame.pack(side='right', fill='both', expand=True, padx=5)

            img_64 = np.array(self.image.resize((64, 64)))
            img_bin = (img_64 < 128).astype(np.uint8)

            if np.sum(img_bin) > 0:

                skeleton = skeletonize(img_bin)

                skel_img = (skeleton * 255).astype(np.uint8)
                skel_pil = Image.fromarray(skel_img).resize((250, 250))

                ys, xs = np.where(skeleton > 0)
                points = list(zip(xs, ys))

                analysis_text = f"🔍 SKELETON ANALYSIS:\n\n"
                analysis_text += f"📊 Total skeleton pixels: {len(points)}\n"

                if points:

                    path = self.build_skeleton_path(points)
                    analysis_text += f"🛤️ Path length: {len(path)} points\n"

                    turns, turn_angles = self.analyze_path_turns(path)
                    analysis_text += f"🔄 Sharp turns detected: {turns}\n"
                    analysis_text += f"📐 Average turn angle: {np.mean(turn_angles):.1f}°\n"

                    geometric_guess = self.geometric_digit_guess(turns, len(path), turn_angles)
                    analysis_text += f"\n🎯 Geometric guess: {geometric_guess}\n"

                    vector_img = self.create_vector_visualization(skeleton, path, turn_angles)
                    vector_pil = Image.fromarray(vector_img).resize((250, 250))
                    vector_photo = ImageTk.PhotoImage(vector_pil)
                else:
                    analysis_text += "\n⚠️ No skeleton path found"
                    vector_photo = ImageTk.PhotoImage(skel_pil)
            else:
                analysis_text = "⚠️ No drawing detected for analysis"
                skel_pil = Image.new('L', (250, 250), 255)
                vector_photo = ImageTk.PhotoImage(skel_pil)

            skel_photo = ImageTk.PhotoImage(skel_pil)
            skel_canvas = tk.Canvas(skel_frame, width=260, height=260,
                                    bg='white', highlightthickness=1,
                                    highlightbackground=self.colors['border'])
            skel_canvas.pack(padx=10, pady=10)
            skel_canvas.create_image(5, 5, anchor='nw', image=vector_photo)

            analysis_frame = tk.LabelFrame(main_frame,
                                           text="📊 Analysis Results",
                                           font=('Segoe UI', 12, 'bold'),
                                           bg=self.colors['bg_medium'],
                                           fg=self.colors['text_primary'])
            analysis_frame.pack(fill='x', pady=10)

            analysis_label = tk.Label(analysis_frame,
                                      text=analysis_text,
                                      font=('Segoe UI', 10),
                                      bg=self.colors['bg_medium'],
                                      fg=self.colors['text_primary'],
                                      justify='left')
            analysis_label.pack(padx=20, pady=10, anchor='w')

            buttons_frame = tk.Frame(main_frame, bg=self.colors['bg_dark'])
            buttons_frame.pack(fill='x', pady=10)

            tk.Button(buttons_frame, text="💾 Save Skeleton",
                      command=lambda: self.save_skeleton_image(skel_pil),
                      font=('Segoe UI', 10, 'bold'),
                      bg=self.colors['accent_green'],
                      fg=self.colors['text_primary']).pack(side='left', padx=5)

            tk.Button(buttons_frame, text="🔄 Refresh Analysis",
                      command=lambda: [skeleton_window.destroy(), self.show_skeleton_analysis()],
                      font=('Segoe UI', 10, 'bold'),
                      bg=self.colors['accent_blue'],
                      fg=self.colors['text_primary']).pack(side='left', padx=5)

            tk.Button(buttons_frame, text="❌ Close",
                      command=skeleton_window.destroy,
                      font=('Segoe UI', 10, 'bold'),
                      bg=self.colors['error'],
                      fg=self.colors['text_primary']).pack(side='right', padx=5)

            skeleton_window.orig_photo = orig_photo
            skeleton_window.skel_photo = skel_photo
            skeleton_window.vector_photo = vector_photo

            self.update_status("✅ Skeleton analysis complete")

        except Exception as e:
            self.update_status("❌ Skeleton analysis failed")
            messagebox.showerror("Error", f"Skeleton analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def build_skeleton_path(self, points):
        if not points:
            return []

        path = [points[0]]
        points_set = set(points)
        points_set.remove(points[0])

        while points_set:
            last = path[-1]

            nearest = min(points_set,
                          key=lambda p: (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2)
            path.append(nearest)
            points_set.remove(nearest)

        return path

    def analyze_path_turns(self, path):
        if len(path) < 3:
            return 0, []

        turns = 0
        turn_angles = []

        for i in range(2, len(path)):

            prev_vec = np.array([path[i - 1][0] - path[i - 2][0],
                                 path[i - 1][1] - path[i - 2][1]])
            curr_vec = np.array([path[i][0] - path[i - 1][0],
                                 path[i][1] - path[i - 1][1]])

            prev_norm = np.linalg.norm(prev_vec)
            curr_norm = np.linalg.norm(curr_vec)

            if prev_norm > 0 and curr_norm > 0:
                cos_angle = np.dot(prev_vec, curr_vec) / (prev_norm * curr_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)

                turn_angles.append(angle_deg)

                if angle_deg > 30:
                    turns += 1

        return turns, turn_angles

    def geometric_digit_guess(self, turns, path_length, turn_angles):
        avg_angle = np.mean(turn_angles) if turn_angles else 0

        if turns == 0:
            return "1 (straight line)"
        elif turns == 1:
            return "7 or L-shape"
        elif turns == 2:
            if avg_angle > 60:
                return "2, 3, or 5"
            else:
                return "4 or 7"
        elif turns >= 3:
            if path_length > 30:
                return "0, 6, 8, or 9 (complex curve)"
            else:
                return "3, 5, or 6"
        else:
            return "Unknown pattern"

    def create_vector_visualization(self, skeleton, path, turn_angles):

        h, w = skeleton.shape
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)

        vis_img.fill(255)

        vis_img[skeleton > 0] = [128, 128, 128]

        if len(path) < 2:
            return vis_img

        for i in range(1, len(path)):
            x0, y0 = path[i - 1]
            x1, y1 = path[i]

            if i - 1 < len(turn_angles):
                angle = turn_angles[i - 1]
                if angle < 15:
                    color = [0, 255, 0]
                elif angle < 45:
                    color = [255, 255, 0]
                elif angle < 90:
                    color = [255, 165, 0]
                else:
                    color = [255, 0, 0]
            else:
                color = [0, 255, 0]

            if 0 <= x0 < w and 0 <= y0 < h:
                vis_img[y0, x0] = color
            if 0 <= x1 < w and 0 <= y1 < h:
                vis_img[y1, x1] = color

        return vis_img

    def save_skeleton_image(self, skeleton_img):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
                title="Save Skeleton Image"
            )
            if filename:
                skeleton_img.save(filename)
                messagebox.showinfo("Success", f"Skeleton saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save skeleton: {str(e)}")

    def toggle_vectors(self):
        self.show_vectors = not self.show_vectors

        if self.show_vectors:
            self.analyze_drawing_vectors()
            self.vectors_btn.config(text="🚫 Hide Vectors", bg=self.colors['error'])
        else:

            if hasattr(self, 'line_refs'):
                for line in self.line_refs:
                    self.canvas.delete(line)
                self.line_refs = []
            self.vectors_btn.config(text="📐 Vectors", bg=self.colors['accent_orange'])
        if hasattr(self, 'line_refs'):
            for line in self.line_refs:
                self.canvas.delete(line)
        self.line_refs = []

        img = np.array(self.image.resize((64, 64)))
        img_bin = (img < 128).astype(np.uint8)

        if np.sum(img_bin) == 0:
            return

        try:
            skel = skeletonize(img_bin).astype(np.uint8) * 255
            ys, xs = np.where(skel > 0)
            points = list(zip(xs, ys))
            if not points:
                return

            path = [points[0]]
            points_set = set(points)
            points_set.remove(points[0])

            while points_set:
                last = path[-1]
                nearest = min(points_set, key=lambda p: (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2)
                path.append(nearest)
                points_set.remove(nearest)

            scale = self.canvas_size / 64
            turn_count = 0

            for i in range(1, len(path)):
                x0, y0 = path[i - 1]
                x1, y1 = path[i]
                x0, y0 = x0 * scale, y0 * scale
                x1, y1 = x1 * scale, y1 * scale

                if i > 1:
                    prev_vec = np.array([path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1]])
                    curr_vec = np.array([path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]])

                    prev_norm = np.linalg.norm(prev_vec)
                    curr_norm = np.linalg.norm(curr_vec)

                    if prev_norm > 0 and curr_norm > 0:
                        cos_angle = np.dot(prev_vec, curr_vec) / (prev_norm * curr_norm)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)

                        if angle > 0.5:
                            turn_count += 1
                            color = 'red'
                        else:
                            color = 'green'
                    else:
                        angle = 0
                        color = 'green'
                else:
                    color = 'green'

                line = self.canvas.create_line(x0, y0, x1, y1, fill=color, width=2)
                self.line_refs.append(line)

        except Exception as e:
            print(f"Vector analysis error: {e}")

    def analyze_drawing_vectors(self):
        try:
            img_array = np.array(self.image)
            if np.all(img_array >= 250):
                messagebox.showwarning("Warning", "Please draw a digit first!")
                self.update_status("⚠️ No drawing detected")
                return

            processed_images = [
                self.preprocess_image_for_recognition(self.image),
                Image.fromarray(cv2.GaussianBlur(img_array, (3, 3), 0)),
                Image.fromarray(cv2.morphologyEx(255 - img_array, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
            ]

            best_prediction = None
            best_confidence = 0

            for proc_img in processed_images:
                model_input_size = 64
                img_resized = proc_img.resize((model_input_size, model_input_size), Image.LANCZOS)
                img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    self.model.eval()
                    labels = torch.zeros(1, self.model.num_classes, device=self.device)
                    z, _ = self.model.encode(img_tensor, labels)

                    if hasattr(self.model, 'classify'):
                        logits = self.model.classify(z, labels)
                        probs = torch.softmax(logits, dim=1)
                        pred_class = torch.argmax(probs, dim=1).item()
                        confidence = probs[0, pred_class].item()
                    else:
                        similarities = []
                        for digit in range(self.model.num_classes):
                            digit_label = torch.zeros(1, self.model.num_classes, device=self.device)
                            digit_label[0, digit] = 1
                            generated = self.model.decode(z, digit_label)
                            similarity = -torch.nn.functional.mse_loss(img_tensor, generated).item()
                            similarities.append((digit, similarity))
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        pred_class = similarities[0][0]
                        confidence = max(0, min(1, (similarities[0][1] + 1) / 2))

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_prediction = (pred_class, z, labels)

            if best_prediction:
                self.last_prediction = best_prediction[0]
                self.update_status(f"✅ Vector analysis: predicted {best_prediction[0]} ({best_confidence * 100:.1f}%)")
            else:
                self.update_status("❌ Vector analysis failed")
                return

        except Exception as e:
            print(f"Prediction error: {e}")
            self.update_status("❌ Vector analysis failed")
            return

        try:
            if hasattr(self, 'line_refs'):
                for line in self.line_refs:
                    self.canvas.delete(line)
            self.line_refs = []

            img_small = np.array(self.image.resize((64, 64)))
            img_bin = (img_small < 128).astype(np.uint8)

            if np.sum(img_bin) == 0:
                return

            skel = skeletonize(img_bin).astype(np.uint8) * 255
            ys, xs = np.where(skel > 0)
            points = list(zip(xs, ys))
            if not points:
                return

            path = [points[0]]
            points_set = set(points)
            points_set.remove(points[0])

            while points_set:
                last = path[-1]
                nearest = min(points_set, key=lambda p: (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2)
                path.append(nearest)
                points_set.remove(nearest)

            scale = self.canvas_size / 64
            for i in range(1, len(path)):
                x0, y0 = path[i - 1]
                x1, y1 = path[i]
                x0, y0 = x0 * scale, y0 * scale
                x1, y1 = x1 * scale, y1 * scale

                if i > 1:
                    prev_vec = np.array([path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1]])
                    curr_vec = np.array([path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]])
                    prev_norm = np.linalg.norm(prev_vec)
                    curr_norm = np.linalg.norm(curr_vec)
                    if prev_norm > 0 and curr_norm > 0:
                        cos_angle = np.dot(prev_vec, curr_vec) / (prev_norm * curr_norm)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        color = 'red' if angle > 0.5 else 'green'
                    else:
                        color = 'green'
                else:
                    color = 'green'

                line = self.canvas.create_line(x0, y0, x1, y1, fill=color, width=2)
                self.line_refs.append(line)

        except Exception as e:
            print(f"Vector drawing error: {e}")

    def tensor_to_pil(self, gen_img):
        if isinstance(gen_img, torch.Tensor):
            gen_img = gen_img.detach().cpu()

        if gen_img.ndim == 4:
            gen_img = gen_img[0, 0]
        elif gen_img.ndim == 3:
            gen_img = gen_img[0]
        elif gen_img.ndim == 2 and gen_img.shape[1] == 4096:
            gen_img = gen_img.view(-1, 64, 64)[0]
        elif gen_img.ndim == 1 and gen_img.numel() == 4096:
            gen_img = gen_img.view(64, 64)
        elif gen_img.ndim == 1:
            size = int(np.sqrt(gen_img.numel()))
            if size * size == gen_img.numel():
                gen_img = gen_img.view(size, size)
            else:
                raise ValueError(f"Cannot reshape tensor with {gen_img.numel()} elements to square image")

        gen_img = gen_img.numpy()

        if np.issubdtype(gen_img.dtype, np.floating):

            if gen_img.min() >= -1 and gen_img.max() <= 1:
                if gen_img.min() < 0:
                    gen_img = (gen_img + 1) / 2

                gen_img = np.clip(gen_img * 255, 0, 255)
            else:

                gen_img = gen_img - gen_img.min()
                if gen_img.max() > 0:
                    gen_img = gen_img / gen_img.max() * 255
                gen_img = np.clip(gen_img, 0, 255)

        gen_img = gen_img.astype(np.uint8)

        pil_img = Image.fromarray(gen_img, mode='L')
        pil_img = pil_img.resize((self.generated_size, self.generated_size), Image.LANCZOS)
        return pil_img

    def predict_text(self):
        text = self.entry_digit.get().strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some digits")
            return

        if not text.replace(' ', '').isdigit():
            messagebox.showwarning("Warning", "Please enter only digits (0-9)")
            return

        self.last_text_input = text
        self.generate_variants(text)

    def generate_variants(self, text):
        self.update_status("🎨 Generating images...", True)
        self.generated_canvas.delete("all")
        self.photo_refs = []

        num_variants = self.variants_scale.get()
        y_offset = 10
        max_width = 0

        for char in text.replace(' ', ''):
            if not char.isdigit():
                continue
            digit = int(char)
            x_offset = 10
            row_height = 0

            self.generated_canvas.create_text(
                x_offset, y_offset,
                text=f"Digit: {digit}",
                font=('Segoe UI', 12, 'bold'),
                fill=self.colors['text_primary'],
                anchor='nw'
            )
            y_offset += 25

            for v in range(num_variants):

                onehot_gen = torch.zeros(1, self.model.num_classes, device=self.device)
                onehot_gen[0, digit] = 1

                if v == 0:
                    z = torch.zeros(1, self.model.latent_dim, device=self.device)
                else:
                    z = torch.randn(1, self.model.latent_dim, device=self.device) * 0.5

                with torch.no_grad():
                    gen_img = self.model.decode(z, onehot_gen)
                    pil_img = self.tensor_to_pil(gen_img)
                    photo = ImageTk.PhotoImage(pil_img)

                if x_offset + self.generated_size > 600:
                    x_offset = 10
                    y_offset += row_height + 10
                    row_height = 0

                self.generated_canvas.create_rectangle(
                    x_offset - 2, y_offset - 2,
                    x_offset + self.generated_size + 2,
                    y_offset + self.generated_size + 2,
                    outline=self.colors['border'], width=2
                )

                self.generated_canvas.create_image(x_offset, y_offset, anchor='nw', image=photo)

                self.generated_canvas.create_text(
                    x_offset + 5, y_offset + 5,
                    text=f"#{v + 1}",
                    font=('Segoe UI', 8, 'bold'),
                    fill=self.colors['accent_blue']
                )

                self.photo_refs.append(photo)

                x_offset += self.generated_size + 10
                row_height = max(row_height, self.generated_size)
                max_width = max(max_width, x_offset)

            y_offset += row_height + 30

        self.generated_canvas.configure(scrollregion=(0, 0, max_width, y_offset))
        self.update_status(f"✅ Generated {num_variants} variants for each digit")

    def add_variants(self):
        current_variants = self.variants_scale.get()
        max_variants = self.variants_scale['to']
        if current_variants < max_variants:
            self.variants_scale.set(current_variants + 1)

        if self.mode == '2' and self.last_text_input:
            self.generate_variants(self.last_text_input)
        elif self.mode == '1' and hasattr(self, 'last_prediction'):

            self.predict_drawing()

    def clear_all(self):
        self.clear_canvas()
        self.generated_canvas.delete("all")
        self.photo_refs = []
        self.label_main_digit.config(text="—")
        self.last_text_input = ""
        if hasattr(self, 'line_refs'):
            for line in self.line_refs:
                self.canvas.delete(line)
            self.line_refs = []
        self.update_status("🗑️ Cleared all")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def show_debug_info(self):
        try:
            info = f"🔧 DEBUG INFORMATION\n\n"
            info += f"Model Type: {type(self.model).__name__}\n"
            info += f"Device: {self.device}\n"
            info += f"Latent Dimension: {getattr(self.model, 'latent_dim', 'Unknown')}\n"
            info += f"Number of Classes: {getattr(self.model, 'num_classes', 'Unknown')}\n"
            info += f"Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n"
            info += f"Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n\n"

            if hasattr(self, 'photo_refs'):
                info += f"Generated Images: {len(self.photo_refs)}\n"

            info += f"Canvas Size: {self.canvas_size}x{self.canvas_size}\n"
            info += f"Generated Image Size: {self.generated_size}x{self.generated_size}\n"
            info += f"Current Mode: {'Drawing' if self.mode == '1' else 'Text Generation'}\n"
            info += f"Variants per Digit: {self.variants_scale.get()}\n"
            info += f"Brush Size: {self.brush_scale.get()}\n"
            info += f"Eraser Mode: {self.eraser_mode}\n"

            if self.device == 'cuda' and torch.cuda.is_available():
                info += f"\n🖥️ GPU INFO:\n"
                info += f"GPU Name: {torch.cuda.get_device_name()}\n"
                info += f"Total Memory: {torch.cuda.get_device_properties(0).total_memory // 1024 ** 3} GB\n"
                info += f"Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB\n"
                info += f"Cached: {torch.cuda.memory_reserved() // 1024 ** 2} MB\n"

            messagebox.showinfo("Debug Information", info)

        except Exception as e:
            messagebox.showerror("Debug Error", f"Failed to collect debug info: {str(e)}")

    def export_sequence(self):
        if not hasattr(self, 'photo_orig') or not self.photo_orig:
            messagebox.showwarning("Warning", "No generated images to export")
            return

        try:
            export_window = tk.Toplevel(self.root)
            export_window.title("💾 Export Options")
            export_window.geometry("400x300")
            export_window.configure(bg=self.colors['bg_dark'])
            export_window.transient(self.root)
            export_window.grab_set()

            tk.Label(export_window, text="📁 Export Generated Images",
                     font=('Segoe UI', 14, 'bold'),
                     bg=self.colors['bg_dark'],
                     fg=self.colors['accent_blue']).pack(pady=10)

            options_frame = tk.LabelFrame(export_window, text="📋 Export Options",
                                          font=('Segoe UI', 11, 'bold'),
                                          bg=self.colors['bg_medium'],
                                          fg=self.colors['text_primary'])
            options_frame.pack(fill='x', padx=20, pady=10)

            tk.Label(options_frame, text="Format:", bg=self.colors['bg_medium'], fg=self.colors['text_secondary']).pack(
                anchor='w', padx=10, pady=(10, 0))
            format_var = tk.StringVar(value='combined')
            format_options = [('combined', '🖼️ Single Combined Image'),
                              ('separate', '📁 Separate Images'),
                              ('both', '📂 Both Formats')]
            for value, text in format_options:
                tk.Radiobutton(options_frame, text=text, variable=format_var, value=value,
                               bg=self.colors['bg_medium'], fg=self.colors['text_primary'],
                               selectcolor=self.colors['accent_green']).pack(anchor='w', padx=20, pady=2)

            tk.Label(options_frame, text="Image Quality:", bg=self.colors['bg_medium'],
                     fg=self.colors['text_secondary']).pack(anchor='w', padx=10, pady=(10, 0))
            quality_var = tk.IntVar(value=95)
            tk.Scale(options_frame, from_=50, to=100, orient='horizontal', variable=quality_var,
                     bg=self.colors['bg_medium'], fg=self.colors['text_primary'], highlightthickness=0).pack(fill='x',
                                                                                                             padx=10,
                                                                                                             pady=5)

            tk.Label(options_frame, text="Export Size:", bg=self.colors['bg_medium'],
                     fg=self.colors['text_secondary']).pack(anchor='w', padx=10, pady=(10, 0))
            size_var = tk.StringVar(value='original')
            size_frame = tk.Frame(options_frame, bg=self.colors['bg_medium'])
            size_frame.pack(fill='x', padx=10, pady=5)
            for value, text in [('original', 'Original'), ('256', '256x256'), ('512', '512x512')]:
                tk.Radiobutton(size_frame, text=text, variable=size_var, value=value,
                               bg=self.colors['bg_medium'], fg=self.colors['text_primary'],
                               selectcolor=self.colors['accent_green']).pack(side='left', padx=10)

            def do_export():
                export_window.destroy()
                self._perform_export(format_var.get(), quality_var.get(), size_var.get())

            buttons_frame = tk.Frame(export_window, bg=self.colors['bg_dark'])
            buttons_frame.pack(fill='x', padx=20, pady=20)
            tk.Button(buttons_frame, text="💾 Export", command=do_export,
                      font=('Segoe UI', 11, 'bold'), bg=self.colors['accent_green'],
                      fg=self.colors['text_primary']).pack(side='left', padx=5)
            tk.Button(buttons_frame, text="❌ Cancel", command=export_window.destroy,
                      font=('Segoe UI', 11, 'bold'), bg=self.colors['error'],
                      fg=self.colors['text_primary']).pack(side='right', padx=5)

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to open export dialog: {str(e)}")

    def _perform_export(self, format_type, quality, size_option):
        try:
            self.update_status("💾 Exporting images...", True)
            export_size = self.generated_size if size_option == 'original' else int(size_option)
            digits_text = self.last_text_input.replace(' ', '') if self.last_text_input else str(
                getattr(self, 'last_prediction', 0))
            num_variants = self.variants_scale.get()

            if format_type in ['combined', 'both']:
                self._export_combined(digits_text, num_variants, export_size, quality)
            if format_type in ['separate', 'both']:
                self._export_separate(digits_text, num_variants, export_size, quality)

            self.update_status("✅ Export completed successfully")
        except Exception as e:
            self.update_status("❌ Export failed")
            messagebox.showerror("Export Error", f"Export failed: {str(e)}")

    def _export_combined(self, digits_text, num_variants, export_size, quality):
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
                                                title="Save Combined Image")
        if not filename:
            return

        images_per_row = min(6, num_variants)
        rows_per_digit = (num_variants + images_per_row - 1) // images_per_row
        total_width = images_per_row * export_size + (images_per_row - 1) * 20 + 40
        total_height = len(digits_text) * rows_per_digit * (export_size + 60) + 40
        combined = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(combined)

        y_pos = 20
        img_idx = 0
        for char in digits_text:
            if not char.isdigit(): continue
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            draw.text((20, y_pos), f"Digit: {char}", fill='black', font=font)
            y_pos += 40

            x_offset = 20
            row_height = 0
            for v in range(num_variants):
                if img_idx >= len(self.photo_orig): break
                try:
                    img_pil = self.photo_orig[img_idx].resize((export_size, export_size), Image.LANCZOS)
                    if x_offset + export_size > total_width:
                        x_offset = 20
                        y_pos += row_height + 20
                        row_height = 0
                    combined.paste(img_pil, (x_offset, y_pos))
                    x_offset += export_size + 20
                    row_height = max(row_height, export_size)
                except Exception as e:
                    print(f"Error processing image {img_idx}: {e}")
                img_idx += 1
            y_pos += row_height + 40

        if filename.lower().endswith(('.jpg', '.jpeg')):
            combined.save(filename, 'JPEG', quality=quality, optimize=True)
        else:
            combined.save(filename, 'PNG', optimize=True)

        messagebox.showinfo("Success", f"Combined image saved to:\n{filename}")

    def _export_separate(self, digits_text, num_variants, export_size, quality):
        folder = filedialog.askdirectory(title="Select Folder for Separate Images")
        if not folder: return

        img_idx = 0
        saved_count = 0
        for char in digits_text:
            if not char.isdigit(): continue
            for v in range(num_variants):
                if img_idx >= len(self.photo_orig): break
                try:
                    img_resized = self.photo_orig[img_idx].resize((export_size, export_size), Image.LANCZOS)
                    filename = f"digit_{char}_variant_{v + 1:02d}.png"
                    filepath = os.path.join(folder, filename)
                    img_resized.save(filepath, 'PNG', optimize=True)
                    saved_count += 1
                except Exception as e:
                    print(f"Error saving image {img_idx}: {e}")
                img_idx += 1

        messagebox.showinfo("Success", f"Saved {saved_count} separate images to:\n{folder}")

    def refresh_model_list(self):
        try:
            def extract_epoch(f):
                m = re.search(r'epoch_(\d+)', f)
                return int(m.group(1)) if m else 0

            model_files = sorted(
                [f for f in os.listdir(self.model_dir) if f.endswith(".pth")],
                key=extract_epoch
            )

            if not model_files:
                messagebox.showwarning("Warning", "No models found in the directory")
                return

            self.model_files = model_files
            self.model_map = {f: os.path.join(self.model_dir, f) for f in model_files}

            if hasattr(self, 'model_combo'):
                self.model_combo['values'] = self.model_files
                self.model_var.set(self.model_files[-1])

            self.update_status("✅ Model list refreshed")
            messagebox.showinfo("Success", f"Found {len(model_files)} models")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh model list: {str(e)}")
            self.update_status("❌ Failed to refresh model list")

    def run(self):
        self.update_status("🚀 Neural Network GUI Ready")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("Fatal Error", f"GUI crashed: {str(e)}")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            try:
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.root.quit()
                self.root.destroy()
            except:
                pass

    @staticmethod
    def image_to_tensor(pil_image, size=64, device='cuda'):
        img_resized = pil_image.resize((size, size)).convert('L')
        img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)
        return img_tensor

    def predict_digit_probabilities(self, pil_image):
        self.model.eval()
        img_tensor = self.image_to_tensor(pil_image, device=self.device)
        labels = torch.zeros(1, self.model.num_classes, device=self.device)

        with torch.no_grad():
            z, _ = self.model.encode(img_tensor, labels)
            if hasattr(self.model, 'classify'):
                logits = self.model.classify(z, labels)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                probs = np.zeros(self.model.num_classes)

        return probs, z.cpu().numpy()

    def display_digit_probabilities(self, probs):

        text = " | ".join([f"{i}: {p * 100:.1f}%" for i, p in enumerate(probs)])
        self.label_main_digit.config(text=f"NN Probabilities:\n{text}")
        top_idx = np.argmax(probs)
        self.update_status(f"→ Most likely digit: {top_idx} ({probs[top_idx] * 100:.1f}%)")


class DigitGUI(ModernDigitGUI):
    pass
