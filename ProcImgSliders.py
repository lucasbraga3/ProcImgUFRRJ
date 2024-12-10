import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2 

def load_image():
  global img_np
  file_path = filedialog.askopenfilename()
  if file_path:
    img_bgr = cv2.imread(file_path)
    img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Converter para RGB afim de exibir corretamente
    display_image(img_np, original=True)
    refresh_canvas()

def display_image(img, original=False):
  img_pil = Image.fromarray(img)
  img_width, img_height = img_pil.size
  max_size = 500
  img_pil.thumbnail((max_size, max_size))
  img_tk = ImageTk.PhotoImage(img_pil)
  canvas_width, canvas_height = max_size, max_size
  x_offset = (canvas_width - img_pil.width) // 2
  y_offset = (canvas_height - img_pil.height) // 2

  if original:
    original_image_canvas.delete("all")
    original_image_canvas.image = img_tk
    original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
  else:
    edited_image_canvas.delete("all")
    edited_image_canvas.image = img_tk
    edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def apply_filter(filter_type, value=None):
  global img_np
  if img_np is None:
    return

  # Mudar parametros de filtros de acordo com o valor do slider
  if filter_type == "low_pass":
    kernel_size = value if value is not None else 3  # Tamanho padrao de kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
    filtered_img = cv2.filter2D(img_np, -1, kernel)
  elif filter_type == "high_pass":
    ksize = value if value is not None else 1  # Tamanho padrao de kernel
    if ksize % 2 == 0:  # Detalhe: Kernel deve ser impar
      ksize += 1
    filtered_img = cv2.Laplacian(img_np, cv2.CV_64F, ksize=ksize)
    filtered_img = cv2.convertScaleAbs(filtered_img)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
  elif filter_type == "ideal_lowpass":
    D0 = value if value is not None else 40  # Frequencia de corte padrao
    filtered_img = np.zeros_like(img_np)
    for i in range(3):  # Aplicar Filtro a cada canal de cor
      channel = img_np[:, :, i]
      filtered_channel = ideal_filter(channel, "lowpass", D0)
      filtered_img[:, :, i] = filtered_channel
  elif filter_type == "ideal_highpass":
    D0 = value if value is not None else 40  # Frequencia de corte padrao
    filtered_img = np.zeros_like(img_np)
    for i in range(3):  # Aplicar Filtro a cada canal de cor
      channel = img_np[:, :, i]
      filtered_channel = ideal_filter(channel, "highpass", D0)
      filtered_img[:, :, i] = filtered_channel
  elif filter_type in ("erosion", "dilation", "opening", "closing"):
    kernel_size = value if value is not None else 3  # Tamanho padrao de kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_img = np.zeros_like(img_np)
    for i in range(3):  # Aplicar operacao morfologica a cada canal de cor
      channel = img_np[:, :, i]
      if filter_type == "erosion":
        filtered_channel = cv2.erode(channel, kernel)
      elif filter_type == "dilation":
        filtered_channel = cv2.dilate(channel, kernel)
      elif filter_type == "opening":
        filtered_channel = cv2.morphologyEx(channel, cv2.MORPH_OPEN, kernel)
      elif filter_type == "closing":
        filtered_channel = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)
      filtered_img[:, :, i] = filtered_channel
  elif filter_type == "thresholding":
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    threshold_value = value if value is not None else 127  # threshold padrao
    ret, thresh = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    filtered_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
  elif filter_type == "adaptive_thresholding":
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    block_size = value if value is not None else 11 # Tamanho do bloco padrao
    C = 2
    if block_size % 2 == 0: # Tamanho do bloco deve ser impar
      block_size += 1
    thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    filtered_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
  else:
    raise ValueError("Unsupported filter type")

  display_image(filtered_img, original=False)

def ideal_filter(img_gray, filter_type, D0=40):
  F = np.fft.fft2(img_gray)
  Fshift = np.fft.fftshift(F)
  M, N = img_gray.shape
  H = np.zeros((M, N))
  for i in range(M):
    for j in range(N):
      D = np.sqrt((i - M / 2) ** 2 + (j - N / 2) ** 2)
      if filter_type == "lowpass":
        H[i, j] = 1 if D <= D0 else 0
      elif filter_type == "highpass":
        H[i, j] = 0 if D <= D0 else 1
  Gshift = Fshift * H
  G = np.fft.ifftshift(Gshift)
  g = np.abs(np.fft.ifft2(G))
  filtered_img = np.clip(g, 0, 255).astype(np.uint8)
  return filtered_img

def create_slider_frame():
  slider_frame = tk.Frame(root, bg="#2e2e2e")
  slider_frame.grid(row=0, column=3, columnspan=2, sticky="we")

  slider_labels = [
      ("Tamanho do Kernel Passa-Baixa:", 3, 21, 3, "low_pass"),
      ("Tamanho do Kernel Passa-Alta:", 1, 31, 1, "high_pass"), 
      ("Frequencia de corte Passa-Baixa Ideal(pode demorar):", 10, 100, 40, "ideal_lowpass"),
      ("Frequencia de corte Passa-Alta Ideal(pode demorar):", 10, 100, 40, "ideal_highpass"),
      ("Tamanho do Kernel Erosao:", 3, 11, 3, "erosion"),
      ("Tamanho do Kernel Dilatacao:", 3, 11, 3, "dilation"),
      ("Tamanho do Kernel Abertura", 3, 11, 3, "opening"),
      ("Tamanho do Kernel Fechamento:", 3, 11, 3, "closing"),
      ("Valor Threshold:", 0, 255, 127, "thresholding"),
      ("Tamanho do Bloco Threshold Adaptativo:", 3, 21, 11, "adaptive_thresholding"),
  ]

  for i, (label_text, min_value, max_value, initial_value, filter_type) in enumerate(slider_labels):
    label = tk.Label(slider_frame, text=label_text, bg="#2e2e2e", fg="white")
    label.grid(row=i, column=0, sticky="w")

    slider = tk.Scale(slider_frame, from_=min_value, to=max_value, orient=tk.HORIZONTAL, bg="#2e2e2e", fg="white", highlightthickness=0, troughcolor="#444444", sliderlength=20, command=lambda value, f_type=filter_type: apply_filter(f_type, int(value)))
    slider.set(initial_value)
    slider.grid(row=i, column=1, sticky="we")

def refresh_canvas():
    edited_image_canvas.delete("all")

root = tk.Tk()
root.title("App de processamento de Imagens")
root.geometry("2000x550")
root.config(bg="#2e2e2e")
img_np = None

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Arquivo", menu=file_menu)
file_menu.add_command(label="Carregar Imagem", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filtros simples", menu=filters_menu)
filters_menu.add_command(label="Filtro Passa-Baixa", command=lambda: apply_filter("low_pass"))
filters_menu.add_command(label="Filtro Passa-Alta", command=lambda: apply_filter("high_pass"))
filters_menu.add_command(label="Filtro Passa-Baixa Ideal", command=lambda: apply_filter("ideal_lowpass"))
filters_menu.add_command(label="Filtro Passa-Alta Ideal", command=lambda: apply_filter("ideal_highpass"))
filters_menu.add_command(label="Erosao", command=lambda: apply_filter("erosion"))
filters_menu.add_command(label="Dilatacao", command=lambda: apply_filter("dilation"))
filters_menu.add_command(label="Abertura", command=lambda: apply_filter("opening"))
filters_menu.add_command(label="Fechamento", command=lambda: apply_filter("closing"))
filters_menu.add_command(label="Thresholding", command=lambda: apply_filter("thresholding"))
filters_menu.add_command(label="Thresholding Adaptativo", command=lambda: apply_filter("adaptive_thresholding"))

original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=10)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=10)


create_slider_frame()
root.mainloop()
