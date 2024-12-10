import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter
import numpy as np

def load_image():
    global img_np
    file_path = filedialog.askopenfilename()
    if file_path:
        img_pil = Image.open(file_path)
        img_np = np.array(img_pil)
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

def apply_filter(filter_type):
    global img_np
    if img_np is None:
        return

    if filter_type in ("erosion", "dilation", "opening", "closing"):
        filtered_img = morphological_transform(img_np, filter_type)
    elif filter_type == "low_pass":
        img_pil = Image.fromarray(img_np).filter(ImageFilter.GaussianBlur(2))
        filtered_img = np.array(img_pil)
    elif filter_type == "high_pass":
        img_gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
        #Laplaciano na escala cinza
        img_laplacian = np.abs(np.gradient(np.gradient(img_gray)[0])[0])
        #Normalização
        img_laplacian = (img_laplacian - np.min(img_laplacian)) / (np.max(img_laplacian) - np.min(img_laplacian))
        #Intensidade 
        img_laplacian = img_laplacian * 255 
        #Conversão para RGB
        filtered_img = np.stack([img_laplacian] * 3, axis=-1)
        filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
    elif filter_type == "ideal_lowpass":
        img_gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        filtered_img = ideal_filter(img_gray, "lowpass", D0=40)
        filtered_img = np.stack([filtered_img] * 3, axis=-1).astype(np.uint8)
    elif filter_type == "ideal_highpass":
        img_gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
        filtered_img = ideal_filter(img_gray, "highpass", D0=40)
        filtered_img = np.stack([filtered_img] * 3, axis=-1).astype(np.uint8)
    elif filter_type == "threshold":
        filtered_img = thresholding(img_np)
    elif filter_type == "adaptive_threshold":
        filtered_img = adaptive_thresholding(img_np)
    else:
        raise ValueError("Unsupported filter type")

    display_image(filtered_img, original=False)


def morphological_transform(img_np, transform_type):
    
    gray_img = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])

 
    kernel = np.ones((3, 3), dtype=np.uint8)  

    if transform_type == "erosion":
        morphed_img = erosion(gray_img, kernel)
    elif transform_type == "dilation":
        morphed_img = dilation(gray_img, kernel)
    elif transform_type == "opening":
        morphed_img = opening(gray_img, kernel)
    elif transform_type == "closing":
        morphed_img = closing(gray_img, kernel)
    else:
        raise ValueError("Unsupported morphological operation")

    # Convert back to RGB and clip values
    morphed_img_rgb = np.stack([morphed_img] * 3, axis=-1).astype(np.uint8)
    return np.clip(morphed_img_rgb, 0, 255)

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
    filtered_img = np.clip(g, 0, 255)
    return filtered_img

def pad_image(image, kernel_size):
    pad_width = kernel_size // 2
    return np.pad(image, pad_width, mode='constant')

def erosion(image, kernel):
    padded_image = pad_image(image, kernel.shape[0])
    eroded_img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            eroded_img[i, j] = np.min(window * kernel)
    return eroded_img

def dilation(image, kernel):
    padded_image = pad_image(image, kernel.shape[0])
    dilated_img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            dilated_img[i, j] = np.max(window * kernel)
    return dilated_img

def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)

def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)

def thresholding(img_np):
    gray_img = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
    threshold = 100
    binary_img = (gray_img > threshold).astype(np.uint8) * 255
    return binary_img

def adaptive_thresholding(img_np):
    gray_img = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
    block_size = 11
    C = 2

    thresholded_img = np.zeros_like(gray_img)
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            local_region = gray_img[max(0, i - block_size // 2):min(i + block_size // 2 + 1, gray_img.shape[0]),
                                     max(0, j - block_size // 2):min(j + block_size // 2 + 1, gray_img.shape[1])]
            mean = np.mean(local_region)
            thresholded_img[i, j] = 255 if gray_img[i, j] > mean - C else 0

    thresholded_img_rgb = np.stack([thresholded_img] * 3, axis=-1).astype(np.uint8)
    return thresholded_img_rgb

def refresh_canvas():
    edited_image_canvas.delete("all")

root = tk.Tk()
root.title("App de processamento de Imagens")
root.geometry("1085x550")
root.config(bg="#2e2e2e")
img_np = None

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Arquico", menu=file_menu)
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
filters_menu.add_command(label="Thresholding", command=lambda: apply_filter("threshold"))
filters_menu.add_command(label="Thresholding Adaptativo", command=lambda: apply_filter("adaptive_threshold"))

original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

root.mainloop()
