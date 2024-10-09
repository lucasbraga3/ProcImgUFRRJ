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

    if filter_type == "low_pass":
        img_pil = Image.fromarray(img_np).filter(ImageFilter.GaussianBlur(2))
        filtered_img = np.array(img_pil)
    elif filter_type == "high_pass":

        #Conversão para escala de cinza
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
        img_gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
        filtered_img = ideal_filter(img_gray, "lowpass", D0=40)
    elif filter_type == "ideal_highpass":
        img_gray = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
        filtered_img = ideal_filter(img_gray, "highpass", D0=40)

    display_image(filtered_img, original=False)

def ideal_filter(img_gray, filter_type, D0=40):
    # Aplicação da Transformada de Fourier e filtro ideal
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
    return np.stack([filtered_img]*3, axis=-1).astype(np.uint8)  # Convertendo para RGB

def refresh_canvas():
    edited_image_canvas.delete("all")

root = tk.Tk()
root.title("Image Processing App")
root.geometry("1085x550")
root.config(bg="#2e2e2e")
img_np = None

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)
filters_menu.add_command(label="Low Pass Filter", command=lambda: apply_filter("low_pass"))
filters_menu.add_command(label="High Pass Filter", command=lambda: apply_filter("high_pass"))
filters_menu.add_command(label="Ideal Low Pass Filter", command=lambda: apply_filter("ideal_lowpass"))
filters_menu.add_command(label="Ideal High Pass Filter", command=lambda: apply_filter("ideal_highpass"))

original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

root.mainloop()
