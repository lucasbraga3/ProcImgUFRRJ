import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

def load_image():
    global img_cv
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        display_image(img_cv, original=True)
        refresh_canvas()

def display_image(img, original=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
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
        edited_image_canvas.delete("all")  # Limapa a canvas
        edited_image_canvas.image = img_tk
        edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def apply_filter(filter_type):
    if img_cv is None:
        return
    if filter_type == "low_pass":
        filtered_img = cv2.GaussianBlur(img_cv, (15, 15), 0)
    elif filter_type == "high_pass":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.Laplacian(gray, cv2.CV_64F)
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    elif filter_type == "ideal_lowpass":
        f = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        F = np.fft.fft2(f)
        Fshift = np.fft.fftshift(F)
        M, N = f.shape
        H = np.zeros((M,N))
        D0 = 40
        for i in range(M):
            for j in range(N):
                D = np.sqrt((i - M / 2) ** 2 + (j - N / 2) ** 2)
                if D <= D0:
                    H[i, j] = 1
                else:
                    H[i, j] = 0
        Gshift = Fshift * H
        G = np.fft.ifftshift(Gshift)
        g = np.abs(np.fft.ifft2(G))
        filtered_img = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        filtered_img = cv2.cvtColor(filtered_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif filter_type == "ideal_highpass":
        f = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        F = np.fft.fft2(f)
        Fshift = np.fft.fftshift(F)
        M, N = f.shape
        H = np.zeros((M,N))
        D0 = 40 
        for i in range(M):
            for j in range(N):
                D = np.sqrt((i - M / 2) ** 2 + (j - N / 2) ** 2)
                if D <= D0:
                    H[i,j] = 1
                else:
                    H[i,j] = 0
        H = 1 - H 
        Gshift = Fshift * H
        G = np.fft.ifftshift(Gshift)
        g = np.abs(np.fft.ifft2(G))
        filtered_img = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX) 
        filtered_img = cv2.cvtColor(filtered_img.astype(np.uint8), cv2.COLOR_GRAY2BGR) 
    display_image(filtered_img, original=False)

def refresh_canvas():
    edited_image_canvas.delete("all") 
root = tk.Tk()
root.title("Image Processing App")
root.geometry("1085x550")
root.config(bg="#2e2e2e")
img_cv = None
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
