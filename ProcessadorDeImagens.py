import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
import tkinter as tk
from tkinter import *
class ProcessadorDeImagens:
    def __init__(self, nome_img):
        self.nome_img = nome_img
        self.f = cv2.imread(nome_img, 0)
        self.F = np.fft.fft2(self.f)
        self.Fshift = np.fft.fftshift(self.F)
    def LowPassFlt_Ideal(self):
        M, N = self.f.shape
        H = np.zeros((M,N))
        D0 = 60
        for i in range(M):
            for j in range(N):
                D = pow(i-M/2,2) + pow(j-N/2,2)
                D = np.sqrt(D)
                if D <= D0:
                    H[i,j] = 1
                else:
                    H[i,j] = 0
        Gshift = self.Fshift * H
        G = np.fft.ifftshift(Gshift)
        g = np.abs(np.fft.ifft2(G))
        return g
    def HighPassFlt_Ideal(self):
        M, N = self.f.shape
        H = np.zeros((M,N))
        
        D0 = 60
        for i in range(M):
            for j in range(N):
                D = pow(i-M/2,2) + pow(j-N/2,2)
                D = np.sqrt(D)
                if D <= D0:
                    H[i,j] = 1
                else:
                    H[i,j] = 0
        H = 1 - H
        Gshift = self.Fshift * H
        G = np.fft.ifftshift(Gshift)
        g = np.abs(np.fft.ifft2(G))
        return g
    
def plot(img, img2, img3): 
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(img2, cmap='gray')
    axs[1].set_title('High-Pass')
    axs[2].imshow(img3, cmap='gray')
    axs[2].set_title('Low-Pass')

    for ax in axs:
        ax.axis('off')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack()

if __name__ == '__main__': 
    img = ProcessadorDeImagens('House.png')
    lowg = img.LowPassFlt_Ideal()
    highg = img.HighPassFlt_Ideal()
    window = Tk()
    window.title("App de processamento de imagens")
    window.geometry("800x600") 
    plot_button = Button(master = window,
                        command = lambda:plot(img.f,highg,lowg),  
                        height = 1, 
                        width = 5, 
                        text = "Plot") 
    plot_button.pack()
    window.mainloop()
    #plt.imshow(highg, cmap ='gray')
   