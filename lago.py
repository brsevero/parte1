import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('lago.jpg') #pegando a img
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #convetendo pra gray

borrao = cv.bilateralFilter(img,55,80,80)

plt.subplot(121),plt.imshow(img,"gray")
plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(borrao,"gray")
plt.title('Imagem Borrada'), plt.xticks([]), plt.yticks([])
plt.show()

h = cv.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title("Histograma do Lago")
plt.xlabel("Intensidade")
plt.ylabel("Qtde de Pixels")
plt.plot(h)
plt.xlim([0, 256])
plt.show()
cv.waitKey(0)

T,th1 = cv.threshold(img,57,255,cv.THRESH_BINARY_INV)

plt.subplot(121),plt.imshow(img,"gray")
plt.title('original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(th1,"gray")
plt.title('Limearizada'), plt.xticks([]), plt.yticks([])
plt.show()

bordas = cv.Canny(th1, 60, 120)

plt.subplot(121),plt.imshow(th1,"gray")
plt.title('Limearizada'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(bordas,"gray")
plt.title('Contornos'), plt.xticks([]), plt.yticks([])
plt.show()
