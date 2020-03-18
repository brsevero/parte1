import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('moedas.jpg') #pegando a img
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #convetendo pra gray

#fazendo um borrão simples
borrao = cv.blur(img,(9,9))

T,thresh1 = cv.threshold(img,115,255,cv.THRESH_BINARY_INV) #faz uma limearização binaria com a imagem original
T,thresh2 = cv.threshold(borrao,115,255,cv.THRESH_BINARY_INV) #A mesma coisa usando a imagem borrada

#funcão que retorna uma img com contornos
lap = cv.Laplacian(thresh2, cv.CV_64F) 
lap = np.uint8(np.absolute(lap))

plt.subplot(141),plt.imshow(img,'gray')
plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(borrao,'gray')
plt.title('Imagem borrao'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(thresh1,'gray')
plt.title('Imagem Limeada original'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(thresh2,'gray')
plt.title('Imagem Limeada borrao'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(thresh2,'gray')
plt.title('Imagem Limeada borrao'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(lap,"gray")
plt.title('laplace'), plt.xticks([]), plt.yticks([])
plt.show()

"""In OpenCV, finding contours is like finding white object from black background.
So remember, object to be found should be white and background should be black."""

#função que retorna a identificação dos contornos
lista, hierarquia = cv.findContours(lap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#função para desenhar os contornos
final = cv.drawContours(img, lista, -1, (0,255,0), 3)
final = cv.resize(final, (550, 700))  
cv.imshow("ultima",final)
cv.waitKey(0)
print("quantidade de objetos na foto: " + str(len(lista)))
 