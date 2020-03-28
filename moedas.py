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
"""In OpenCV, finding contours is like finding white object from black background.
So remember, object to be found should be white and background should be black."""

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
plt.title('Laplace'), plt.xticks([]), plt.yticks([])
plt.show()

#função que retorna a identificação dos contornos
lista, hierarquia = cv.findContours(lap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #2 ultimos parametros são importantes

#função para desenhar os contornos
final = cv.drawContours(img, lista, -1, (0,255,0), 3)

print("quantidade de objetos na foto: " + str(len(lista)))

total = 0 #variavel para enumerar 
areas = 0 #variavel de total do valor
for i in lista:
    total += 1
    M = cv.moments(i) #estrutura usada pra pegar detalhes de contornos

    #centroide é dado pela relação cx = M10/M00 e cy = M01/M00 definido pelo opencv
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    posicao = (cx-150,cy)

    #area
    area = cv.contourArea(i)
    if (area >= 20000.0 and area <= 30000.0):
        areas += 0.10
    elif area > 30000.0 and area <= 35000.0:
        areas += 0.05
    else:
        areas += 0.25

    #perimetro
    perimeter = cv.arcLength(i,True)

    cv.putText(final,
                '{} {}'.format(total,area),
                posicao,
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                255,
                3)

    print("Objeto {}:\ncentro:({},{})\narea: {}\nperimetro {}\n".format(total,cx,cy,area,perimeter))

plt.imshow(final,cmap=plt.cm.gray)
plt.axis('off')
plt.show()

print("Temos {} R$ na foto".format(areas))