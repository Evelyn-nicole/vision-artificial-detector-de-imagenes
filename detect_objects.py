import cv2
import numpy as np

# Cargar la imagen.
image = cv2.imread('Image_flokito.jpg')

# Verifición carga correcta de la Imagen.
if image is None:
    print("Error: no se puede cargar la imagen.")
    exit()

# Ajuste pixeles imagen para ventanas emergentes.
scale_percent = 20  # Porcentaje de la escala.
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Mostrar la imagen en una ventana emergente.
cv2.imshow('Imagen Original', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertir la imagen a escala de grises.
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Detectación de bordes usando Canny.
edges = cv2.Canny(gray, 100, 200)

# Mostrar los bordes detectados en una ventana emergente.
cv2.imshow('Bordes Detectados', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Encontrar contornos.
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos.
cv2.drawContours(resized_image, contours, -1, (0, 255, 0), 2)

# Mostrar la imagen con los contornos dibujados en una ventana emergente.
cv2.imshow('Contornos Detectados', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen con contornos en un archivo nuevo.
cv2.imwrite('imagen_con_contornos.jpg', resized_image)
cv2.imwrite('imagen_con_bordes_detectados.jpg', edges)


