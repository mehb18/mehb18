- 👋 Hi, I’m @mehb18
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
mehb18/mehb18 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->cambia este codigo para que ejecute los mismos plots, pero cambiando la estructura el orden(por jemplo poner while i eso):
import matplotlib.pyplot as plt
import math as math
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

presión = 101325
densidad = 1.225
vel_aire = 100
foto = input("Which type of airfoil do you want to use :p1,p2,p3,p4  Please, write the airfoil name : ")
iterations = input("Number of iterations ? : ")
def matriz_imagen(nombre):
    image = Image.open(nombre).convert('RGBA') #convertimos la imagen directamente en binario (ya que la imagen solo contiene blanco y negro)
    imagen_bin = image.convert('1')
    vec_imagen = np.array(imagen_bin) # convertimos la imagen en  un array
    mat_imagen = np.empty((vec_imagen.shape[0], vec_imagen.shape[1]), None) #creamos una matriz vacia de las dimensiones de la imagenes.

    for i in range(len(vec_imagen)):  # asignamos los pixeles del contorno como NaN (ya que se asignará el valor en las iteraciones), y los de la figura como 0.
        for j in range(len(mat_imagen[i])):
            if vec_imagen[i][j] == True:
                mat_imagen [i][j] = 0
            else:
                mat_imagen[i][j] = np.nan

    return mat_imagen    #retorna la matriz binaria.

def contorno_imagen (matriz_binaria):
    contorno_img = np.zeros((matriz_binaria.shape[0], matriz_binaria.shape[1])) #creamos matriz vacia de la dimención de la imagen.
    #Hacemos un loop para asignar 3 diferentes valores en la matriz: 0 si es el fondo, 1 si es contrno figura y 2 interior de la figura.
    for i in range(1, matriz_binaria.shape[0] - 1):
        for j in range(1, matriz_binaria.shape[1] - 1):
            if np.isnan(matriz_binaria[i][j]):
                contorno_img[i][j] = 2
            elif (np.isnan(matriz_binaria[i][j - 1])) or (np.isnan(matriz_binaria[i][j + 1])) or \
                    (np.isnan(matriz_binaria[i + 1][j])) or (np.isnan(matriz_binaria[i - 1][j])):
                contorno_img[i][j] = 1
    return contorno_img # retornamos la matriz resultado.

#Función principal para el método numérico
def metodo_num (matriz, it):
    matriz_contorno = contorno_imagen(matriz) #utilizamos la función contorno_imagen para obtener la matriz de 0,1,2
    Lx = float(matriz.shape[0] - 1)  # 2 vectores en cada dirección
    Ly = float(matriz.shape[1] - 1)
    m = matriz.shape[1] - 1
    n = matriz.shape[0] - 1
    dx = Lx / n # diferencial en cada dirección
    dy = Ly / m
    vec_X = [] # vectores vacios
    vec_Y = []


    matriz_presion = np.zeros((matriz.shape[0], matriz.shape[1]))  # matriz inicial de presion
    matriz_potencial = np.zeros((matriz.shape[0], matriz.shape[1]))  # matriz inicial de potencial
    matriz_corriente = np.zeros((matriz.shape[0], matriz.shape[1]))  # matriz inicial de corriente
    matriz_vel = np.zeros((matriz.shape[0], matriz.shape[1]))  # martiz inicial de velocidad
    matriz_velx = np.zeros((matriz.shape[0], matriz.shape[1]))  # matriz inicial de velocidad en horizontal
    matriz_vely = np.zeros((matriz.shape[0], matriz.shape[1]))  # matriz inicial de velocidad en vertical

    for j in range(0, m + 1):  # vector de puntos en direccion Y
        vec_Y.append(j * dy)

    for i in range(0, n + 1):  # vector de puntos en direccion Y
        vec_X.append(i * dx)
    # asignamos cada punto de la matriz de la suncion a la matriz de cada variable.
    for j in range(0, m + 1):
        for i in range(0, n + 1):
            matriz_presion[i][j] = matriz[i][j]
            matriz_potencial[i][j] = matriz[i][j]
            matriz_corriente[i][j] = matriz[i][j]
            matriz_vel[i][j] = matriz[i][j]
            matriz_velx[i][j] = matriz[i][j]
            matriz_vely[i][j] = matriz[i][j]

    # Establecemos las condiciones de contorno
    for i in range(0, n + 1):
        matriz_presion[i][0] = presión   #establecemos la presion atmosferica en los bordes de la imagen
        matriz_presion[i][m] = presión
        matriz_potencial[i][0] = 0.           #establecemos el potencial en los bordes (0 para punto inicial y v*dx para el resto)
        matriz_potencial[i][m] = vel_aire * Lx
        matriz_corriente[i][0] = vel_aire * vec_Y[i]  #establecemos el stream funtion en los bordes
        matriz_corriente[i][m] = vel_aire * vec_Y[i]
        matriz_vel[i][0] = vel_aire   #velocidad de aire constante en todos los lugares
        matriz_vel[i][m] = vel_aire
        matriz_velx[i][0] = vel_aire
        matriz_velx[i][m] = vel_aire
        matriz_vely[i][0] = 0. #la velocidad es en direccion x por lo tanto es 0 en dirección y
        matriz_vely[i][m] = 0.
    #Asignamos la condiciones de la misma manera en los pixeles en direccion vertical
    for j in range(0, m +1):
        matriz_presion[0][j] = presión
        matriz_presion[n][j] = presión
        matriz_potencial[0][j] = vel_aire * vec_X[j]
        matriz_potencial[n][j] = vel_aire * vec_X[j]
        matriz_corriente[0][j] = 0.
        matriz_corriente[n][j] = vel_aire * Ly
        matriz_vel[0][j] = vel_aire
        matriz_vel[n][j] = vel_aire
        matriz_velx[0][j] = vel_aire
        matriz_velx[n][j] = vel_aire
        matriz_vely[0][j] = 0.
        matriz_vely[n][j] = 0.

    # vamos asignando valores correspodientes a los puntos de la matriz para cada tipo(corriente, potencial, presión)
    for j in range(1, m):  # empezamos desde 1 porque los bordes ya tienen valores asignados anteriormente
        for i in range(1, n):
            if (matriz_contorno[i][j] == 0) or (matriz_contorno[i][j] == 1): # si el valor era 2 queria decir que era el interior de la figura
                matriz_corriente[i][j] = (vel_aire * Ly)/2  #establecemos los valores iniciales de stream y potencial, como la mitad del producto de la velocidad de aire por la variacion de distancia
                matriz_potencial[i][j] = (vel_aire * Lx)/2
                matriz_presion[i][j] = presión

    # ahora calculamos la funcion de corriente y de potencial con la ecuacion discretizada en el short report.
    for num_it in range(0, int(iterations)):
        for j in range(1, m):
            for i in range(1, n):
                if matriz_contorno[i][j] == 0:
                    matriz_corriente[i][j] = ((dy ** 2) * (matriz_corriente[i + 1][j] + matriz_corriente[i - 1][j]) + (dx ** 2) * (matriz_corriente[i][j + 1] + matriz_corriente[i][j - 1])) / (2 * ((dx ** 2) + (dy ** 2)))
                    matriz_potencial[i][j] = ((dy ** 2) * (matriz_potencial[i + 1][j] + matriz_potencial[i - 1][j]) + (dx ** 2) * (matriz_potencial[i][j + 1] + matriz_potencial[i][j - 1])) / (2 * ((dx ** 2) + (dy ** 2)))

        if num_it % 100 == 0 and num_it != 0:
            print("Iterationes completadas: " + str(num_it) + ".")

    # calculamos la velocidad utilizanto la funcion de corriente y de presion con la ecuación de Bernoulli.
    for j in range(1, m):
        for i in range(1, n):
            if matriz_contorno[i][j] == 0:
                matriz_velx[i][j] = (matriz_corriente[i][j + 1] - matriz_corriente[i][j - 1]) / 2 * dx
                matriz_vely[i][j] = - ((matriz_corriente[i + 1][j]) - (matriz_corriente[i - 1][j])) / 2 * dy
                matriz_vel[i][j] = math.sqrt(((matriz_vely[i][j]) ** 2 + (matriz_velx[i][j]) ** 2))
                matriz_presion[i][j] = presión + (densidad / 2) * (vel_aire ** 2 - (matriz_vel[i][j] ** 2))



    # streamlines plot
    plt.figure(1)
    plt.title("STREAMLINES DISTRIBUTION")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.gca().invert_yaxis()
    contour = plt.contourf(vec_X, vec_Y, matriz_corriente, 75)
    a=plt.colorbar(contour)
    a.set_label('Streamlines distribution', rotation=270, labelpad=15)
    plt.contour(vec_X, vec_Y, matriz_corriente, 75, colors=('darkgray'), linewidths=(0.5), linestyles='solid')
    plt.axis("scaled")

    # potential plot
    plt.figure(2)
    plt.title("EQUIPOTENTIAL DISTRIBUTION")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.gca().invert_yaxis()
    equi = plt.contourf(vec_X, vec_Y, matriz_potencial, 75)
    b=plt.colorbar(equi)
    b.set_label('Equipotential', rotation=270, labelpad=15)
    plt.contour(vec_X, vec_Y, matriz_potencial, 75, colors=('darkgray'), linewidths=(0.5), linestyles='solid')
    plt.axis("scaled")

    # pressure plot
    plt.figure(3)
    plt.title("PRESSURE DISTRIBUTION (Pa)",fontsize=16)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.gca().invert_yaxis()
    pres = plt.contourf(vec_X, vec_Y, matriz_presion, 20)
    plt.contour(vec_X, vec_Y, matriz_presion, 20, colors='darkgray', linewidths=0.5, linestyles='solid')
    c=plt.colorbar(pres)
    c.set_label('Pressure(Pa)', rotation=270, labelpad=15)
    plt.axis("scaled")

    # velocity field plot
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    plt.title("VELOCITY FIELD ")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.gca().invert_yaxis()
    ax.streamplot(np.array(vec_X), np.array(vec_Y), -matriz_vely, matriz_velx, color='darkgray', linewidth=1)
    vel = ax.contourf(vec_X, vec_Y, matriz_vel, 50)
    d=plt.colorbar(vel)
    d.set_label('Velocity Magnitude (m/s)', rotation=270, labelpad=15)
    plt.axis("scaled")
    plt.show()

nueva_matriz = matriz_imagen(foto + ".jpg")
metodo_num(nueva_matriz, iterations)

