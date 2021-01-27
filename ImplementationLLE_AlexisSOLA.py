import numpy as np
import random
import time

from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import eig
import matplotlib.pyplot as plt

k = 10
Size = 100
#Construire des points en N dimensions aléatoires
def InitData(N):
    data = []
    for i in range(0, N):
        case = []
        for j in range(0, k):
            case.append(random.randint(1, 100))

        data.append(case)

    return data

M =  np.asmatrix(InitData(Size))

#On cherche les k plus prochent voisins
N = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(M)

#Génération de la matrice creuse contenant les k voisins de chaque instances
Graph = np.asmatrix(N.kneighbors_graph(M).toarray())

def GetWeight(N, eps):

    W = np.asmatrix(Graph)

    for i in range(0, N):
        Z = []
        #On ajoute les voisins à la matrice Z
        for j in range(0, N):
            voisin = Graph[i, j]
            if voisin == 1:
                Z.append(M[j])

        for w in range(0, k):
            #On soustrait Xi a Z
            Z[w] = [Z_elmt - X_elmt for X_elmt, Z_elmt in zip(M[i], Z[w])]

        Z2 = []
        for r in range(0, k):
            vec = []
            for c in range(0, k):
                vec.append(np.asarray(Z[r])[0, 0][c])
            Z2.append(vec)

        Z2 = np.asmatrix(Z2)
        COV = np.dot(Z2, np.transpose(Z2))

        #On ajoute un terme de regularisation
        trace = np.trace(COV)
        #On veut éviter que COV ne soit singulière pour la régression linéaire
        if trace > 0:
            R = eps * trace
        else:
            R = eps

        COV = COV + R
        weights = linalg.solve(COV, np.ones(k).T, sym_pos=True)
        weights = weights / weights.sum()

        cpt = 0
        for j in range(0, N):
            voisin = Graph[i, j]
            if voisin == 1:
                W[i, j] = weights[cpt]
                cpt = cpt + 1
            else:
                W[i, j] = 0

    return np.array(W)

start_time = time.time()

W = GetWeight(Size, 0.0001)
I = np.identity(Size)

M2 = np.dot(I - W.T, I - W)

#On détermine nos vecteurs propres
D, V = eig(M2)
#On classe les vecteur propres
V.sort(1)

interval = time.time() - start_time
print('Total time in seconds:', interval)

def pts3D():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #x, y, z = np.indices((8, 8, 8))
    x =M[:, 0]
    y =M[:, 1]
    z =M[:, 2]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def graphique():
    X = V[0]
    Y = V[1]
    plt.scatter(X, Y)
    plt.show()

pts3D()
graphique()