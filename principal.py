# Feature Selection Programm with Madelon data set
import pandas as pd
import numpy as np
from itertools import chain, combinations
from sklearn.neighbors import KNeighborsClassifier
from random import sample
from sklearn.ensemble import ExtraTreesClassifier

import time


espacio = ' '
newMinA = 0.0
newMaxA = 1.0

def vectorizacion2D(nombreArchivo):
    vectorMaxMin = []
    archivo = open(nombreArchivo, 'r')
    for linea in archivo.readlines():
        vector = linea.split(' ')
        vector.pop(len(vector) - 1)

        vector = list(map(float, vector))  # es como el map de JAVA

        vectorMaxMin.append(vector)

    vector2D = np.array(vectorMaxMin)
    archivo.close()
    return vector2D


def encontrarMaxMin(vector2Ddata):
    maxA = np.amax(vector2Ddata)
    minA = np.mean(vector2Ddata)
    return set([maxA,minA])
    # print(maxA, minA)


def normalizacionCorrelacion(vector2dData, minA, maxA):
    with np.nditer(vector2dData, op_flags=['readwrite']) as it:
      for x in it:
        x[...] = (((x-minA)/(maxA-minA))*(newMaxA-newMinA))+newMinA

    df = pd.DataFrame(vector2dData)

    matriz = pd.np.triu(df.corr(method='pearson').values)
    np.fill_diagonal(matriz,0)

    return matriz


def gruposThreadshold(matriz, threadshold):
    vectorPosiciones = []
    for fila in matriz:
        datoPrevio = list(map(abs,fila))
        datoFinal = [x for x in datoPrevio if x!=0]
        fila = np.array(datoFinal)

        if threadshold >= 0.6:
            columnas1 = np.where(fila > threadshold)
            if columnas1:
                vectorPosiciones.append(columnas1)
        if threadshold <= 0.2:
            columnas2 = np.where(fila < threadshold)
            vectorPosiciones.append(columnas2)

    return vectorPosiciones


def vectorizacionGrupos(vectoresThreashold):
    vectorFinal = []
    contador = 0
    for data in vectoresThreashold:
        contador = contador + 1
        if list(data[0]):
            lista = list(data[0])
            lista.insert(0,contador)
            vectorFinal.extend(lista)
    vectorFinal = list(set(vectorFinal))
    return vectorFinal

def targetVector(nombreArchivo):
    arrayTarget = []
    archivoTarget = open(nombreArchivo,'r')
    for targets in archivoTarget:
        arrayTarget.append(int(targets.replace('\n','')))
    return arrayTarget


def subconjuntos(lista, tamano):
    return list(set(list(combinations(lista, tamano))))


def main():

    vectorArchivo = vectorizacion2D('madelon_train(2).txt')
    vectorTarget = targetVector('madelon_train(3).txt')

    vectorArchivoValidacion = vectorizacion2D('madelon_valid(1).txt')
    vectorTargetValidacion = targetVector('madelon_valid(2).txt')


    targetColumn = pd.DataFrame({200: vectorTarget})

    y_train = targetColumn.to_numpy()

    Y = y_train.tolist()

    targetColumnValidacion = pd.DataFrame({200: vectorTargetValidacion})
    y_validation = targetColumnValidacion.to_numpy()



    model = ExtraTreesClassifier(n_estimators=500)
    model.fit(vectorArchivo.tolist(), np.ravel(Y))
    score = model.score(vectorArchivo.tolist(), np.ravel(Y))
    print(model.feature_importances_)
    dda = str(model.feature_importances_).replace('[','').replace(']','').replace('\n','').split(' ')
    final =[]
    for data in dda:
        if data:
            final.append(float(data))

    print(final)
    dat = np.array(final)
    promedio = np.mean(dat)
    vectorAnalisisFinal = np.where(dat>promedio)[0]
    print(vectorAnalisisFinal)

    df = pd.DataFrame(vectorArchivo)[vectorAnalisisFinal]
    X = df.values

    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs = SFS(knn,
              k_features=40,
              forward=True,
              floating=False,
              verbose=2,
              scoring='accuracy',
              cv=0)

    sfs.fit(df, targetColumn.values.ravel())

    conjuntoDeHipotesis = []
    for data in sfs.k_feature_names_.__iter__():
        conjuntoDeHipotesis.append(data)
    flag = False
    for i in range(10000):
        for j in range(7, 35):
            subset = sample(conjuntoDeHipotesis, j)
            dfPrubeas = pd.DataFrame(vectorArchivo)[subset]
            x_train = dfPrubeas.to_numpy()
            neighbor = KNeighborsClassifier(n_neighbors=12)

            dfValidacion = pd.DataFrame(vectorArchivoValidacion)[subset]
            x_validacion = dfValidacion.to_numpy()

            neighbor.fit(x_train, y_train.ravel())
            score = neighbor.score(x_train, y_train.ravel())
            score2 = neighbor.score(x_validacion, y_validation.ravel())
            print(score, subset)
            if score >= 0.92:
                print('***Dato valido', score,'-',score2, subset)
                flag = True
                break
        if flag== True:
            break



if __name__ == '__main__':
    # Para medir el tiempo
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))









































