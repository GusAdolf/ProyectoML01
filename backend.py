import numpy as np
import pandas as pd
from string import digits
import csv
#libreria para eliminar caracteres especiales
import re
#Para la remosion de palabras vacias
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import SnowballStemmer
import itertools
import math
from nanonets import NANONETSOCR
import os
model = NANONETSOCR()
model.set_token('d-AfG5kaGRX00gMvW4W5epeg6QY3FIeR')
bandera = 0

"""Carga de documentos y colecciones"""

def cargaListas(lista):
  with open(lista, 'r') as file:
      data = file.read().replace('\n', ' ')
  return data

def cargaDoc(documento):
  name, extension = os.path.splitext(documento) 
  if extension =='.pdf':
    data = model.convert_to_string(documento,formatting='none')
  elif extension == '.txt':
      data = cargaListas(documento)
  elif extension == '.csv':
    with open(documento, 'r') as csvfile:
      data = ''.join(map(str,csvfile.readlines()))
  else:
    data = model.convert_to_string(documento,formatting='none')
  bandera = 1
  return data,bandera

def cargaColecTxt(documento):
    file = open(documento, "r")
    data = file.read()
    data = data.split("\n")
    data = [x for x in data if x != '']
    file.close()
    return data

def cargaColec(documento):
  name, extension = os.path.splitext(documento)
  bandera = 2
  if extension == '.txt':
    data = cargaColecTxt(documento)
  elif extension == '.csv':
    items = list()
    data = []
    with open(documento,'r') as fp: 
        for line in fp.readlines(): 
            col = line.strip().split(",") 
            items.append(col)
    for lista in items:
      data.append(''.join(map(str,lista)))
  return data,bandera

def cargaListas(lista):
  with open(lista, 'r') as file:
      data = file.read().replace('\n', ' ')
  return data

def cargaDoc(documento):
  name, extension = os.path.splitext(documento) 
  if extension =='.pdf':
    data = model.convert_to_string(documento,formatting='none')
  elif extension == '.txt':
      data = cargaListas(documento)
  elif extension == '.csv':
    with open(documento, 'r') as csvfile:
      data = ''.join(map(str,csvfile.readlines()))
  else:
    data = model.convert_to_string(documento,formatting='none')
  bandera = 1
  return data,bandera

def cargaColecTxt(documento):
    file = open(documento, "r")
    data = file.read()
    data = data.split("\n")
    data = [x for x in data if x != '']
    file.close()
    return data

def cargaColec(documento):
  name, extension = os.path.splitext(documento)
  bandera = 2
  if extension == '.txt':
    data = cargaColecTxt(documento)
  elif extension == '.csv':
    items = list()
    data = []
    with open(documento,'r') as fp: 
        for line in fp.readlines(): 
            col = line.strip().split(",") 
            items.append(col)
    for lista in items:
      data.append(''.join(map(str,lista)))
  return data,bandera

"""Lectura archivos externos"""

def colecCompleta(res,l1,l2,l3,l4):
  datos,flag = res
  if flag == 1:
    datos = [datos,l1,l2,l3,l4]
  elif flag == 2:
    datos.extend((l1,l2,l3,l4))
  return datos

"""NLP"""

#NLP
def normalize(s):
  #if type(s) is str:
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s
  #else:
    #return s


def eliminarCaracteres(doc):
  #doc = normalize(doc)
  elim = []
  for i in range(len(doc)):
    texto = normalize(doc[i])
    puntuación = r'[,;.:¡!¿?@#$%&[\](){}<>~=+\-*/|\\_^`´"\']'
    texto = re.sub(puntuación, ' ', texto)
    texto = re.sub(r'[^A-Za-z0-9]+',' ', texto)
    texto = re.sub('\d', ' ', texto)
    texto = re.sub('\n', ' ', texto)
    texto = re.sub('\t', ' ', texto)
    texto = re.sub('\ufeff', ' ', texto)
    elim.append(texto)
  #print(doc)
  return elim

#Minusculas
def minusculas(docu):
  
  for i in range(len(docu)):
    docu[i] =docu[i].lower()

  return docu

#proceso de tokenizacion
def tokenizacion(doc):
  docf = []
  for pos in range(len(doc)):
    docf.append(doc[pos].split())
  
  f = list(itertools.chain(*docf))
  return f

#StopWords
def stop_word(documento):
  documento = [word for word in documento if not word in set(stopwords.words('spanish'))]
  documento = [word for word in documento if not word in set(cargaColecTxt('static\FILES\spanish.txt'))]
  return documento

#Stemmer
def stemmer(documento):
  spanishstemmer=SnowballStemmer('spanish')
  d = [] #Lista vacía para agregar las palabras por el proceso de stemming
  for word in documento:
      d.append(spanishstemmer.stem(word))
  return d

# Función NLP
def nlp(documento):
  texto = eliminarCaracteres(documento)
  minusculas(texto)
  texto = tokenizacion(texto)
  texto = stop_word(texto)
  texto = stemmer(texto)
  return texto

"""Full Inverted Index"""

def inverted_index(text):
    """        
    Creación de un Inverted index de cada documento específico
    {word:[posiciones]}
    """
    inverted = {}
    for index, word in enumerate(text):
        locations = inverted.setdefault(word, [f"fr: {text.count(word)}"])
        locations.append(index+1)
    return inverted
    
def inverted_index_add(inverted, doc_id, doc_index):
    """
    Añade al Inverted-Index el doc_index del documento con su doc_id
    respectivo, usando el doc_id como identificador.
    using doc_id as document identifier.
        {word:{doc_id:[locations]}}
    """
    for word, locations in doc_index.items():
        indices = inverted.setdefault(word, {})
        indices[doc_id] = locations
    return inverted

def construcInverted(coleccion):
  #Full Inverted Index
  #Diccionario de Resúmenes
  dicColec = { i+1 : [coleccion[i]] for i in range(0, len(coleccion) ) }
  #Construcción de Full Inverted Index de todos los documentos
  invertedColec = {}
  for doc_id, text in dicColec.items():
    text=nlp(text)
    doc_index = inverted_index(text)
    inverted_index_add(invertedColec, doc_id, doc_index)
  return dicColec,invertedColec

"""Jaccard"""

#JACCARD

#Funcion NLP Jaccard -> return Lista de listas
def nlpJacard(lista):
  d = { i+1 : [lista[i]] for i in range(0, len(lista) ) }
  listaNlp=[]
  for clave , valor in d.items():
    listaNlp.append(nlp(valor))
  return listaNlp 

def jaccard(d1,d2):
    intersec = len(np.intersect1d(d1,d2))   
    uni = len(np.union1d(d1,d2)) 
    return round(intersec/uni,5)

def jaccardMatrix(documento):
  m1 = np.empty((len(documento), len(documento)))
  for i in range(len(documento)):
    for j in range(len(documento)):
      m1[i][j] = m1[j][i] = jaccard(documento[i],documento[j])  
  return m1

def jaccardCompleto(documento):
    return jaccardMatrix(nlpJacard(documento))

#Función de la Bolsa de Palabras 
def bagWordsBinaria(inverted,dicResumen): 
  bagWord = np.zeros((len(inverted),len(dicResumen)))
  i=0
  for tokens, text in inverted.items():
    for docId, l1 in text.items():
      bagWord[i,docId-1] = 1
    i+=1
  return bagWord

"""TF-IDF"""

#Función de la Bolsa de Palabras
def bagWords(inverted,dicResumen):
  bagWord = np.zeros((len(inverted),len(dicResumen)))
  i=0
  for tokens, text in inverted.items():
    for docId, l1 in text.items():
      bagWord[i,docId-1] = l1[0].split(" ")[1]
    i+=1
  return bagWord
#Función de Pesado del Término
def wTF(tf):
  if tf>0:
    return 1 + math.log10(tf)
  else:
    return 0
#Función que me devuelve la Matriz de Pesos TF
def matrixWTF(mTF):
  filas, columnas = mTF.shape
  mWTF = np.zeros((filas,columnas))
  for i in range(filas):
    for j in range(columnas):
      mWTF[i][j] = wTF(mTF[i][j])

  return mWTF
#Función que retorna el Document Frequency (df)
def df(mTF):
  df1 = []
  for listaToken in mTF:
    df1.append(np.count_nonzero(listaToken))
  return df1

#Función para cálculo IDF
def idf(mTF,df1):
  filasTF, N = mTF.shape
  idf1=[]
  for elemento in df1:
        idf1.append(math.log10(N/elemento))
  return idf1

#Función para cálculo de TF-IDF
def tfIDF(mWtf,idf1):
  matriztfIDF = np.zeros((len(mWtf),len(mWtf[0])))
  i = j = 0
  while True:
      matriztfIDF[i][j] = mWtf[i][j]*idf1[i]
      j += 1
      if j == len(mWtf[0]):
          j = 0
          i += 1
      if i == len(mWtf):
          break
  return matriztfIDF

"""Coseno Vectorial"""

#Función de Normalización de la Matriz
def normMatrix(matrix):
  tranMatrix = np.transpose(matrix)
  normMatrix = []
  for vector in tranMatrix:
    modulo = np.linalg.norm(vector)
    normMatrix.append(vector/modulo)
  return normMatrix

#Función para definir la matriz de Distancias
def distMatrix(matrix):
  filas = columnas = len(matrix)
  distMatrix = np.zeros((filas,columnas))
  for i in range(filas):
    for j in range(columnas):
      if distMatrix[i][j] == 0:
        distMatrix[i][j] = distMatrix[j][i] = round(np.dot(matrix[i],matrix[j]),8)
  return distMatrix

"""Carga de Colección o Documento"""

def cosenoVect(coleccion):
    """Construcción del diccionario, full inverted index y Matriz TF-IDF"""
    dicGeneral, inverGeneral = construcInverted(coleccion)
    bg = bagWords(inverGeneral,dicGeneral)
    mWTF = matrixWTF(bg)
    dfGeneral = df(mWTF)
    idfGeneral = idf(mWTF,dfGeneral)
    matrizTFidf = tfIDF(mWTF,idfGeneral)
    """Coseno Vectorial"""
    #Normalización de la Matriz
    matrizNorm = normMatrix(matrizTFidf)
    #Matriz Distancia --> Matriz de los Abstract
    matrizDistAbs = distMatrix(matrizNorm)
    return matrizDistAbs