import numpy as np
#libreria para eliminar caracteres especiales
import re
#Para la remosion de palabras vacias
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import SnowballStemmer
import itertools
from nanonets import NANONETSOCR
import os
import math
from sklearn.metrics import jaccard_score
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
  replacements = (
      ("á", "a"),
      ("é", "e"),
      ("í", "i"),
      ("ó", "o"),
      ("ú", "u"),
      ("Á", "A"),
      ("É", "E"),
      ("Í", "I"),
      ("Ó", "O"),
      ("Ú", "U"),
  )
  for a, b in replacements:
      s = s.replace(a, b).replace(a.upper(), b.upper())
  return s

def eliminarCaracteres(doc):
  #doc = normalize(doc)
  elim = []
  for i in range(len(doc)):
    texto = normalize(doc[i])
    puntuación = r'[,;.:¡!¿?@#$%&[\](){}<>~=+\-*/|\\_^`´"\']'
    texto = re.sub(puntuación, ' ', texto)
    texto = re.sub(r'[^A-Za-z0-9ñÑ]+',' ', texto)
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
  documento = [word for word in documento if not word in set(cargaColecTxt('static/UPD/spanish.txt'))]
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

"""N-gramas"""

def generate_N_grams(text,ngram=1):
  text = eliminarCaracteres(text)
  minusculas(text)
  text = tokenizacion(text)
  text = stop_word(text)
  words = stemmer(text)
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans

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

"""Construcción del diccionario con n-gramas"""
def construcInvertedN(coleccion,n1,n2,n3):
  #Full Inverted Index
  #Diccionario de Resúmenes
  dicColec = { i+1 : [coleccion[i]] for i in range(0, len(coleccion) ) }
  #Construcción de Full Inverted Index de todos los documentos
  invertedColec = {}
  if n2 == 0 and n3 == 0: #1 sintagma
    for doc_id, text in dicColec.items():
      text=generate_N_grams(text,ngram=n1)
      doc_index = inverted_index(text)
      inverted_index_add(invertedColec, doc_id, doc_index)  
  elif n3 == 0: #2 sintagmas
    for doc_id, text in dicColec.items():
      text=generate_N_grams(text,ngram=n1)
      doc_index = inverted_index(text)
      inverted_index_add(invertedColec, doc_id, doc_index)
    for doc_id, text in dicColec.items():
      text=generate_N_grams(text,ngram=n2)
      doc_index = inverted_index(text)
      inverted_index_add(invertedColec, doc_id, doc_index)
  else: #3 sintagmas
    for doc_id, text in dicColec.items():
      text=generate_N_grams(text,ngram=n1)
      doc_index = inverted_index(text)
      inverted_index_add(invertedColec, doc_id, doc_index)
    for doc_id, text in dicColec.items():
      text=generate_N_grams(text,ngram=n2)
      doc_index = inverted_index(text)
      inverted_index_add(invertedColec, doc_id, doc_index)
    for doc_id, text in dicColec.items():
      text=generate_N_grams(text,ngram=n3)
      doc_index = inverted_index(text)
      inverted_index_add(invertedColec, doc_id, doc_index) 
  return dicColec,invertedColec

"""Jaccard y Sorensen"""

#JACCARD
#Función de la Bolsa de Palabras
def bagWordsBinaria(inverted,dicResumen):
  bagWord = np.zeros((len(inverted),len(dicResumen)))
  i=0
  for tokens, text in inverted.items():
    for docId, l1 in text.items():
      bagWord[i,docId-1] = 1
    i+=1
  return bagWord

def jaccardMatrix(matrixbin):
  filas,columnas = matrixbin.shape
  m1 = np.empty((columnas, columnas))
  for i in range(columnas):
    for j in range(columnas):
      m1[i][j] = m1[j][i] =  jaccard_score(matrixbin[:,i],matrixbin[:,j])  
  return np.round(m1,5)*100

def jaccardCompleto(coleccion,n1,n2,n3):
  dicGeneral, inverGeneral = construcInvertedN(coleccion,n1,n2,n3)
  bgBin = bagWordsBinaria(inverGeneral,dicGeneral)
  return jaccardMatrix(bgBin)

#Sorensen
def sorensen(d1,d2):
  intersection = np.logical_and(d1, d2)
  return 2. * intersection.sum() / (d1.sum() + d2.sum())

def sorensenMatrix(matrixbin):
  filas,columnas = matrixbin.shape
  m1 = np.empty((columnas, columnas))
  for i in range(columnas):
    for j in range(columnas):
      m1[i][j] = m1[j][i] =  sorensen(matrixbin[:,i],matrixbin[:,j])  
  return np.round(m1,5)*100

def sorensenCompleto(coleccion,n1,n2,n3):
  dicGeneral, inverGeneral = construcInvertedN(coleccion,n1,n2,n3)
  bgBin = bagWordsBinaria(inverGeneral,dicGeneral)
  return sorensenMatrix(bgBin)

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
  return distMatrix*100

"""Carga de Colección o Documento"""
def cosenoVectN(coleccion,n1,n2,n3):
    """Construcción del diccionario, full inverted index y Matriz TF-IDF"""
    dicGeneral, inverGeneral = construcInvertedN(coleccion,n1,n2,n3)
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
    return np.round(matrizDistAbs,5)

"""Extracción respuesta"""

def respuesta(matrix,n):
  filas, columnas = matrix.shape
  res = np.zeros((filas-4, columnas-n))
  for i in range(filas - 4):
    for j in range(columnas - n):
      res[i][j] = matrix[i][j+n]
  return res