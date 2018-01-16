import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json

maxar = 20180505
minar = 0

def bugaga(b0, b1, b2, b3):
	i = 0
	n = 0
	asss = []
	while n < 4:	
		asss.append([])
		i = 0
		while i < 30:
			if n == 0:
				asss[n].append(b0['Close Price'][i])
			if n == 1:
				asss[n].append(b1['Close Price'][i])
			if n == 2:
				asss[n].append(b2['Close Price'][i])
			if n == 3:
				asss[n].append(b3['Close Price'][i])	
			i = i + 1
		n = n + 1
	return asss

def okey_data(v):
	v = v[:4] + v[5:7] + v[8:10]
	return int(v)

def normalization(v):
	global maxar
	global minar
	g = 0
	g = (v - minar)/(maxar - minar)
	return g

if os.path.exists("./newb.csv"):
	btc = pd.read_csv("./newb.csv")
else:
	btc = pd.read_csv("./btc.csv")
	btc['Date'] = btc['Date'].apply(okey_data)
	minar = int(btc['Date'][0])
	btc['Date'] = btc['Date'].apply(normalization)
	print(btc.head())
	btc['Price'] = btc['Price'] / 100000 
	print(btc.head())
	btc.to_csv('./newb.csv')

mas = []
answer = []
sz = btc['Price'].count()
inum = 30
sz = sz - inum
for i in range(sz):
	mas.append([])
	answer.append(btc['Price'][i + inum])
	for j in range(inum):
		mas[i].append(btc['Price'][i + j])
mas = np.asarray(mas)
answer = np.asarray(answer)
Big_train, test_train, Big_target, test_target = train_test_split(
	mas, answer,test_size =  0.13)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
modelN = model_from_json(loaded_model_json)
modelN.load_weights("model.h5")
modelN.compile(loss="mean_squared_error", optimizer="SGD", metrics=["accuracy"])
print(modelN.summary())

# yn = input("Train model Y-Yes or N-No: ")
yn = 'y'
if yn == 'y' or yn == 'Y':
	modelN.fit(Big_train, Big_target,epochs=5, verbose=1) #2.1907e-05
	resultN = modelN.predict(test_train)
	z = len(resultN) - 1
	saf=  z
	resultN = resultN*10000
	test_target = test_target * 10000
	while z > 0:
		print(resultN[z], test_target[z])
		z = z -1
	print(mean_squared_error(resultN, test_target)/saf)#9.8-14
	print("\n\n\n")
	b0 = pd.read_csv("./0.csv")
	b1 = pd.read_csv("./1.csv")
	b2 = pd.read_csv("./2.csv")
	b3 = pd.read_csv("./3.csv")
	b0['Close Price'] = b0['Close Price'] / 100000
	b1['Close Price'] = b1['Close Price'] / 100000
	b2['Close Price'] = b2['Close Price'] / 100000
	b3['Close Price'] = b3['Close Price'] / 100000
	a0 = b0['Close Price'][30]
	a1 = b1['Close Price'][30]
	a2 = b2['Close Price'][30]
	a3 = b3['Close Price'][30]
	mas0 = bugaga(b0, b1, b2, b3)
	ks = modelN.predict(mas0) * 100000
	print(ks[0], a0 * 100000)
	print(ks[1], a1 * 100000)
	print(ks[2], a2 * 100000)
	print(ks[3], a3 * 100000)
	yn = input("Save Neuro Y-Yes N-No: ")
	if (yn == 'y' or yn == 'Y'):
		model_json = modelN.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		modelN.save_weights("model.h5")

# 3.149e-06
# [14333.67729187] 14427.87
# [13862.21796274] 12629.81
# [13174.53682423] 13860.14
# [13313.0133152] 13202.76