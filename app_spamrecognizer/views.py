from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import string
import json
import random

# Method awal untuk merender halaman html
def index(request):
	temp = getTest()
	response = {'hasil': temp}
	html = 'app_spamrecognizer/spamrecognizer.html'
	return render(request, html, response)

# Method untuk mengolah input yang diberikan untuk menghasilkan output spam atau not spam dari text yang diberikan
@csrf_exempt
def main(request):
	if request.method == 'POST':
		string = request.POST['paramm']
		senArr = string.lower().split(" ")
		dictio = getDict()
		dictTrain, dictTest, labelTrain, labelTest = testing(dictio)
		arrTemp = start(senArr,dictTrain,labelTrain)
		temp = abs(arrTemp[0] - arrTemp[1]) < 0.0001 
		if (arrTemp[0] > arrTemp[1]):   
			res = {"hasil":"Text tersebut adalah spam"}
			return JsonResponse(res)
		else:
			res = {"hasil":"Text tersebut bukan spam"}
			return JsonResponse(res)

# Method untuk melakukan testing pada agent yang dibuat (mencari akurasi)
def getTest():
	dictio = getDict()
	dictTrain, dictTest, labelTrain, labelTest = testing(dictio)
	akurasi,true_positive, true_negative, false_positive, false_negative = getAccuracy(dictTrain,labelTrain,dictTest,labelTest)
	res = str(akurasi)+"%"
	return res

def start(senArr,dictTrain,labelTrain):
	probArr = InitProb(labelTrain) # Return array of probability spam and not spam
	dictWordSpam = {}
	dictWordNotSpam = {}
	dictTotal = {}
	# Loop untuk mengiterasi inputan per kata untuk dibuang tanda bacanya lalu dicari kemunculan kata tersebutdi dataset
	# baik kata tersebut spam atau not spam juga akan dihitung kemunculannya, lalu gabungan dari kemunculan kata di bagian spam dan not spam
	# akan digabungkan di dictionary bernama dictTotal, dengan key = kata dan value adalah total kemunculan kata tersebut (di spam dan not spam)
	for iter1 in senArr: 
		iter1 = iter1.strip(string.punctuation)
		counterSpam = 0
		counterNotSpam = 0
		for iter2 in dictTrain:
			if (iter1 in iter2.lower()):
				if(dictTrain[iter2] == 0 or dictTrain[iter2] == 2):
					counterNotSpam+=1
				elif(dictTrain[iter2] == 1):
					counterSpam+=1
			dictWordSpam[iter1] = counterSpam
			dictWordNotSpam[iter1] = counterNotSpam
			dictTotal[iter1] = counterSpam+counterNotSpam

	# Memanggil method hitunganProb untuk melakukan penghitungan tahap akhir yang akan mencari nilai probabilitas yang sebenarnya (untuk
	# menghindari adanya nilai 0 pada kemunculan kata dari input)
	arrHasil = hitunganProb(dictWordSpam, dictWordNotSpam, dictTotal, probArr)
	hasilSpam = arrHasil[0]*probArr[1]
	hasilNotSpam = arrHasil[1]*probArr[3]
	return [hasilSpam, hasilNotSpam]

# Method untuk membaca dataSet dari file .csv menjadi dictionary dengan key = pesan, dan valuenya = label
def getDict():
	# Untuk baca dataset tabelnya
	readData = pd.read_csv('app_spamrecognizer/dataSet.csv')
	dictio = {}
	teksSMS = readData['Teks']
	listLabel = readData['label']
	for i in range(0, len(listLabel)):  # Membaca teks SMS dan labelnya yang akan diinput ke dictionary
		dictio[teksSMS[i]] = listLabel[i]
	return dictio

def testing(dictio): #generate data dan label train beserta data dan label test sebanyak 100 data
	dictTrain = {}
	dictTest = {}
	labelTrain = []
	labelTest = []
	allKey = list(dictio.keys())
	cnt = 1
	for i in allKey:
		if(cnt <= 100):
			dictTest[i] = dictio[i]
			labelTest.append(dictio[i])
		else:
			dictTrain[i] = dictio[i]
			labelTrain.append(dictio[i])
		cnt+=1
	return [dictTrain,dictTest,labelTrain,labelTest]

# Method untuk mengkalkulasi nilai probabilitas tahap akhir per kata sesuai dengan rumus probabilitas yang diberikan
def hitunganProb(dictWordSpam, dictWordNotSpam, dictTotal, probArr):
	hasilAllWordSpam = 1
	hasilAllwordNotSpam = 1
	arrHasil = list(dictTotal.values())
	konstanta = 1
	peluangBantuan = 0.5
	i = 0
	j = 0
	for iter1 in dictWordSpam:
		tempNilai = (dictWordSpam[iter1]/(probArr[0]+probArr[2]))
		temp = tempNilai/probArr[1]
		res = (konstanta*peluangBantuan + arrHasil[i]*temp)/(konstanta+arrHasil[i])
		hasilAllWordSpam = hasilAllWordSpam*res
		i = i+1
	for iter2 in dictWordNotSpam:
		tempNilai = (dictWordNotSpam[iter2]/(probArr[0]+probArr[2]))
		temp = tempNilai/probArr[3]
		res = (konstanta*peluangBantuan + arrHasil[j]*temp)/(konstanta+arrHasil[j])
		hasilAllwordNotSpam = hasilAllwordNotSpam*res
		j = j+1
	return [hasilAllWordSpam, hasilAllwordNotSpam]

def getAccuracy(dictTrain,labelTrain,dictTest,labelTest): #mendapatkan akurasi dari data test
	true_positive = 0
	false_positive = 0
	true_negative = 0
	false_negative = 0

	listStatus = translateLabelToStatus(labelTest)
	hasilTest = []
	for i in dictTest.keys():
		messageTest = i
		arrTemp = start(messageTest.split(" "),dictTrain,labelTrain)

		if (arrTemp[0] > arrTemp[1]):
			hasilTest.append("s")# Kondisi untuk mengecek nilai dari perbandingan nilai probabilitas spam dan not spam
		else:
			hasilTest.append("ns")

	# Mengiterasi array berupa kumpulan label untuk mencari nilai True Positive, False Positive, True Negative, dan False Negative
	length = len(listStatus)
	for i in range(length):
		if(hasilTest[i] == "s" and listStatus[i] == hasilTest[i]):
			true_positive += 1
		elif(hasilTest[i] == "ns" and listStatus[i] == hasilTest[i]):
			true_negative += 1
		elif(hasilTest[i] == "s" and listStatus[i] != hasilTest[i]):
			false_positive += 1
		else:
			false_negative += 1

	return [((true_positive+true_negative)/length)*100,true_positive,true_negative,false_positive,false_negative]

# Method untuk mentranslate nilai dari label yang diberikan dari 0, 1, 2 menjadi s (untuk 1) dan ns (untuk 0 dan 2)
def translateLabelToStatus(label): #translate dari label yg tadinya 0,1,2 jadi "s" dan "ns"
	listStatus = []
	for i in label:
		if(i == 0 or i == 2):
			listStatus.append("ns")
		else:
			listStatus.append("s")
	return listStatus

def InitProb(listLabel):    # Method untuk mengkalkulasi nilai spam dan not spam dari dataSet yang ada (didapatkan dari nilai label)
							 # untuk label 1 = spam, dan 0 serta 2 merupakan not spam
	hasilSpam = 0
	hasilBknSpam = 0
	for i in listLabel:
		if i == 2 or i == 0:
			hasilBknSpam+=1
		elif i == 1:
			hasilSpam+=1
	temp1 = hasilSpam
	temp2 = hasilBknSpam
	hasilSpam = hasilSpam/len(listLabel)
	hasilBknSpam = hasilBknSpam/len(listLabel)
	return [temp1, hasilSpam, temp2, hasilBknSpam]