# K-N neighbors algorithms

# Data preparation
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head()) # print 1st line header of fish_csv file

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy() # 뽑은 data array를 numpy 포맷으로 변경하여 fish_input에 넣음
fish_target = fish['Species'].to_numpy()

# print(type(fish_input))
# print(fish_target)

# Separation of data usnig train_test_split 메서드
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# Transforamtion using mean and standard deviation
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# Design a model using K-N neighbors algorithm
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3) #
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

print(kn.classes_) # 모델이 train target 으로 부터 추출한 값, 알파벳 순서
print(kn.predict(test_scaled[:9]))

# Compute probability of luck backs
import numpy as np

proba = kn.predict_proba(test_scaled[:9]) # 확률 계산 메서드를 대부분 제공해 주고 있음
print(np.round(proba, decimals=4))
