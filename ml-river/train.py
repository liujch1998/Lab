import csv

loc = ['RJMN', 'NRSP', 'DELH', 'BGLR', 'VRNS', 'KLKT', 'GWHT']
def time_to_float (s):
	if ':' in s:
		[h, m] = s.split(':')
		[h, m] = [float(h), float(m)]
		return h * 60.0 + m
	if '.' in s:
		[h, m] = s.split('.')
		[h, m] = [float(h), float(m)]
		return h * 60.0 + m
	return None
X, Y1, Y2 = [], [], []
with open('data.csv','r') as f:
	lines = csv.reader(f, delimiter=',')
	for row in lines:
		if row[0] == 'Sample me':
			continue
		x = [loc.index(row[1])] + [float(s) if s != '' else None for s in row[2:4]] + [time_to_float(row[4])] + [float(s) if s != '' else None for s in row[5:11]]
		y1 = float(row[14])
		y2 = float(row[15])
		X.append(x)
		Y1.append(y1)
		Y2.append(y2)

import numpy as np
np.random.seed(0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

perm = np.random.permutation(len(X))
X = np.array(X, dtype=np.float)[perm]
Y1 = np.array(Y1, dtype=np.float)[perm]
Y2 = np.array(Y2, dtype=np.float)[perm]
X_train, X_test = X[:177,:], X[177:,:]
Y1_train, Y1_test = Y1[:177], Y1[177:]
Y2_train, Y2_test = Y2[:177], Y2[177:]
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

'''
estimator = Pipeline([('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)), ('forest', RandomForestRegressor(random_state=0, n_estimators=100))])
score = cross_val_score(estimator, X, Y1).mean()
print(score)
'''

print('COD')
regressor1 = RandomForestRegressor(n_estimators=200, random_state=0)
regressor1.fit(X_train, Y1_train)
print('Training set R^2 is %.4f' % regressor1.score(X_train, Y1_train))
print('Testing set R^2 is %.4f' % regressor1.score(X_test, Y1_test))

print('BOD')
regressor2 = RandomForestRegressor(n_estimators=200, random_state=0)
regressor2.fit(X_train, Y2_train)
print('Training set R^2 is %.4f' % regressor2.score(X_train, Y2_train))
print('Testing set R^2 is %.4f' % regressor2.score(X_test, Y2_test))

