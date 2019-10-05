from math import tanh
from numpy import array, matmul, newaxis, append, arange, array_equal
from numpy.random import permutation
from random import random, shuffle
from time import time
import matplotlib.pyplot as plt
from json import load, dump

start_time = time()
# function definition and load data
sigma = lambda x: tanh(x)
sigma_prime = lambda x: (1 - x ** 2)
rand = lambda: (random() / 2 - .25)  # random number between -.25 and .25s

with open('output/out.txt') as file:
	data = load(file)

train_num = int(len(data['X']) * 1)
y_labels = ["hello", "on", "off", "go"]

total_data = array(data['X'])

z_d_total = array([[1 if int(i == (y_labels.index(d))) else -1 for i in range(len(y_labels))] for d in data['Y']])

c = list(zip(total_data, z_d_total))
shuffle(c)
total_data, z_d_total = zip(*c)

train = array(total_data[:train_num])
test = array(total_data[train_num:])

z_d = z_d_total[:train_num]
z_d_test = z_d_total[train_num:]

hiden_neuron = 12

v = array([[rand() for _ in range(len(train[0]) + 1)] for _ in range(hiden_neuron)])
w = array([[rand() for _ in range(hiden_neuron + 1)] for _ in range(len(y_labels))])

eta_w = .09
eta_v = .04
error_th = .001 * train_num * len(z_d[0])
print error_th
alpha_v = .8
alpha_w = .7
errors = []

last_del_v = 0
last_del_w = 0
finished = False
epoch = 0
error = error_th + .1
while error > error_th :
	error = 0
	for i in permutation(len(train)):
		y = list(map(sigma, matmul(append(train[i], array([-1])), v.T)))
		y_prime = list(map(sigma_prime, y))
		y.append(-1)

		z = list(map(sigma, matmul(y, w.T)))
		z_prime = list(map(sigma_prime, z))

		delta_z = array(z_prime * (z_d[i] - z))
		delta_y = array(y_prime * matmul(w.T[:-1], delta_z))[newaxis]

		del_v = eta_v * matmul(delta_y.T, append(train[i], array([-1]))[newaxis])
		del_w = eta_w * matmul(array(delta_z)[newaxis].T, array(y)[newaxis])

		v += del_v + alpha_v * last_del_v
		w += del_w + alpha_w * last_del_w

		last_del_v = del_v
		last_del_w = del_w

		y = list(map(sigma, matmul(append(train[i], array([-1])), v.T)))
		y.append(-1)
		z = list(map(sigma, matmul(y, w.T)))
		error += sum([(z[k] - z_d[i][k]) ** 2 for k in range(len(z_d[0]))])
	epoch += 1

	errors.append(error)
	print(epoch, error)

print("learned weights after " + str(epoch) + " epoch  in " + str(
	time() - start_time) + " seconds with normalized error :" + str(error / (len(train) * len(z_d[0]))))

plt.ylabel("E")
plt.xlabel("epoch")
plt.plot(array(arange(len(errors))), array(errors))
plt.show()
# testing

correct_num = 0

for i in range(len(test)):
	y = list(map(sigma, matmul(append(test[i], array([-1])), v.T)))
	y.append(-1)
	z = list(map(sigma, matmul(y, w.T)))
	max_z = max(z)
	out = array([1 if d is max_z else -1 for d in z])
	if array_equal(out, z_d_test[i]):
		correct_num += 1

if len(test) > 0:
	print("percentage of correct tests in test data is : " + str(float(correct_num) / len(test) * 100))

with open("./output/weight.txt", "w") as weight:
	dump({"W": w.tolist(), "V": v.tolist()}, weight)
