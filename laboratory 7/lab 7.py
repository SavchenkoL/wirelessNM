import matplotlib.pyplot as plt
import numpy as np

def get_k(G1, alpha):
    return ((2 / G1) - (1 - np.cos(alpha / 2))) / (1 + np.cos(alpha / 2))

def get_G2(G1, k):
    return G1*k

G1_64x1 = 57.51
G1_32x1 = 28.76

alpha_64x1 = 1.585*(np.pi/180)
alpha_32x1 = 3.171*(np.pi/180)

K_64x1 = get_k(G1_64x1, alpha_64x1)
G2_64x2 = get_G2(G1_64x1, K_64x1)
print(f"K при G1_64x1 = {K_64x1}, соответственно G2 = {G2_64x2}")
K_32x1 = get_k(G1_32x1, alpha_32x1)
G2_32x2 = get_G2(G1_32x1, K_32x1)
print(f"K при G1_32x1 = {K_32x1}, соответственно G2 = {G2_32x2}")

h_A = 10
h_U = 1.4
h_B = 1.7

lambda_b = np.linspace(0.1, 3, 100)
dist = 5
rad_b = 0.3
def prob_2d(lambda_b, rad_b, dist):
    return 1 - np.exp(-lambda_b*2*rad_b*dist)
def prob_3d(lambda_b, rad_b, dist, h_A, h_U, h_B):
    return 1 - np.exp(-lambda_b*2*rad_b*((dist*((h_B-h_U)/(h_A-h_U)))+rad_b))

probs_2d = []
for i in lambda_b:
    probs_2d.append(prob_2d(i, rad_b=rad_b, dist=dist))

probs_3d = []
for i in lambda_b:
    probs_3d.append(prob_3d(i, rad_b=rad_b, dist=dist, h_A=h_A, h_U=h_U, h_B=h_B))

plt.plot(lambda_b, probs_2d, label='Вероятности 2d')
plt.plot(lambda_b, probs_3d, label='Вероятности 3d')
plt.xlabel('Lambda')
plt.ylabel('Вероятность')
plt.title('Отношение вероятности к интенсивности блокатора')
plt.legend()
plt.show()

