import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math

from PIL.ImageChops import difference
from numpy.linalg import norm
from numpy import arccos, dot, pi, cross

guard_size = 10
poisson_lam = 0.3
radius_block = 0.5
distance = 3
list_point=np.array

def plot_circle(x,y,r):
    angles=np.linspace(0.2*np.pi,50)
    x_cir=x+r*np.cos(angles)
    y_cir=y+r*np.sin(angles)
    plt.plot(x_cir,y_cir,'lavender')

def poisson_point_process(lambda0,area_size):
    N = np.random.poisson(lambda0*area_size**2)
    x = np.random.uniform(0,area_size,size=N)
    y = np.random.uniform(0,area_size, size=N)
    return x,y
def paint_rectangle(x_point1, y_point1, x_point2, y_point2, angle, radius_block):
    difference_angle = 2 * np.pi - angle
    reverse_angle = np.pi / 2 -  difference_angle
    opposite_angle = reverse_angle + np.pi
    x_rectang_A = x_point1 + radius_block * np.cos(opposite_angle)
    y_rectang_A = y_point1 + radius_block * np.sin(opposite_angle)
    x_rectang_B = x_point1 + radius_block * np.cos(reverse_angle)
    y_rectang_B = y_point1 + radius_block * np.sin(reverse_angle)
    x_rectang_C = x_point2 + radius_block * np.cos(reverse_angle)
    y_rectang_C = y_point2 + radius_block * np.sin(reverse_angle)
    x_rectang_D = x_point2 + radius_block * np.cos(opposite_angle)
    y_rectang_D = y_point2 + radius_block * np.sin(opposite_angle)
    return x_rectang_A, y_rectang_A, x_rectang_B, y_rectang_B, x_rectang_C, y_rectang_C, x_rectang_D, y_rectang_D


def check_distance(A, B, C):
    CA = (C - A) / norm(C - A)
    BA = (B - A) / norm(B - A)
    CB = (C - B) / norm(C - B)
    AB = (A - B) / norm(A - B)

    if arccos(dot(CA, BA)) > 1:
        return norm(C - A)
    if arccos(dot(CB, AB)) > 1:
        return norm(C - B)
    return norm(cross(A - B, A - C)) / norm(B - A)


def crossing(x1, y1, x2, y2, x, y, radius_block):
    circle_point = []
    point_1 = []
    point_2 = []
    point_1.extend([x1, y1])
    point_2.extend([x2, y2])
    for i in range(len(x)):
        circle_point.append([x[i], y[i]])

    for i in range(len(x)):
        if (np.round(check_distance(list_point(point_1), list_point(point_2), list_point(circle_point[i])),
                     1) <= radius_block):
            return True


x, y = poisson_point_process(poisson_lam, guard_size)
x1 = np.random.uniform(0, guard_size)
y1 = np.random.uniform(0, guard_size)

angle = np.random.uniform(0, 2 * np.pi)

x2 = x1 + distance * np.cos(angle)
y2 = y1 + distance * np.sin(angle)

x_rectang_A, y_rectang_A, x_rectang_B, y_rectang_B, x_rectang_C, y_rectang_C, x_rectang_D, y_rectang_D = paint_rectangle(
    x1, y1, x2, y2, angle, radius_block)

plt.figure(dpi=100, figsize=(8, 8), facecolor='pink')
plt.title('Coverage area')

plt.plot(x, y, '.', alpha=0.7, label='fist', lw=5, mec='b', mew=2, ms=10)
for i in range(len(x)):
    plot_circle(x[i], y[i], radius_block)
plt.plot([x1, x2], [y1, y2], '.-g')

plt.plot([x_rectang_A, x_rectang_B], [y_rectang_A, y_rectang_B], '-.b')
plt.plot([x_rectang_A, x_rectang_D], [y_rectang_A, y_rectang_D], '-.b')
plt.plot([x_rectang_B, x_rectang_C], [y_rectang_B, y_rectang_C], '-.b')
plt.plot([x_rectang_D, x_rectang_C], [y_rectang_D, y_rectang_C], '-.b')

plt.xlim(0, guard_size)
plt.ylim(0, guard_size)
plt.savefig('plot_lab4_1.png')
plt.show()

def check_pol(line_d1, line_d2, distance, angle, fixang):
    x1, y1 = line_d1
    x2, y2 = line_d2

    atan1=math.atan2((y2-y1),(x2-x1))

    if atan1<0:
        atan1=2*np.pi+atan1
    d1=atan1>angle and atan1<(angle+fixang)

    if d1==True:
        return True
    else:
        return False

x1 = np.random.uniform(0,guard_size)
y1 = np.random.uniform(0,guard_size)

angle = np.random.uniform(0.2*np.pi)

x2 = x1 + distance * np.cos(angle)
y2 = y1 + distance * np.sin(angle)

angle1 = np.random.uniform(0.2*np.pi)
angle2 = np.random.uniform(0.2*np.pi)

distance1=np.random.uniform(distance,1.5*distance)
fixang=np.pi/4

x31=x1+distance*np.cos(angle1)
y31=y1+distance*np.sin(angle1)
x41=x1+distance*np.cos(angle1+fixang)
y41=y1+distance*np.sin(angle1+fixang)

x32=x2+distance*np.cos(angle2)
y32=y2+distance*np.sin(angle2)
x42=x2+distance*np.cos(angle2+fixang)
y42=y2+distance*np.sin(angle2+fixang)

poll1 = check_pol([x1,y1], [x2,y2],distance1,angle1,fixang)
poll2 = check_pol([x2,y2], [x1,y1],distance1,angle2,fixang)

plt.figure(dpi=100, figsize=(8,8), facecolor= 'pink')
plt.title('Coverage area')
plt.plot([x1,x2],[y1,y2], '.-g')

x_rectang_A, y_rectang_A, x_rectang_B, y_rectang_B, x_rectang_C,y_rectang_C, x_rectang_D, y_rectang_D=paint_rectangle(x1,y1,x2,y2,angle,radius_block) #получение координат для отрисовки зоны блокировки вокруг отрезка

plt.plot(x,y,'.', alpha=0.7,label='fist',lw=5,mec='b',mew=2,ms=10)
for i in range(len(x)):
    plot_circle(x[i],y[i], radius_block)
plt.plot([x1,x2],[y1,y2], '.-g')

plt.plot([x_rectang_A,x_rectang_B],[y_rectang_A,y_rectang_B], '-.b')
plt.plot([x_rectang_A,x_rectang_D],[y_rectang_A,y_rectang_D], '-.b')
plt.plot([x_rectang_B,x_rectang_C],[y_rectang_B,y_rectang_C], '-.b')
plt.plot([x_rectang_D,x_rectang_C],[y_rectang_D,y_rectang_C], '-.b')
plt.xlim(0,guard_size)
plt.ylim(0,guard_size)

plt.plot([x1,x31],[y1,y31], '-.b')
plt.plot([x1,x41],[y1,y41], '-.b')
plt.plot([x2,x32],[y2,y32], '-.r')
plt.plot([x2,x42],[y2,y42], '-.r')
plt.xlim(0,guard_size)
plt.ylim(0,guard_size)
plt.savefig('plot_lab4_2.png')
plt.show()

num_experiments = 1000
summa = 0
for i in range(num_experiments):
    x, y = poisson_point_process(poisson_lam, guard_size)

    x1 = np.random.uniform(0, guard_size)
    y1 = np.random.uniform(0, guard_size)

    angle = np.random.uniform(0, 2 * np.pi)

    x2 = x1 + distance * np.cos(angle)
    y2 = y1 + distance * np.sin(angle)

    angle1 = np.random.uniform(0, 2 * np.pi)
    angle2 = np.random.uniform(0, 2 * np.pi)

    distance1 = np.random.uniform(distance, 1.5 * distance)
    fixang = np.pi / 4

    poll1 = check_pol([x1, y1], [x2, y2], distance1, angle1, fixang)
    poll2 = check_pol([x2, y2], [x1, y1], distance1, angle2, fixang)

    fl = True

    if (crossing(x1, y1, x2, y2, x, y, radius_block)):
        fl = False

    if fl == True and poll1 == True and poll2 == True:
        summa += 1

print('Probability = ', summa / num_experiments)

s_B=2*radius_block*distance
lambda1=poisson_lam*s_B
p_block=1-math.exp(-lambda1)
print('Probability by formula = ',(1-p_block)*(fixang/(2*np.pi))**2)