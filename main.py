
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import math
from cramer import cramer


#Новая часть
def access_matrix(n : int, lam : float ) -> float:
    prob_matrix = [0] * n #матрица коеэффициентов переходов
    for i in range(n):
        prob_matrix[i] = [0] * n
    for i in range(0, n):#Перебор всех состояний начала
        for j in range(max(0, i-1), n):#перебор всех состояний конечной вершины. max(0, i-1) служит для того,
            # чтобы учесть возможность перехода в предыдущий элемент, и не ломать программу при i = 0
            result = round(probability_mark_process(i, j, lam), 5)
            prob_matrix[i][j] = result

    coef_matrix = [0] * (n + 1) #делаем матрицу коеффициентов
    for i in range(n+1):
        coef_matrix[i] = [0] * (n)

    for i in range(n): #забиваем её значениями 
        for j in range(n):            
            coef_matrix[i][j] = prob_matrix[j][i]
            if i == j:
                coef_matrix[j][i] -= 1.
   
    for i in range(n): #Последняя строка из единиц
        coef_matrix[n][i] = 1

    #prob_matrix = np.asarray(prob_matrix)
    #print("---------------------------------")
    #print(prob_matrix)
    #print("---------------------------------")
    new_coef_matrix = np.asarray(coef_matrix)
    #print(new_coef_matrix)
    #print("---------------------------------")

    B_matrix = [0] * (n + 1) # делаем матрицу ответов (правые части)
    
    B_matrix[n] = 1

    B_matrix = np.asarray(B_matrix)
    #print(B_matrix)

    pseudo_inv = np.linalg.pinv(new_coef_matrix) #находим матрицу переменных через обратную матрицу
    X = np.dot(pseudo_inv, B_matrix)

    # print(X)
    M = 0
    for i in range(n): # считаем матожидание
        M += i * X[i]
    #print(M)
    return M # возвращаем его из функции
#Конец новой части



   #for i in range(n+1):
   #     coef_matrix[n-2][i] -= prob_matrix[n-1][n-2]
   #     coef_matrix[n-1][i] -= prob_matrix[n-1][n-1] - 1

    
    #coef_matrix = np.asarray(coef_matrix)
    #print(coef_matrix)

    #print("---------------------------------")
    #M_k = [0]*n
    #for i in range(n):
    #    M_k[i] = [0]*n

    #for i in range(n):
    #    for j in range(n):
    #        M_k[i][j] = coef_matrix[i][j]
    #for i in range(n):
    #    M_k[0][i] = 1

    #M_k = np.asarray(M_k)
    #print(M_k)
     
    #print("---------------------------------")
    
    #V_k = [0]*n
    #for i in range(n):
    #    V_k[i] = coef_matrix[i][n]
    #V_k[0] = 1
    #V_k = np.asarray(V_k)
    #print(V_k)

    #print('----------------------\n Answer:')

    #ans_1 = np.linalg.solve(M_k, V_k)
    #print(np.asarray(ans_1))
    #cramer(coef_matrix)

    #cr = cramer(coef_matrix)

    #mds = cr.matr()

    ## generate a list with the coefficients obtained in each matrix after using the matr() method
    #deltas = []
    #for i in mds:
    #    deltas.append(cr.delta(i))

    ## to obtain the values ​​of each unknown, divide the different coefficients obtained by the first one, corresponding to that of the matrix of unknowns

    #for i in range(1 , len(deltas)):
    #    print("Variable" , i , ":")
    #    print(deltas[i] / deltas[0])

def mark_model(n : int):# функция, которая формирует граф для Марковских переходов и рисует его
    G = ig.Graph(directed = True)

    G.add_vertices(n)# добавление вершин
    G.vs["label"] = range(0, n)#названия для вершин


    
    weight = []#список весов
    edges = []#список ребер, представляет из себя список кортежей(пар), где первый элемент показывает начало ребра, второе - конец
    self = []#список булевых значений, который показывает является ли реберо - петлей
    for i in range(0, n):#Перебор всех состояний начала
        for j in range(max(0, i-1), n):#перебор всех состояний конечной вершины. max(0, i-1) служит для того,
            # чтобы учесть возможность перехода в предыдущий элемент, и не ломать программу при i = 0
            edges.append((i, j))
            result = round(probability_mark_process(i, j, 0.30), 5)
            weight.append(result)# вычисление вероятности перехода
            prob_matrix[i][j] = result
            if(i == j):#запись, если петля
                self.append(True)
            else:
                self.append(False)

    # print(prob_matrix)

    

    G.add_edges(edges) #добавление вершин
    G.es["weight"] = weight #добавление весов
    G.es["self"] = self #Добавление записей для петель
    fig, ax = plt.subplots(figsize=(11, 11)) #создание окна для рисования
    ig.plot(G, #отрисовка графа
            target=ax,
            layout="kk",  # print nodes in a kawada kiwai layout
            vertex_size=30.0, #размер вершины
            vertex_color="purple", #цвет вершин
            vertex_frame_width=4.0, #
            vertex_frame_color="white", #цвет фона
            vertex_label=G.vs["label"], # добавление подписей
            vertex_label_size=12.0, # размер подписей вершин
            edge_label=G.es["weight"], # добавление ребер
            edge_label_size=[5 if self else 13.0 for self in G.es["self"]] # если ребро - петля уменьшаем надпись, чтобы не мещала
            )
    plt.show()

def probability_mark_process(i: int, j: int, lam: float) -> float:# функция вычисления вероятности перехода из i в j
    if i == 0:
        return pow(lam, j) / math.factorial(j) * math.exp(-lam)
    elif j == i - 1:
        return (math.factorial(i)/math.factorial(i-1)) * float(1/i) * pow(1 - float(1/i), i - 1) * math.exp(-lam)
    elif j == i:
        return ((1 - math.factorial(i)/math.factorial(i-1) * float(1/i) * pow(1 - float(1/i), i - 1)) * math.exp(-lam)
                + (math.factorial(i)/math.factorial(i-1)) * float(1/i) * pow(1 - float(1/i), i - 1)) * lam * math.exp(-lam)
    elif j > i:
        return (math.factorial(i) / math.factorial(i-1) * float(1/i) * pow(1 - float(1/i), i - 1) * pow(lam, j - i + 1) / math.factorial(j - i + 1) * math.exp(-lam)
                + (1 - math.factorial(i)/math.factorial(i-1) * float(1/i) * pow(1 - float(1/i), i - 1)) * pow(lam, j - i) / math.factorial(j - i) * math.exp(-lam))
    else:
        return 0




def indicator_func(r_k: int) -> int:  # индикторная функция
    if r_k == 1:
        return 1
    else:
        return 0


def base_station_modulation(lam: float, slots: int) -> float:#функция, модулирующая работу базовой станции
    # входные параметры -лямбда и количество слотов, возвращает среднее количество абонентов.
    P_k = np.random.poisson(lam, slots) #формирование списка количества абонентов,
    # появляющихся в слоте, на основе распределения Пусассона
    N_k = [] * slots #список активных абонентов
    R_k = [] * slots #список абонентов пытающихся отправить сообщение
    N_k.append(0)
    for i in range(1, slots): #перебор всех слотов
        if N_k[i-1] == 0:
            R_k.append(0)
        else:
            R_k.append(np.random.binomial(N_k[i-1], 1/N_k[i-1])) #получить количество абонентов, которые пытаются отправить сообщение
        N_k.append(N_k[i-1] - indicator_func(R_k[i-1]) + P_k[i-1]) #вычисление следующего N_k
    sum = 0
    for i in N_k:
        sum += i
    return (1/slots)*sum #возвращаем среднее количество

#mark_model(4)

access_matrix(3, 0.50)


N_avg = [] * 10 #массив среднего количества абонентов в модели
T_avg = [] * 10 #массив среднего времени отправки сообщения (в слотах)
lam_list = [] * 10 #массив лямбда
M_list = [] * 10 #Массив МатОжиданий
for i in range(10): #перебор по различным лямбда
    lam = 0.05 + i * 0.05
    result = base_station_modulation(lam, 10000) #модуляция работы Базовой станции
    N_avg.append(result) #добавление N среднего
    T_avg.append(result / lam) #добавление T среднего
    lam_list.append(lam) #добавление лямбда
    M_list.append(access_matrix(170, lam)) #добавление МатОжиданий

plt.plot(lam_list, N_avg) #построение графика среднего количества абонентов от лямбда
plt.xlabel("Среднее количество абонентов, появляющееся в одном слоте")
plt.ylabel("Среднее количество абонентов")
plt.grid()
plt.show()

plt.plot(lam_list, M_list) #построение графика Мат Ожидания от лямбда
plt.xlabel("Среднее количество абонентов, появляющееся в одном слоте")
plt.ylabel("Математическое ожидание по марковским процессам")
plt.grid()
plt.show()

plt.plot(lam_list, T_avg) #построение графика среденго времени от ламбда
plt.xlabel("Среднее количество абонентов, появляющееся в одном слоте")
plt.ylabel("Среднее количество отправки сообщения")
plt.grid()
plt.show()



