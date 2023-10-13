import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import math

# Construct a graph with 5 vertices
# G = ig.Graph(directed = True)
#
# G.add_vertices(12)
# G.vs["label"] = ['Jan', 'Fed', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# G.add_edges([(0, 1), (0, 9), (0, 7), (1, 2), (1, 5), (2, 10), (3, 1), (3, 2), (4, 3), (5, 4), (5, 6), (6, 0), (6, 1),
#              (8, 5), (8, 6), (9, 8), (10, 11), (11, 3)])
#
# G.es["weight"] = [4, 3, 2, 3, 5, 1, 7, 2, 8, 7, 3, 5, 3, 2, 1, 6, 4, 1]
# G.es["label"] = range(18)
#
# # Set attributes for the graph, nodes, and edges
#
#
#
#
# # Plot in matplotlib
# # Note that attributes can be set globally (e.g. vertex_size), or set individually using arrays (e.g. vertex_color)
# fig, ax = plt.subplots(figsize=(7, 7))
# ig.plot( G,
#     target=ax,
#     layout= "kk", # print nodes in a circular layout
#     vertex_size=0.3,
#     vertex_color="purple",
#     vertex_frame_width=4.0,
#     vertex_frame_color="white",
#     vertex_label=G.vs["label"],
#     vertex_label_size=7.0
# )
#
# plt.show()

def mark_model(n : int):
    G = ig.Graph(directed = True)

    G.add_vertices(n)
    G.vs["label"] = range(0, n)
    weight = []
    edges = []
    self = []
    for i in range(0, n):
        for j in range(max(0, i-1), n):
            edges.append((i, j))
            weight.append(round(probability_mark_process(i, j, 0.30), 5))
            if(i == j):
                self.append(True)
            else:
                self.append(False)
    #
    # G.add_edges([(0, 1), (0, 9), (0, 7), (1, 2), (1, 5), (2, 10), (3, 1), (3, 2), (4, 3), (5, 4), (5, 6), (6, 0), (6, 1),
    #      (8, 5), (8, 6), (9, 8), (10, 11), (11, 3)])
    #
    # G.es["weight"] = [4, 3, 2, 3, 5, 1, 7, 2, 8, 7, 3, 5, 3, 2, 1, 6, 4, 1]
    # G.es["label"] = range(18)

    G.add_edges(edges)
    G.es["weight"] = weight
    G.es["self"] = self
    fig, ax = plt.subplots(figsize=(7, 7))
    ig.plot(G,
            target=ax,
            layout="kk",  # print nodes in a circular layout
            vertex_size=0.4,
            vertex_color="purple",
            vertex_frame_width=4.0,
            vertex_frame_color="white",
            vertex_label=G.vs["label"],
            vertex_label_size=12.0,
            edge_label=G.es["weight"],
            edge_label_size=[5 if self else 11.0 for self in G.es["self"]]
            )
    plt.show()

def probability_mark_process(i: int, j: int, lam: float) -> float:
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


def base_state_modulation(lam: float, slots: int) -> float:
    P_k = np.random.poisson(lam, slots)
    N_k = [] * slots
    R_k = [] * slots
    N_k.append(0)
    for i in range(1, slots):
        if N_k[i-1] == 0:
            R_k.append(0)
        else:
            R_k.append(np.random.binomial(N_k[i-1], 1/N_k[i-1]))
        N_k.append(N_k[i-1] - indicator_func(R_k[i-1]) + P_k[i-1])
    sum = 0
    for i in N_k:
        sum += i
    return (1/slots)*sum

mark_model(4)

N_avg = [] * 10
T_avg = [] * 10
lam_list = [] * 10
for i in range(10):
    lam = 0.05 + i * 0.05
    result = base_state_modulation(lam, 10000)
    N_avg.append(result)
    T_avg.append(result / lam)
    lam_list.append(lam)

plt.plot(lam_list, N_avg)
plt.xlabel("среднее количество абонентов, появляющееся в одном слоте")
plt.ylabel("Среднее количество абонентов")
plt.show()

plt.plot(lam_list, T_avg)
plt.xlabel("среднее количество абонентов, появляющееся в одном слоте")
plt.ylabel("Среднее количество отправки сообщения")
plt.show()



