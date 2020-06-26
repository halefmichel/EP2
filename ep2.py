# Nomes: Halef Michel (10774994) e Lucas Leone (10278868)

import math
import numpy as np
import random
import matplotlib.pyplot as plt


def main():
    # Definindo variáveis para os casos a e b
    T = 1
    teste = str(input('Digite qual teste você quer rodar: '))
    nf = 0
    linha = 128
    delta_x = 1.0 / linha
    delta_t = delta_x
    lambida = linha

    # CONDICIONAL PARA ENTRAR NO CASO a
    if teste.lower() == 'a':
        # definindo os pontos das fontes
        lista_p = np.array([0.35])

        # número de fontes
        nf = 1

        # intensidade da fonte
        lista_coef = np.array([7])

        # chamada de funções específicas para esse caso
        matriz_u = Crank(delta_t, linha, lambida, nf, lista_p)
        u_T = uT(nf, matriz_u, linha, lista_coef)

    # CONDICIONAL PARA ENTRAR NO CASO b
    elif teste.lower() == 'b':
        # definindo os pontos das fontes para o caso b
        lista_p = np.array([0.15, 0.3, 0.7, 0.8])

        # número de fontes
        nf = 4

        # intensidade das fontes
        lista_coef = np.array([2.3, 3.7, 0.3, 4.2])

        # chamada de funções específicas para esse caso
        matriz_u = Crank(delta_t, linha, lambida, nf, lista_p)
        u_T = uT(nf, matriz_u, linha, lista_coef)

    # CONDICIONAL PARA ENTRAR NO CASO c
    elif teste.lower() == 'c':
        # definindo as variáveis para o caso c
        linha = int(input('Digite o valor de N: '))
        delta_x = 1.0 / linha
        delta_t = delta_x
        lambida = linha

        # abrindo o arquivo .txt, adicionando em listas auxiliares e transformando os valores em float
        with open('./input.txt', 'r') as fp:
            line = fp.readline()
            lista_p = [float(elt.strip()) for elt in line.split(' ')]
            aux = []
            while line:
                line = fp.readline().lstrip()
                if line.strip() != '':
                    aux.append(float(line.strip()))

        # pegando os valores de u_T adequados do arquivo .txt
        u_T = []
        passo = math.floor(2048 / linha)
        for i in range(0, len(aux) - 1, passo):
            if i != 0:
                u_T.append(aux[i])
        u_T = np.array(u_T)

        # adequando o valor do número de fontes para o caso c
        nf = len(lista_p)

        # chamada de funções específicas para esse caso
        matriz_u = Crank(delta_t, linha, lambida, nf, lista_p)

    # CONDICIONAL PARA ENTRAR NO CASO d
    elif teste.lower() == 'd':
        # definindo as variáveis para o caso d
        linha = int(input('Digite o valor de N: '))
        delta_x = 1.0 / linha
        delta_t = delta_x
        lambida = linha

        # abrindo o arquivo .txt, adicionando em listas auxiliares e transformando os valores em float
        with open('./input.txt', 'r') as fp:
            line = fp.readline()
            lista_p = [float(elt.strip()) for elt in line.split(' ')]
            aux = []
            while line:
                line = fp.readline().lstrip()
                if line.strip() != '':
                    aux.append(float(line.strip()))

        # pegando os valores de u_T adequados do arquivo .txt
        u_T = []
        passo = math.floor(2048 / linha)
        for i in range(0, len(aux) - 1, passo):
            if i != 0:
                u_T.append(aux[i])

        # multiplicando os valores de u_T por números aleatórios entre -1 e 1
        u_T = np.array(u_T)
        for i in range(len(u_T)):
            a = 1 + (random.uniform(-1, 1)) * 0.01
            u_T[i] = a * u_T[i]

        # adequando o valor do número de fontes para o caso d
        nf = len(lista_p)

        # chamada de funções específicas para esse caso
        matriz_u = Crank(delta_t, linha, lambida, nf, lista_p)

    # chamada das funções comuns aos casos
    b, P = Prod_Escalar(nf, matriz_u, u_T)
    x = solve2(P, nf, b)
    erro(linha, matriz_u, u_T, x, nf)
    grafico(u_T, linha, nf, matriz_u, delta_x)


# FUNÇÃO QUE CALCULA O ERRO QUADRÁTICO
def erro(linha, matriz_u, u_T, x, nf):
    soma_2 = 0
    delta_x = 1 / linha

    # multiplicando as colunas da matriz que contém os valores de uk pelos valores das fontes encontradas
    for i in range(linha - 1):
        for j in range(nf):
            matriz_u[i][j] = matriz_u[i][j] * x[j]

    # difernça entre pos valores dos arquivos .txt para u_T e os valores encontrados, seguindo a teoria MMQ
    encontrado = np.zeros(len(u_T))
    for i in range(linha - 1):
        soma = 0
        for j in range(nf):
            soma += matriz_u[i][j]
        soma_2 += (u_T[i] - soma) ** 2

    # prints das intensidade das fontes encontradas e do valor do erro quadrático
    print()
    print("Para o N = %.f as fontes de calor encontradas foram:" % linha)
    for i in range(nf):
        print("a%.f = %.16f" % (i + 1, x[i]))
    print("O erro quadrático para N = %.f foi: %.16f" % (linha, math.sqrt(soma_2 * delta_x)))


# FUNÇÃO reset: FUNÇÃO QUE RESETA OS VALORES DA matriz_u E DAS LISTAS x E t
# Função usada no método de Crank-Nicolson
def reset(linha, delta_x, delta_t, coluna):
    # matriz de zeros
    u = np.zeros((linha + 1, coluna + 1))

    # valores assumidos pela posicao
    x = np.zeros(linha + 1)
    for i in range(linha + 1):
        x[i] = i * delta_x

    # valores assumidos pelo tempo
    t = np.zeros(coluna + 1)
    for k in range(1, coluna + 1):
        t[k] = k * delta_t

    return u, x, t


# FUNÇÃO LDL1: FUNÇÃO QUE RECEBE DOIS VETORES QUE COMPÕEM A2 E RETORNA UM VETOR L E OUTRO D
# Função usada no método de Crank-Nicolson
def LDL1(diagonal_A, subdiagonal_A):
    linha = len(diagonal_A) + 1

    # Montando o vetor D
    D = np.zeros(linha - 1)
    D[0] = diagonal_A[0]
    for i in range(1, linha - 2):
        D[i] = diagonal_A[i] - (subdiagonal_A[i] ** 2) / D[i - 1]
    D[linha - 2] = diagonal_A[linha - 2] - (subdiagonal_A[linha - 3] ** 2) / D[linha - 3]

    # Montando o vetor L
    L = np.zeros(linha - 2)
    for i in range(linha - 2):
        L[i] = subdiagonal_A[i] / D[i]

    return L, D


# FUNÇÃO solve: RESOLVE O SISTEMA LINEAR USANDO O MÉTODO LDLt PARA UMA MATRIZ TRIDIAGONAL SIMÉTRICA
# Função usada no método de Crank-Nicolson
def solve(b, L, D, linha):
    # Ax = b .: LDLtx = b .: L(DLtx) = b .: Ly = b e DLtx = y

    # lei de recorrência para os valores de y no sistema Ly = b
    y = np.zeros(linha - 1)
    y[0] = b[0]
    for i in range(1, linha - 1):
        y[i] = b[i] - L[i - 1] * y[i - 1]

    # lei de recorrência para os valores de x no sistema DLtx = y
    solucao = np.zeros(linha - 1)
    solucao[linha - 2] = y[linha - 2] / D[linha - 2]
    for i in range(linha - 3, 0, -1):
        solucao[i] = (y[i] - D[i] * L[i] * solucao[i + 1]) / D[i]
    solucao[0] = (y[0] - D[0] * L[0] * solucao[1]) / D[0]

    return solucao


# FUNÇÃO chama_A2: GERANDO UMA MATRIZ TRIDIAGONAL SIMÉTRICA
# Função usada no método de Crank-Nicolson
def chama_A2(linha, lambida):
    # matriz A2
    A2 = np.zeros((linha - 1, linha - 1))

    # definição dos valores da principal principal e subdiagonais
    A2[0][0] = 1 + lambida
    A2[0][1] = - lambida / 2
    for i in range(1, linha - 2):
        A2[i][i - 1] = - (lambida / 2)
        A2[i][i] = 1 + lambida
        A2[i][i + 1] = - (lambida / 2)
    A2[linha - 2][linha - 3] = - (lambida / 2)
    A2[linha - 2][linha - 2] = 1 + lambida

    # retorno de dois vetores que armazenam os valores da diagonal principal e da subdiagonal inferior
    return LDL1(np.array(A2.diagonal()), np.array(A2.diagonal(-1)))


# FUNÇÃO Crank: FUNÇÃO QUE GERA OS VETORES uk PARA CADA PONTO P
# Função usada no método de Crank-Nicolson
def Crank(delta_t, linha, lambida, nf, lista_p):
    h = delta_t
    matiz_u = np.zeros((linha - 1, nf))
    for pontos in range(nf):
        u, x, t = reset(linha, delta_t, delta_t, linha)
        p = lista_p[pontos]
        funcao = np.zeros((linha + 1, linha + 1))
        for tempo in range(linha + 1):
            for posicao in range(1, linha):
                if p - (h / 2) <= x[posicao] <= p + (h / 2):
                    g = (1 / h)
                    funcao[posicao][tempo] = 10 * (1 + math.cos(5 * t[tempo])) * g
                else:
                    funcao[posicao][tempo] = 0

        # gerando o vetor "b" de partida
        b = np.zeros(linha - 1)
        for posicao in range(linha - 1):
            b[posicao] = u[posicao + 1][0] + (delta_t / 2) * (funcao[posicao + 1][0] + funcao[posicao + 1][1]) + (
                    lambida / 2) * (u[posicao][0] - 2 * u[posicao + 1][0] + u[posicao + 2][0])

        # definindo os vetores provenientes da função "chama_A2"
        L, D = chama_A2(linha, lambida)

        # utilizando os vetores "b" na função "solve" obtém-se os vetores "solução"
        # coloca-se os vetores "solução" nas colunas de "u"
        for tempo in range(1, linha):
            solucao = solve(b, L, D, linha)
            u[1:linha, tempo] = solucao
            for posicao in range(linha - 1):
                b[posicao] = solucao[posicao] + (delta_t / 2) * (
                        funcao[posicao + 1][tempo] + funcao[posicao + 1][tempo + 1]) + (lambida / 2) * (
                                     u[posicao][tempo] - 2 * u[posicao + 1][tempo] + u[posicao + 2][tempo])
        solucao = solve(b, L, D, linha)
        u[1:linha, linha] = solucao
        matiz_u[:, pontos] = u[1:linha, linha]

    return matiz_u


# FUNÇÃO uT: FUNÇÃO QUE CRIA OS VALORES DE u_T PARA OS CASOS a E b
def uT(nf, matriz_u, linha, lista_coef):
    u_T = np.zeros(linha - 1)

    # definindo os valores de u_T conforme enunciado
    for i in range(linha - 1):
        for j in range(nf):
            u_T[i] += lista_coef[j] * matriz_u[i, j]

    return u_T


# FUNÇÃO Prod_Escalar: FUNÇÃO QUE FAZ O PRODUTO ESCALAR ENTRE OS VETORES QUE COMPÕEM A matriz_u
def Prod_Escalar(nf, matriz_u, u_T):
    # gerando a matriz dos coeficientes
    P = np.zeros((nf, nf))
    b = np.zeros(nf)

    # montando a matriz dos produtos internos dos coeficientes do sistema
    for j in range(nf):
        for i in range(j, nf):
            P[i, j] = np.dot(matriz_u[:, i], matriz_u[:, j])
            if i != j:
                P[j, i] = P[i, j]

        # montando o vetor dos termos independentes
        b[j] = np.dot(u_T, matriz_u[:, j])

    return b, P


# FUNÇÃO LDL2: FUNÇÃO QUE RECEBE UMA MATRIZ P E RETORNA UM VETOR D E UMA MATRIZ TRINGULAR INFERIOR L
def LDL2(P, nf):
    M = np.copy(P)
    L = np.zeros((nf, nf))
    D = np.ones(nf)

    # lei de reccorência dos elementos que compõem L a partir da decomposição de P
    for i in range(nf):
        for j in range(nf):
            soma = sum([M[i][k] * M[j][k] * D[k] for k in range(j)])
            M[i][j] = (P[i][j] - soma) / D[j]

        # lei de reccorência dos elementos que compõem D
        D[i] = P[i][i] - sum([M[i][k] * M[i][k] * D[k] for k in range(i)])

    # valores que os elementos de L assumem
    for k in range(nf):
        L[k][0:k] = M[k][0:k]
        L[k][k] = 1.0

    return L, D


# FUNÇÃO solve2: RESOLVE O SISTEMA LINEAR USANDO O MÉTODO LDLt PARA UMA MATRIZ QUALQUER
def solve2(P, nf, b):
    x = np.zeros(nf)
    y = np.zeros(nf)
    (L, D) = LDL2(P, nf)

    # lei de recorrência para os valores de y no sistema Ly = b
    y[0] = b[0] / L[0][0]
    for i in range(1, nf):
        soma = 0.0
        for j in range(i):
            soma += L[i][j] * y[j]
        y[i] = (b[i] - soma) / L[i][i]

    # lei de recorrência para os valores de x no sistema DLtx = y
    for i in range(nf - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, nf):
            soma += L[j][i] * x[j]
        x[i] = (y[i] / D[i]) - soma

    return x


def grafico(u_T, linha, nf, matriz_u, delta_x):

    encontrado = np.zeros(len(u_T))
    for i in range(linha - 1):
        soma = 0
        for j in range(nf):
            soma += matriz_u[i][j]
        encontrado[i] = soma

    eixo_x = np.zeros(linha - 1)
    for i in range(linha - 1):
        eixo_x[i] = i * delta_x

    plt.plot(eixo_x, encontrado, label="u_T encontrado")
    plt.plot(eixo_x, u_T, "k--", label="u_T exato", color="orange")
    plt.legend(fontsize="small")
    plt.xlabel("Posição da fonte de calor")
    plt.ylabel("Temperaturas das fontes de calor")
    plt.show()


main()
