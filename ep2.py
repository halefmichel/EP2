# Nomes: Halef Michel (10774994) e Lucas Leone (10278868)

import math
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Definindo variáveis
    T = 1
    linha = int(input('Digite o valor de N: '))
    delta_x = 1.0 / linha
    delta_t = delta_x
    lambida = linha
    nf = int(input('Digite a quantidade de pontos p: '))
    lista_p = np.zeros(nf)
    for i in range(nf):
        str = 'Digite o valor do coeficiente de p%d: ' % (i + 1)
        lista_p[i] = float(input(str))

    # Matriz A dos coeficientes do método de Crank-Nicolson
    A = np.zeros((linha - 1, linha - 1))
    A[0][0] = 1 + 2 * lambida
    A[0][1] = -lambida
    for i in range(1, linha - 2):
        A[i][i - 1] = -lambida
        A[i][i] = 1 + 2 * lambida
        A[i][i + 1] = -lambida
    A[linha - 2][linha - 3] = -lambida
    A[linha - 2][linha - 2] = 1 + 2 * lambida

    # Chamada das funções
    matriz_u = Crank(delta_t, linha, lambida, nf, lista_p)
    uT(nf, matriz_u, linha)
    Prod_Escalar(nf, matriz_u)


# Função que reseta os valores da matriz u e das listas x e t
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


# Função P2a: função que recebe dois vetores que compõem A e retorna um vetor L e outro D
# A matriz A foi junto com as variáveis
def P2a(diagonal_A, subdiagonal_A):
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


# Função solve: resolve o sistema linear usando o método LDLt
def solve(b, L, D, linha):
    # Ax = b .: LDLtx = b .: L(DLtx) = b .: Ly = b e DLtx = y

    y = np.zeros(linha - 1)
    y[0] = b[0]
    for i in range(1, linha - 1):
        y[i] = b[i] - L[i - 1] * y[i - 1]

    solucao = np.zeros(linha - 1)
    solucao[linha - 2] = y[linha - 2] / D[linha - 2]
    for i in range(linha - 3, 0, -1):
        solucao[i] = (y[i] - D[i] * L[i] * solucao[i + 1]) / D[i]
    solucao[0] = (y[0] - D[0] * L[0] * solucao[1]) / D[0]

    return solucao


# Matriz A2
# Matriz tridiagonal simétrica usada no método de Crank-Nicolson na parte 2 do EP
def chama_A2(linha, lambida):
    # matriz A2
    A2 = np.zeros((linha - 1, linha - 1))
    A2[0][0] = 1 + lambida
    A2[0][1] = - lambida / 2
    for i in range(1, linha - 2):
        A2[i][i - 1] = - (lambida / 2)
        A2[i][i] = 1 + lambida
        A2[i][i + 1] = - (lambida / 2)
    A2[linha - 2][linha - 3] = - (lambida / 2)
    A2[linha - 2][linha - 2] = 1 + lambida

    return P2a(np.array(A2.diagonal()), np.array(A2.diagonal(-1)))


# Função Crank gera os vetores uk para cada ponto p, utilizando o método de Crank-Nicolson
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

        # Gerando o vetor "b" de partida
        b = np.zeros(linha - 1)
        for posicao in range(linha - 1):
            b[posicao] = u[posicao + 1][0] + (delta_t / 2) * (funcao[posicao + 1][0] + funcao[posicao + 1][1]) + (
                    lambida / 2) * (u[posicao][0] - 2 * u[posicao + 1][0] + u[posicao + 2][0])

        # Definindo os vetores provenientes da função "chama_A2"
        L, D = chama_A2(linha, lambida)

        # Utilizando os vetores "b" na função "solve" obtém-se os vetores "solução"
        # Coloca-se os vetores "solução" nas colunas de "u"
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


def uT(nf, matriz_u, linha):
    lista_coef = np.zeros(nf)
    u_T = np.zeros(linha - 1)

    for i in range(nf):
        str = 'Digite o valor do coeficiente de u%d: ' % (i + 1)
        lista_coef[i] = float(input(str))

    for i in range(linha - 1):
        for j in range(nf):
            u_T[i] += lista_coef[j] * matriz_u[i, j]

    return u_T


def Prod_Escalar(nf, matriz_u):
    # Gerando a matriz dos coeficientes
    P = np.zeros((nf, nf))
    b = np.zeros(nf)
    # Montando a matriz dos produtos internos dos coeficientes do sistema
    for j in range(nf):
        for i in range(j, nf):
            P[i, j] = np.dot(matriz_u[:, i], matriz_u[:, j])
            if i != j:
                P[j, i] = P[i, j]

        b[j] = np.dot(matriz_u[:, -1], matriz_u[:, j])


main()
