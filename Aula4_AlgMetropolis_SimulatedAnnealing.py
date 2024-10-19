from math import e, pow
from random import random

def criterioParadaSA(countTemperatura, countMelhoria, maxMetropolis):
    return countTemperatura < maxMetropolis or countMelhoria < 100

def Metropolis(fo: function, vizinhos: list, maxMetropolis: int, s, T: float):
    melhorSolucao = s
    countSemMelhoria: int = 0
    iterT = 0

    while criterioParadaSA(iterT, countSemMelhoria, maxMetropolis):
        iterT = iterT + 1
        vizinho = None  # TO DO: criar estrutura de vizinhança && criar busca na vizinhança
        deltaCusto = fo(vizinho) - fo(s)
        if deltaCusto <= 0:
            s = vizinho
            if fo(vizinho) <= fo(melhorSolucao):    # Movimento de melhora
                melhorSolucao = vizinho
            elif random() < pow(e, -deltaCusto/T):  # Movimento de piora com aceitação
                s = vizinho
                countSemMelhoria = countSemMelhoria + 1
            else:   # Movimento de piora sem aceitação
                countSemMelhoria = countSemMelhoria + 1
    
    return melhorSolucao

def geraAlfa() -> float:
    aleatorio = random()
    if aleatorio > 0.8 and aleatorio < 0.99: return aleatorio
    
    return geraAlfa()

def taxaResfriamentoGeometrico(temperatura: float):
    alfa = geraAlfa()
    return alfa*temperatura

def taxaResfriamentoRapido(temperatura: float, temperaturaInicial: float):
    beta = 1    # TO DO: gerar o Beta
    return temperatura/(1+beta*temperatura)

def taxaResfriamentoHajek(temperatura):
    # TO DO
    return 0

def SimulatedAnnealing(fo: function, vizinhos: list, maxMetropolis: int, s, T0 = 300.0, Tf = 0.1):
    solucao = s
    T = T0

    while T < Tf:
        solucao = Metropolis(fo, vizinhos, maxMetropolis, s, T)        
        T = taxaResfriamentoGeometrico(T) * T

    return solucao

if __name__ == "__main__":
    SimulatedAnnealing({}, [], 100, None)
