# -*- coding: utf-8 -*-

#####
# leba3207
###

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

import numpy as np


#################################
# Solution serpents et échelles #
#################################

#####
# calcul_valeur: Fonction qui retourne le tableau de valeurs d'un plan (politique).
#
# mdp: Spécification du processus de décision markovien (objet de la classe SerpentsEchelles, héritant de MDP).
#
# plan: Un plan donnant l'action associée à chaque état possible (dictionnaire).
#
# retour: Un tableau Numpy 1D de float donnant la valeur de chaque état du mdp, selon leur ordre dans mdp.etats.
###
def calcul_valeur(mdp, plan):
    A = np.zeros((len(mdp.etats), len(mdp.etats)))

    b = mdp.recompenses.copy()
    for s, action in plan.items():
        transitions = mdp.modele_transition[(s, action)]
        for t in transitions:
            A[s, t[0]] = mdp.recompenses[t[0]] + mdp.escompte * t[1] * A[s, t[0]]

    # A = np.linalg.inv(A)
    v = A.dot(b)
    # v = np.linalg.solve(A, b)
    return v
    # return np.zeros(len(mdp.etats))


#####
# calcul_plan: Fonction qui retourne un plan à partir d'un tableau de valeurs.
#
# mdp: Spécification du processus de décision markovien (objet de la classe SerpentsEchelles, héritant de MDP).
#
# valeur: Un tableau de valeurs pour chaque état (tableau Numpy 1D de float).
#
# retour: Un plan (dictionnaire) qui maximise la valeur future espérée, en fonction du tableau "valeur".
### 
def calcul_plan(mdp, valeur):
    plan = dict()
    plan_values = dict()
    for s in mdp.etats:
        for a in mdp.actions[0]:
            val_next_s = mdp.recompenses[s]
            for t in mdp.modele_transition[(s, a)]:
                val_next_s += mdp.escompte * t[1] * valeur[t[0]]
            if round(val_next_s, 2) >= round(valeur[s], 2):
                if plan.get(s) is None:
                    plan[s] = a
                    plan_values[s] = val_next_s
                else:
                    if val_next_s > plan_values[s]:
                        plan[s] = a
                        plan_values[s] = val_next_s
    return plan


#####
# iteration_politiques: Algorithme d'itération par politiques, qui retourne le plan optimal et sa valeur.
#
# plan_initial: Le plan à utiliser pour initialiser l'algorithme d'itération par politiques.
#
# retour: Un tuple contenant le plan optimal et son tableau de valeurs.
### 
def iteration_politiques(mdp, plan_initial):

    return plan_initial, calcul_valeur(mdp, plan_initial)
