# -*- coding: utf-8 -*-

#####
# leba3207
####

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

import numpy as np


class Correcteur:
    def __init__(self, p_init, p_transition, p_observation, int2letters, letters2int):
        """Correcteur de frappes dans un mot.

        Modèle de Markov caché (HMM) permettant de corriger les erreurs de frappes
        dans un mot. La correction est dictée par l'inférence de l'explication
        la plus pausible du modèle.

        Parameters
        ------------
        p_init : array-like shape (N,)
                 Probabilités initiales de l'état caché à la première position.

        p_transition : array-like shape (X,Y)
                       Modèle de transition.

        p_observation : array-like shape (X,Y)
                        Modèle d'observation.

        int2letters : list
                      Associe un entier (l'indice) à une lettre.

        letters2int : dict
                      Associe une lettre (clé) à un entier (valeur).
        """
        self.p_init = p_init
        self.p_transition = p_transition
        self.p_observation = p_observation
        self.int2letters = int2letters
        self.letters2int = letters2int

    def corrige(self, mot):
        """Corrige les frappes dans un mot.

        Retourne la correction du mot donné et la probabilité p(mot, mot corrigé).

        Parameters
        ------------
        mot : string
              Mot à corriger.

        Returns
        -----------
        mot_corrige : string
                      Le mot corrigé.

        prob : float
               Probabilité dans le HMM du mot observé et du mot corrigé.
               C'est-à-dire 'p(mot, mot_corrige)'.
        """

        T = len(mot)
        transitions, alpha = self.calcul_alpha(mot, T, self.p_init.size)
        best_prob, int_best_prob = self.get_best_proba(alpha, T)
        best_probs = self.get_best_probas(alpha, T)

        corrected_word = ''
        for b, i in best_probs:
            print(self.int2letters[int(i)])
            corrected_word += self.int2letters[int(i)]

        path = []
        val = (int_best_prob, T-1)
        for i in reversed(range(T)):
            for e in transitions:
                if e[1] == val:
                    path.append(e)
                    val = e[0]
        path.reverse()

        # word correction
        correct_word = list(mot)
        for p in range(T-1):
            correct_word[p] = path[p][0][0]
        correct_word[T-1] = path[T-2][1][0]  # last letter

        for f in range(T):
            correct_word[f] = self.int2letters[correct_word[f]]
        correct_word = ''.join(correct_word)

        return correct_word, best_prob

    def calcul_alpha(self, mot, T, nb_classes):
        alpha = np.zeros((T, nb_classes))
        transitions = []
        for t in range(T):
            letter = mot[t]
            letter_int = self.letters2int.get(letter)
            if t == 0:
                alpha[t] = self.p_init[letter_int] * self.p_observation[letter_int]
            else:
                for i in range(nb_classes):
                    p_transitions = np.zeros(nb_classes)
                    for j in range(nb_classes):
                        p_transitions[j] = self.p_transition[i, j] * alpha[t-1, j]
                    max_p_trans = np.amax(p_transitions)
                    class_max_p_trans = np.argmax(p_transitions)
                    alpha[t, i] = self.p_observation[letter_int, i] * max_p_trans
                    transitions.append(((class_max_p_trans, t-1), (i, t)))
        return transitions, alpha

    def get_best_proba(self, alpha, T):
        best_prob = np.amax(alpha[T-1])
        int_best_prob = np.argmax(alpha[T-1])
        return best_prob, int_best_prob

    def get_best_probas(self, alpha, T):
        best_prob = np.zeros((T, 2))
        for i in range(T):
            p = np.amax(alpha[i])
            int_p = np.argmax(alpha[i])
            best_prob[i] = [p, int_p]
        return best_prob
