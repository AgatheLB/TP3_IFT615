# -*- coding: utf-8 -*-

#####
# leba3207
####

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

import numpy as np


class Correcteur:
    def __init__(self, p_init, p_transition, p_observation, int2letters, letters2int):
        '''Correcteur de frappes dans un mot.

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
        '''
        self.p_init = p_init
        self.p_transition = p_transition
        self.p_observation = p_observation
        self.int2letters = int2letters
        self.letters2int = letters2int

    def corrige(self, mot):
        '''Corrige les frappes dans un mot.

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
        '''

        # TODO: Implémenter un correcteur de frappes dans un mot basé sur un HMM.
        #       Vous aurez besoin des variable suivantes :
        #       self.p_init
        #       self.p_transition
        #       self.p_observation
        #       self.int2letters
        #       self.letters2int

        # Retourne le mot sans correction avec une probabilité de 0.0 (.~= À MODIFIER =~.)
        return mot, 0.0
