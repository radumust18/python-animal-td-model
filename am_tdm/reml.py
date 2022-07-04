import os
from time import time

import numpy as np


class REML:
    def __init__(self, data=None, animal_col=None, ped=None, fixed_effects=None, LHS=None, RHS=None, Ainv=None,
                 Geninv=None, Hinv=None, fixed_degree=None, random_degree=None, method='em', em_steps=10,
                 use_perm=False, G_init=None, R_init=None, P_init=None, use_blupf90_modules=True, ren_file=None,
                 make_blupf90_files=None, maxrounds=None, conv=None):
        """
        Class used to implement estimation of genetic (co)variances by means of various REML methods. For now, it only
        supports a combination of remlf90 and airemlf90 wrappers.
        :param data: dataframe containing animals' phenotypic data
        :param animal_col: the column in data containing the ids of the animal - can be given as a string or as a number
        :param ped: dataframe containing animals' pedigree data - it is mandatory that it has 3 columns, the first one
        being the animals' ID, the second one the ID of the sire and the third one the ID of the dam
        :param fixed_effects: the fixed effects, which are given as an Iterable. Fixed effects can refer to one column
        and can be either categorical or covariate, or refer to concatenation of multiple columns (which will be
        considered categorical). They can also be nested, either one column is nested based on another one, or each
        column for a concatenated effect is nested based on one column
        :param LHS: MME blocks of the left hand side of the equation not related to the (co)variances, unused for
        blupf90 wrappers
        :param RHS: MME blocks of the right hand side of the equation, unused for blupf90 wrappers
        :param Ainv: inverse of A matrix, passed as a numpy.ndarray
        :param Geninv: inverse of G matrix, passed as a numpy.ndarray
        :param Hinv: inverse of H matrix, passed as a numpy.ndarray
        :param fixed_degree: degree of fixed Legendre polynomials, for test-day model
        :param random_degree: degree of random Legendre polynomials, for test-day model
        :param method: can be any of 'em' (EM-REML), 'ai' (AI-REML) or 'ai-em' (AI-REML with initial EM-REML steps)
        :param em_steps: number of initial EM-REML in case of using AI-REML with this option, ignored if method is not
        'ai-em'
        :param use_perm: boolean parameter which tells us whether or not our model for which we estimate the
        (co)variances makes use of permanent effects or not
        :param G_init: initial estimate of additive genetic variance
        :param R_init: initial estimate of residual variance
        :param P_init: initial estimate of permanent variance
        :param use_blupf90_modules: whether or not blupf90 programs should be used for estimation
        :param ren_file: a renaddXX.ped type of file, unused yet
        :param make_blupf90_files: boolean parameter, if True, renumf90 type of files would be built based on our
        pedigree, unused yet
        :param maxrounds: option to set a maximum number of rounds for REML analysis. If None, then default applies
        :param conv: convergente limit for analysis. If None, then default applies
        """
        self.start = time()
        if use_blupf90_modules:
            # First of all, we check the existence of a renf90.par file. If such file does not exist, then renumf90
            # has not been used before, which should have not been the case. Otherwise, we check that the given REML
            # method is valid. The actions taken for EM-REML or AI-REML are straightforward, however, for AI-REML with
            # initial EM-REML steps, we need to check if em_steps is a positive integer and if that is the case, we
            # need to add a line in the renf90.par file so that airemlf90 knows to begin with the given number of
            # EM-REML steps. Depending on the method used, we open the results file (which is either airemlf90.log or
            # remlf90.log). We are only interested in the variances matrices. If we find a 'Genetic variance(s)' line,
            # then we have a matrix of either additive genetic effect or permanent effect. Depending on the value of
            # g_read, we know which matrix we are reading. Because of their placement in the files, it is also simple
            # to check when we are done reading the elements of a matrix (either the end of the file or the beginning of
            # the next variances). For the residual variance(s), the approach is slightly different - we read lines
            # until we meet a line which does not contain only numerical values. That is when we will know that we had
            # read all the values of the residual variance(s). Finally, the matrices are saved in the G, P and R fields,
            # respectively, as numpy 2D arrays.
            # We also check at the very beginning after for a maxrounds option in case renf90.par exists. If maxrounds
            # is not None, we add the option in the renf90.par and delete after REML analysis is finished.
            if os.path.exists('renf90.par'):
                if maxrounds is not None:
                    if type(maxrounds) != int or maxrounds <= 0:
                        raise ValueError('maxrounds should be a positive integer')
                    with open('renf90.par', 'a') as f:
                        f.write('OPTION maxrounds ' + str(maxrounds) + '\n')
                if conv is not None:
                    if (type(conv) != int and type(conv) != float) or conv <= 0:
                        raise ValueError('conv should be a positive number')
                    with open('renf90.par', 'a') as f:
                        f.write('OPTION conv_crit ' + str(conv) + '\n')
                if method == 'em':
                    os.system('remlf90 renf90.par')
                elif method == 'ai':
                    os.system('airemlf90 renf90.par')
                elif method == 'ai-em':
                    if type(em_steps) != int or em_steps <= 0:
                        raise ValueError('em_steps parameter should be a positive integer')
                    with open('renf90.par', 'a') as f:
                        f.write('OPTION EM-REML ' + str(em_steps))
                    os.system('airemlf90 renf90.par')
                else:
                    raise ValueError('Invalid type of REML specified: can be either \'em\' for EM-REML \'ai\' for '
                                     + 'AI-REML or \'ai-em\' for AI-REML with initial EM-REML steps')

                if maxrounds is not None:
                    with open('renf90.par') as f:
                        lines = f.readlines()
                    filtered_lines = list(filter(lambda x: 'OPTION maxrounds ' not in x, lines))
                    with open('renf90.par', 'w') as f:
                        f.writelines(filtered_lines)

                if conv is not None:
                    with open('renf90.par') as f:
                        lines = f.readlines()
                    filtered_lines = list(filter(lambda x: 'OPTION conv_crit ' not in x, lines))
                    with open('renf90.par', 'w') as f:
                        f.writelines(filtered_lines)

                file = 'airemlf90.log' if 'ai' in method else 'remlf90.log'
                with open(file) as f:
                    g_read, p_read, r_read = False, False, False
                    g_list, p_list, r_list = [], [], []
                    for line in f.readlines():
                        if 'Genetic variance(s)' in line:
                            if g_list:
                                p_read, g_read = True, False
                            else:
                                g_read = True
                        elif 'Residual variance(s)' in line:
                            g_read, p_read, r_read = False, False, True
                        elif g_read:
                            try:
                                g_list.extend(list(map(float, line.split())))
                            except ValueError:
                                g_read = False
                        elif p_read:
                            try:
                                p_list.extend(list(map(float, line.split())))
                            except ValueError:
                                p_read = False
                        elif r_read:
                            try:
                                r_list.extend(list(map(float, line.split())))
                            except ValueError:
                                r_read = False

                    size_g, size_p, size_r = int(np.sqrt(len(g_list))), int(np.sqrt(len(p_list))), int(
                        np.sqrt(len(r_list)))
                    self.G = np.array(g_list).reshape(size_g, size_g)
                    if p_list:
                        self.P = np.array(p_list).reshape(size_p, size_p)
                    else:
                        self.P = None
                    self.R = np.array(r_list).reshape(size_r, size_r)
            else:
                raise FileNotFoundError('Could not find renf90.par file - you should use Renum class first before '
                                        + 'REML class')
        self.duration = time() - self.start
