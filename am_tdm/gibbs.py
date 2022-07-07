import subprocess
import os
from time import time

import numpy as np


class Gibbs:
    def __init__(self, data=None, animal_col=None, ped=None, fixed_effects=None, LHS=None, RHS=None, Ainv=None,
                 Geninv=None, Hinv=None, fixed_degree=None, random_degree=None, rounds=10000, burn_in=1000,
                 sampling=10, use_perm=False, G_init=None, R_init=None, P_init=None,
                 use_blupf90_modules=True, ren_file=None, make_blupf90_files=None):
        """
        Class used to implement estimation of genetic (co)variances by means of Gibbs sampling. For now, it only
        supports a gibbs2f90 wrapper.
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
        :param rounds: the number of total rounds of Gibbs sampling
        :param burn_in: the number of burn in rounds
        :param sampling: the sampling number (if sampling = n, each nth sample will be considered for the final
        estimate)
        :param use_perm: boolean parameter which tells us whether or not our model for which we estimate the
        (co)variances makes use of permanent effects or not
        :param G_init: initial estimate of additive genetic variance
        :param R_init: initial estimate of residual variance
        :param P_init: initial estimate of permanent variance
        :param use_blupf90_modules: whether or not blupf90 programs should be used for estimation
        :param ren_file: a renaddXX.ped type of file, unused yet
        :param make_blupf90_files: boolean parameter, if True, renumf90 type of files would be built based on our
        pedigree, unused yet
        """
        self.start = time()
        if use_blupf90_modules:
            # First of all, we check the existence of a renf90.par file. If such file does not exist, then renumf90
            # has not been used before, which should have not been the case. Otherwise, we check that the given REML
            # method is valid. We check if the rounds, burn_in, sampling and sampling_print values are all positive
            # integers. We also check that the number of rounds is larger than the sum of burn_in and sampling (for
            # obvious reasons). The final estimates are printed on screen and are delimited by lines containing the
            # 'ave G' string for both additive genetic variance and permanent variance, along with 'ave R' string for
            # the residual variance. When we find 'ave G', depending on whether we already have stored values in the
            # g_list variable or not, we know whether we are going to read the additive genetic variance or the
            # permanent variance. We also have three variables g_read, p_read and r_read which tell us which matrix we
            # are reading at a given moment, so that we know which one of g_list, p_list and r_list should be updated.
            # If we meet a line containing 'SD G', we know we are not reading any of the additive genetic and permanent
            # (co)variances matrix anymore, and the same applies to the residual variance matrix when reading the line
            # containing 'SD R'. Finally, we store the results in numpy 2D arrays, more precisely in the fields G, P and
            # R.
            #
            # In order to be able to capture the output printed by gibbs2f90, we need to use subprocess library.
            # However, we also want the lines to still be printed on the screen, basically to preserve the
            # functionality. This is why we are printing each line stored in the subprocess pipe, which also helps when
            # detecting the lines of interest
            if os.path.exists('renf90.par'):
                if type(rounds) != int or rounds <= 0:
                    raise ValueError('rounds parameter should be a positive integer')
                if type(burn_in) != int or burn_in <= 0:
                    raise ValueError('burn_in parameter should be a positive integer')
                if type(sampling) != int or sampling <= 0:
                    raise ValueError('sampling parameter should be a positive integer')
                if rounds <= burn_in:
                    raise ValueError('rounds value should be higher than burn_in value')
                if burn_in + sampling > rounds:
                    raise ValueError('Not enough rounds to get sample based on burn_in and sampling parameters')
                process = subprocess.Popen(['gibbsf90+', 'renf90.par', '--rounds', str(rounds), '--burnin',
                                            str(burn_in), '--interval', str(sampling)],
                                           stdout=subprocess.PIPE)

                g_read, p_read, r_read = False, False, False
                g_list, p_list, r_list = [], [], []
                for line in iter(process.stdout.readline, b''):
                    str_line = line.decode()
                    if 'ave G' in str_line:
                        if g_list:
                            p_read = True
                        else:
                            g_read = True
                    elif 'ave R' in str_line:
                        r_read = True
                    elif 'SD G' in str_line:
                        g_read, p_read = False, False
                    elif 'SD R' in str_line:
                        r_read = False
                    elif g_read:
                        g_list.extend(list(map(float, str_line.split())))
                    elif p_read:
                        p_list.extend(list(map(float, str_line.split())))
                    elif r_read:
                        r_list.extend(list(map(float, str_line.split())))
                    print(str_line)

                size_g, size_p, size_r = int(np.sqrt(len(g_list))), int(np.sqrt(len(p_list))), int(np.sqrt(len(r_list)))
                self.G = np.array(g_list).reshape(size_g, size_g)
                if p_list:
                    self.P = np.array(p_list).reshape(size_p, size_p)
                else:
                    self.P = None
                self.R = np.array(r_list).reshape(size_r, size_r)
            else:
                raise FileNotFoundError('Could not find renf90.par file - you should use Renum class first before '
                                        + 'Gibbs class')
        self.duration = time() - self.start
