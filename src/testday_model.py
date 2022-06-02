import os
import numpy as np

from gibbs import Gibbs
from reml import REML
from renum import Renum


class TestDayModel:
    def __init__(self, data, animal_col, lactation_col, dim_col, fixed_effects, trait_cols, ped, inbreeding=False,
                 dim_range=None, fixed_degree=4, random_degree=2, genomic_data=None, ag_variance=None,
                 res_variance=None, pe_variance=None, estimation_method='em-reml', em_steps=10, rounds=5000,
                 burn_in=1000, sampling=10, sampling_print=10, use_blupf90_modules=False):
        """
        Class used to implement the Test-Day Model with multiple traits and multiple fixed effects and variable DIM
        range
        :param data: dataframe containing animals' phenotypic data
        :param animal_col: the column in data containing the ids of the animal - can be given as a string or as a number
        :param lactation_col: used for test day model, represents the column in data which contains the lactation for
        each record, can be given as a string or as a number
        :param dim_col: used for test day model, represents the column in data which contains the DIM for each record,
        can be given as a string or as a number
        :param fixed_effects: the fixed effects, which are given as an Iterable. Fixed effects can refer to one column
        and can be either categorical or covariate, or refer to concatenation of multiple columns (which will be
        considered categorical). They can also be nested, either one column is nested based on another one, or each
        column for a concatenated effect is nested based on one column
        :param trait_cols: the columns in data associated to traits - the columns are given as an Iterable object and
        can be either strings or numbers
        :param ped: dataframe containing animals' pedigree data - it is mandatory that it has 3 columns, the first one
        being the animals' ID, the second one the ID of the sire and the third one the ID of the dam
        :param inbreeding: boolean parameter which tells whether inbreeding should be accounted for or not
        :param dim_range: used for test day model, a pair of numbers specifying the DIM interval (for example, (5, 310))
        :param fixed_degree: used for test day model, represents the degree of the fixed Legendre polynomial
        :param random_degree: used for test day model, represents the degree of the random Legendre polynomial
        :param genomic_data: dataframe containing animals' genotypic data, if available
        :param ag_variance: initial additive genetic variance, if it exists
        :param res_variance: initial residual variance, if it exists
        :param pe_variance: initial permanent genetic variance, if it exists
        :param estimation_method: can be any of 'em-reml', 'ai-reml', 'ai-em-reml' or 'gibbs'
        :param em_steps: number of initial EM-REML steps (used only if estimation_method='ai-em-reml')
        :param rounds: the number of total rounds of Gibbs sampling, used only if estimation_method = 'gibbs'
        :param burn_in: the number of burn in rounds, used only if estimation_method = 'gibbs'
        :param sampling: the sampling number (if sampling = n, each nth sample will be considered for the final
        estimate), used only if estimation_method = 'gibbs'
        :param sampling_print: the sampling print number (if sampling_print = n, each nth sample will be printed on the
        screen), used only if estimation_method = 'gibbs'
        :param use_blupf90_modules: whether or not to use BLUPF90 modules
        """

        # Firstly, the pedigree is renumbered and reordered
        self.renum = Renum(data, animal_col, ped, inbreeding=inbreeding, genomic_data=genomic_data,
                           use_blupf90_modules=use_blupf90_modules, trait_cols=trait_cols, fixed_effects=fixed_effects,
                           res_variance=res_variance, ag_variance=ag_variance, pe_variance=pe_variance,
                           lactation_col=lactation_col, dim_col=dim_col, dim_range=dim_range, fixed_degree=fixed_degree,
                           random_degree=random_degree)
        self.estimation_method = estimation_method
        self.ag_variance = ag_variance
        self.res_variance = res_variance
        self.pe_variance = pe_variance
        self.use_blupf90_modules = use_blupf90_modules
        self.em_steps = em_steps
        self.rounds = rounds
        self.burn_in = burn_in
        self.sampling = sampling
        self.sampling_print = sampling_print
        self.fixed_degree = fixed_degree
        self.random_degree = random_degree
        self.G = None
        self.P = None
        self.R = None
        self.fixed_effects = fixed_effects
        self.FE = []
        self.fixed_curve_coefficients = []
        self.additive_coefficients = np.zeros((len(trait_cols), self.renum.animal_count, self.random_degree + 1))
        self.permanent_coefficients = np.zeros((len(trait_cols), self.renum.animal_count, self.random_degree + 1))

        if use_blupf90_modules:
            self.__estimate_parameters__()
            self.__add_updated_genetic_parameters__()
            os.system('blupf90 renf90.par')

        self.scaled_dim_range = np.arange(dim_range[0], dim_range[1] + 1)
        self.scaled_dim_range = -1 + 2 * (self.scaled_dim_range - dim_range[0]) / (self.scaled_dim_range - dim_range[1])

    def __estimate_parameters__(self):
        """
        Applies the chosen estimation method. Values of the additive genetic variance, residual variance and permanent
        environmental variance (if used) will be saved in fields of the current instance
        :return: None
        """
        self.variance_estimator = None
        if self.estimation_method == 'em-reml':
            self.variance_estimator = REML(data=self.renum.new_data, animal_col=self.renum.animal_col,
                                           ped=self.renum.new_ped, fixed_effects=self.renum.fixed_effects,
                                           Ainv=self.renum.Ainv, Geninv=self.renum.Ginv, Hinv=self.renum.Hinv,
                                           method='em', G_init=self.ag_variance, P_init=self.pe_variance,
                                           R_init=self.res_variance, use_blupf90_modules=self.use_blupf90_modules,
                                           fixed_degree=self.fixed_degree, random_degree=self.random_degree)
        elif self.estimation_method == 'ai-reml':
            self.variance_estimator = REML(data=self.renum.new_data, animal_col=self.renum.animal_col,
                                           ped=self.renum.new_ped, fixed_effects=self.renum.fixed_effects,
                                           Ainv=self.renum.Ainv, Geninv=self.renum.Ginv, Hinv=self.renum.Hinv,
                                           method='ai', G_init=self.ag_variance, P_init=self.pe_variance,
                                           R_init=self.res_variance, use_blupf90_modules=self.use_blupf90_modules,
                                           fixed_degree=self.fixed_degree, random_degree=self.random_degree)
        elif self.estimation_method == 'ai-em-reml':
            self.variance_estimator = REML(data=self.renum.new_data, animal_col=self.renum.animal_col,
                                           ped=self.renum.new_ped, fixed_effects=self.renum.fixed_effects,
                                           Ainv=self.renum.Ainv, Geninv=self.renum.Ginv, Hinv=self.renum.Hinv,
                                           method='ai-em', em_steps=self.em_steps, G_init=self.ag_variance,
                                           P_init=self.pe_variance, R_init=self.res_variance,
                                           use_blupf90_modules=self.use_blupf90_modules, fixed_degree=self.fixed_degree,
                                           random_degree=self.random_degree)
        elif self.estimation_method == 'gibbs':
            self.variance_estimator = Gibbs(data=self.renum.new_data, animal_col=self.renum.animal_col,
                                            ped=self.renum.new_ped, fixed_effects=self.renum.fixed_effects,
                                            Ainv=self.renum.Ainv, Geninv=self.renum.Ginv, Hinv=self.renum.Hinv,
                                            rounds=self.rounds, burn_in=self.burn_in, sampling=self.sampling,
                                            sampling_print=self.sampling_print, G_init=self.ag_variance,
                                            P_init=self.pe_variance, R_init=self.res_variance,
                                            use_blupf90_modules=self.use_blupf90_modules,
                                            fixed_degree=self.fixed_degree, random_degree=self.random_degree)
        elif self.estimation_method is not None:
            raise ValueError('Invalid genetic parameters estimation method')
        else:
            if (self.ag_variance is None) or (self.res_variance is None)\
                    or (self.pe_variance is None and self.renum.has_perm):
                raise ValueError('When not using an estimation method, genetic (co)variances should be given')
        if self.variance_estimator is not None:
            self.G = self.variance_estimator.G
            self.P = self.variance_estimator.P
            self.R = self.variance_estimator.R
        else:
            self.G = self.ag_variance
            self.P = self.pe_variance
            self.R = self.res_variance

    def __add_updated_genetic_parameters__(self):
        """
        When using BLUPF90, genetic, environmental and residual parameters need to be updated in the renf90.par file,
        because initially it contains dummy values. Having the parameters saved as fields, we iterate through each line
        of the file. When we reach a 'RANDOM_RESIDUAL VALUES', we will replace the next lines with the value of the
        residual (co)variances matrix. When we reach the first '(CO)VARIANCES' line, we will replace the next lines with
        the additive genetic (co)variances matrix, and finally, if we find a second '(CO)VARIANCES' line, we will
        replace the next lines with the permanent environmental (co)variances matrix
        :return: None
        """
        file_lines = []
        read_residual, read_additive, read_permanent = False, False, False
        residual_line, additive_line, permanent_line = 0, 0, 0
        with open('renf90.par') as f:
            for line in f.readlines():
                if 'RANDOM_RESIDUAL VALUES' in line:
                    read_residual = True
                    file_lines.append(line)
                elif '(CO)VARIANCES' in line:
                    if additive_line > 0:
                        read_permanent = True
                    else:
                        read_additive = True
                    file_lines.append(line)
                elif read_residual:
                    if residual_line >= self.R.shape[0]:
                        read_residual = False
                    else:
                        file_lines.append(' '.join(map(str, self.R[residual_line])) + '\n')
                        residual_line += 1
                elif read_additive:
                    if additive_line >= self.G.shape[0]:
                        read_additive = False
                    else:
                        file_lines.append(' '.join(map(str, self.G[additive_line])) + '\n')
                        additive_line += 1
                elif read_permanent:
                    if permanent_line >= self.P.shape[0]:
                        read_permanent = False
                    else:
                        file_lines.append(' '.join(map(str, self.P[permanent_line])) + '\n')
                        permanent_line += 1
                else:
                    file_lines.append(line)
        with open('renf90.par', 'w') as f:
            f.writelines(file_lines)

    def __read_blupf90_solutions__(self):
        """
        Reads the solutions from the same named file generated by BLUPF90. For each trait, the first len(fixed_effects)
        effects will be the fixed effects solutions and thus, they will be added to the FE list. Otherwise, the effects
        from len(fixed_effects) + 1 to len(fixed_effects) + fixed_degree + 1 will represent the fixed lactation curve
        coefficients. The effects from len(fixed_effects) + fixed_degree + 2 to len(fixed_effects) + fixed_degree +
        random_degree + 2 will represent the random additive coefficients for each animal. Finally, the effects from
        len(fixed_effects) + fixed_degree + random_degree + 3 to len(fixed_effects) + fixed_degree + 2 * random_degree +
        3 will represent the random permanent enviromental coefficients for each animal.
        :return: None
        """
        with open('solutions') as f:
            # Ignores the first line, which is only a header
            line = f.readline()

            line = f.readline()
            while line:
                values = line.strip().split()
                trait = int(values[0])
                effect = int(values[1])
                level = int(values[2])
                solution = float(values[3])
                if effect <= len(self.fixed_effects):
                    self.FE.append(solution)
                else:
                    if effect <= len(self.fixed_effects) + self.fixed_degree + 1:
                        self.fixed_curve_coefficients.append(solution)
                    elif effect <= len(self.fixed_effects) + self.fixed_degree + self.random_degree + 2:
                        self.additive_coefficients[trait - 1, level - 1,
                                                   effect - len(self.fixed_effects) - self.fixed_degree - 2] = solution
                    else:
                        self.permanent_coefficients[trait - 1, level - 1,
                                                    effect - len(self.fixed_effects) - self.fixed_degree
                                                    - self.random_degree - 3] = solution
                line = f.readline()
            self.FE = np.array(self.FE)
            self.fixed_curve_coefficients = np.array(self.fixed_curve_coefficients)
