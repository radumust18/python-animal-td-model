import os

import numpy as np

from am_tdm.gibbs import Gibbs
from am_tdm.reml import REML
from am_tdm.renum import Renum


def add_sol_se():
    """
    Adds option to compute standard errors among solutions when using BLUPF90, since they can be further used to
    compute other metrics such as PEVs (prediction error variances)
    :return: None
    """
    with open('renf90.par', 'a') as f:
        f.write('OPTION sol se\n')


class AnimalModel:
    def __init__(self, data, animal_col, fixed_effects, trait_cols, ped=None, inbreeding=False, genomic_data=None,
                 ag_variance=None, res_variance=None, pe_variance=None, estimation_method='em-reml', em_steps=10,
                 reml_maxrounds=None, reml_conv=None, rounds=10000, burn_in=1000, sampling=10,
                 use_blupf90_modules=True, export_A=False, export_Ainv=False, export_G=False, export_Ginv=False,
                 export_Hinv=False, export_A22=False, export_A22inv=False):
        """
        Class used to implement the Animal Model with multiple traits and multiple fixed effects, as well as with
        repeated records or not (which translates into with or without permanent environmental effects)
        :param data: dataframe containing animals' phenotypic data
        :param animal_col: the column in data containing the ids of the animal - can be given as a string or as a number
        :param fixed_effects: the fixed effects, which are given as an Iterable. Fixed effects can refer to one column
        and can be either categorical or covariate, or refer to concatenation of multiple columns (which will be
        considered categorical). They can also be nested, either one column is nested based on another one, or each
        column for a concatenated effect is nested based on one column
        :param trait_cols: the columns in data associated to traits - the columns are given as an Iterable object and
        can be either strings or numbers
        :param ped: dataframe containing animals' pedigree data - it is mandatory that it has 3 columns, the first one
        being the animals' ID, the second one the ID of the sire and the third one the ID of the dam
        :param inbreeding: boolean parameter which tells whether inbreeding should be accounted for or not
        :param genomic_data: dataframe containing animals' genotypic data, if available
        :param ag_variance: initial additive genetic variance, if it exists
        :param res_variance: initial residual variance, if it exists
        :param pe_variance: initial permanent genetic variance, if it exists
        :param estimation_method: can be any of 'em-reml', 'ai-reml', 'ai-em-reml' or 'gibbs'
        :param em_steps: number of initial EM-REML steps (used only if estimation_method='ai-em-reml')
        :param reml_maxrounds: option to set a maximum number of rounds for REML analysis. It applies only if
        estimation_method is 'em-reml', 'ai-reml' or 'ai-em-reml'
        :param reml_conv: option to set a convergence limit for REML analysis. It applies only if
        estimation_method is 'em-reml', 'ai-reml' or 'ai-em-reml'
        :param rounds: the number of total rounds of Gibbs sampling, used only if estimation_method = 'gibbs'
        :param burn_in: the number of burn in rounds, used only if estimation_method = 'gibbs'
        :param sampling: the sampling number (if sampling = n, each nth sample will be considered for the final
        estimate), used only if estimation_method = 'gibbs'
        :param use_blupf90_modules: whether or not to use BLUPF90 modules
        """

        # Firstly, the pedigree is renumbered and reordered
        self.renum = Renum(data, animal_col, ped, inbreeding=inbreeding, genomic_data=genomic_data,
                           use_blupf90_modules=use_blupf90_modules, trait_cols=trait_cols, fixed_effects=fixed_effects,
                           res_variance=res_variance, ag_variance=ag_variance, pe_variance=pe_variance,
                           export_A=export_A, export_Ainv=export_Ainv, export_A22=export_A22,
                           export_A22inv=export_A22inv, export_G=export_G, export_Ginv=export_Ginv,
                           export_Hinv=export_Hinv)
        self.estimation_method = estimation_method
        self.ag_variance = ag_variance
        self.res_variance = res_variance
        self.pe_variance = pe_variance
        self.use_blupf90_modules = use_blupf90_modules
        self.em_steps = em_steps
        self.reml_maxrounds = reml_maxrounds
        self.reml_conv = reml_conv
        self.rounds = rounds
        self.burn_in = burn_in
        self.sampling = sampling
        self.fixed_effects = fixed_effects
        self.inbreeding = inbreeding
        self.FE = []
        self.EBVs = np.zeros((len(trait_cols), self.renum.animal_count))
        self.PERMs = np.zeros((len(trait_cols), self.renum.animal_count))
        self.PEVs = np.zeros((len(trait_cols), self.renum.animal_count))
        self.RELs = np.zeros((len(trait_cols), self.renum.animal_count))
        self.heritabilities = np.zeros(len(trait_cols))
        self.repeatabilities = np.zeros(len(trait_cols))
        self.DRPs = np.zeros((len(trait_cols), self.renum.animal_count))
        self.DRP_RELs = np.zeros((len(trait_cols), self.renum.animal_count))
        self.DRP_weights = np.zeros((len(trait_cols), self.renum.animal_count))

        if use_blupf90_modules:
            self.__estimate_parameters__()
            self.__add_updated_genetic_parameters__()
            add_sol_se()
            self.__add_accuracy__()
            os.system('blupf90+ renf90.par')
            self.__read_blupf90_solutions__()
            self.__read_accuracies__()

        self.__compute_heritabilities__()
        for i in range(len(trait_cols)):
            self.__compute_DRPs__(i)

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
                                           maxrounds=self.reml_maxrounds, conv=self.reml_conv)
        elif self.estimation_method == 'ai-reml':
            self.variance_estimator = REML(data=self.renum.new_data, animal_col=self.renum.animal_col,
                                           ped=self.renum.new_ped, fixed_effects=self.renum.fixed_effects,
                                           Ainv=self.renum.Ainv, Geninv=self.renum.Ginv, Hinv=self.renum.Hinv,
                                           method='ai', G_init=self.ag_variance, P_init=self.pe_variance,
                                           R_init=self.res_variance, use_blupf90_modules=self.use_blupf90_modules,
                                           maxrounds=self.reml_maxrounds, conv=self.reml_conv)
        elif self.estimation_method == 'ai-em-reml':
            self.variance_estimator = REML(data=self.renum.new_data, animal_col=self.renum.animal_col,
                                           ped=self.renum.new_ped, fixed_effects=self.renum.fixed_effects,
                                           Ainv=self.renum.Ainv, Geninv=self.renum.Ginv, Hinv=self.renum.Hinv,
                                           method='ai-em', em_steps=self.em_steps, G_init=self.ag_variance,
                                           P_init=self.pe_variance, R_init=self.res_variance,
                                           use_blupf90_modules=self.use_blupf90_modules,
                                           maxrounds=self.reml_maxrounds, conv=self.reml_conv)
        elif self.estimation_method == 'gibbs':
            self.variance_estimator = Gibbs(data=self.renum.new_data, animal_col=self.renum.animal_col,
                                            ped=self.renum.new_ped, fixed_effects=self.renum.fixed_effects,
                                            Ainv=self.renum.Ainv, Geninv=self.renum.Ginv, Hinv=self.renum.Hinv,
                                            rounds=self.rounds, burn_in=self.burn_in, sampling=self.sampling,
                                            G_init=self.ag_variance, P_init=self.pe_variance,  R_init=self.res_variance,
                                            use_blupf90_modules=self.use_blupf90_modules)
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
            line = f.readline()
            while line:
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
                    file_lines.append(' '.join(map(str, self.R[residual_line])) + '\n')
                    residual_line += 1
                    if residual_line >= self.R.shape[0]:
                        read_residual = False
                        for _ in range(((self.R.shape[0] - 1) // 7) * self.R.shape[0]):
                            line = f.readline()
                elif read_additive:
                    file_lines.append(' '.join(map(str, self.G[additive_line])) + '\n')
                    additive_line += 1
                    if additive_line >= self.G.shape[0]:
                        read_additive = False
                        for _ in range(((self.G.shape[0] - 1) // 7) * self.G.shape[0]):
                            line = f.readline()
                elif read_permanent:
                    file_lines.append(' '.join(map(str, self.P[permanent_line])) + '\n')
                    permanent_line += 1
                    if permanent_line >= self.P.shape[0]:
                        read_permanent = False
                        for _ in range(((self.P.shape[0] - 1) // 7) * self.P.shape[0]):
                            line = f.readline()
                else:
                    file_lines.append(line)
                line = f.readline()
        with open('renf90.par', 'w') as f:
            f.writelines(file_lines)

    def __read_blupf90_solutions__(self):
        """
        Reads the solutions from the same named file generated by BLUPF90. For each trait, the first len(fixed_effects)
        effects will be the fixed effects solutions and thus, they will be added to the FE list. Otherwise, the effect
        numbered len(fixed_effects) + 1 will represent the EBV for the animal with renumbered id equal to the level and
        the effect numbered len(fixed_effects) + 2 will represent the permanent environmental effect for the animal
        with renumbered id equal to the level
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
                se = float(values[4])
                if effect <= len(self.fixed_effects):
                    self.FE.append(solution)
                else:
                    if effect == len(self.fixed_effects) + 1:
                        self.EBVs[trait - 1, level - 1] = solution
                        self.PEVs[trait - 1, level - 1] = se ** 2
                    else:
                        self.PERMs[trait - 1, level - 1] = solution
                line = f.readline()
            self.FE = np.array(self.FE)

    def __add_accuracy__(self):
        """
        Adds the option to compute reliabilities of EBVs. In order to do this, we must know the index of the effect
        associated to EBVs. This is easy to compute as the number of fixed effects plus 1. An option to ignore
        inbreeding is also added if the inbreeding parameter is set to False
        :return: None
        """
        with open('renf90.par', 'a') as f:
            f.write('OPTION store_accuracy ' + str(len(self.fixed_effects) + 1) + '\n')
            if not self.inbreeding:
                f.write('OPTION correct_accuracy_by_inbreeding_direct 0\n')

    def __read_accuracies__(self):
        """
        Reads the reliabilities from the acc_bf90 file generated by blupf90. The reliability is the last value of a line
        from acc_bf90. The other values are trait, the chosen effect and the solution of that effect (the same from the
        solutions file)
        :return:
        """
        with open('acc_bf90') as f:
            for line in f.readlines():
                values = line.strip().split()
                trait = int(values[0])
                animal = int(values[2])
                reliability = float(values[-1])
                self.RELs[trait - 1, animal - 1] = reliability

    def __compute_heritabilities__(self):
        """
        Heritabilities are computed by dividing the additive variance of each trait over the phenotypic variance of
        each trait. The phenotypic variance will be computed by adding the additive variance and residual variance and
        also the permanent one if permanent effects are considered. Heritabilities are then computed by simply dividing
        each element in the main diagonal of the additive variance matrix to each element in the main diagonal of the
        sum of variance matrices. Repeatabilities are also computed, although if there are no permanent effects
        considered, they will be equal to heritabilities
        :return: None
        """
        variance_sum = self.G + self.R
        repeat_sum = self.G
        if self.P is not None:
            variance_sum += self.P
            repeat_sum += self.P
        self.heritabilities = self.G.diagonal() / variance_sum.diagonal()
        self.repeatabilities = repeat_sum.diagonal() / variance_sum.diagonal()

    def __read_deproofsf90_solutions__(self):
        """
        Reads the solutions from deproofs file. These will contain the DRPs we want. For each line, the DRP value will
        be the fourth one, according to deproofsf90 output. The value before the DRP will be the index of the
        associated animal
        :return: None
        """
        with open('deproofs') as f:
            for line in f.readlines():
                values = line.strip().split()
                trait = int(values[0])
                animal = int(values[2])
                solution = float(values[3])
                self.DRPs[trait - 1, animal - 1] = solution

    def __compute_DRPs__(self, trait_idx):
        """
        Computes DRPs based on Garrick's article and based on the R implementation that can be found at
        https://github.com/camult/DRP
        :return:
        """
        r2_gm = (np.where(self.renum.sires > 0, self.RELs[trait_idx - 1, self.renum.sires - 1], 0)
                 + np.where(self.renum.dams > 0, self.RELs[trait_idx - 1, self.renum.dams - 1], 0)) / 4
        alfa = 1 / (0.5 - r2_gm)
        delta = (0.5 - r2_gm) / (1 - self.RELs[trait_idx, :])
        alfa_delta = (alfa ** 2) + (16 / delta)
        lambda_star = (1 - self.heritabilities[trait_idx]) / self.heritabilities[trait_idx]
        Zlgm_Zgm = lambda_star * (0.5 * alfa - 4) + 0.5 * lambda_star * np.sqrt(alfa_delta)
        Zli_Zi = delta * Zlgm_Zgm + 2 * lambda_star * (2 * delta - 1)
        r2i = 1 - lambda_star / (Zli_Zi + lambda_star)
        gm = (np.where(self.renum.sires > 0, self.EBVs[trait_idx - 1, self.renum.sires - 1], 0)
              + np.where(self.renum.dams > 0, self.EBVs[trait_idx - 1, self.renum.dams - 1], 0)) / 2
        y1 = -2 * lambda_star * gm + (Zli_Zi + 2 * lambda_star) * self.EBVs[trait_idx, :]
        DRP = y1 / Zli_Zi
        wi = (1 - self.heritabilities[trait_idx]) / ((0.5 + (1 - r2i) / r2i) * self.heritabilities[trait_idx])
        self.DRPs[trait_idx, :] = np.where(wi > 0.0, DRP, 0.0)
        self.DRP_RELs[trait_idx, :] = np.where(wi > 0.0, r2i, 0.0)
        self.DRP_weights[trait_idx, :] = np.where(wi > 0.0, wi, 0.0)
