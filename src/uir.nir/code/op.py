from mealpy.swarm_based.SFO import BaseSFO
from mealpy.swarm_based.HHO import BaseHHO
from mealpy.swarm_based.PSO import BasePSO
from mealpy.swarm_based.WOA import BaseWOA
from mealpy.swarm_based.FireflyA import BaseFireflyA
from mealpy.swarm_based.SpaSA import BaseSpaSA

import numpy as np
from opfunu.cec.cec2014.function import *
from opfunu.type_based import *
from opfunu.cec.cec2014.unconstraint2 import Model as MD2
from opfunu.cec.cec2014.unconstraint import Model as MD

problem_size = 10
epoch = 5

for i in range(0, 30):
    solution = np.random.uniform(0, 1, problem_size)

    result = MD(problem_size)
    func = MD(problem_size).F2

    print("F2 MD solution", result.F2(solution))

    print("______BaseSpaSA_______")

    temp1 = BaseSpaSA(func, problem_size=10, domain_range=(-100,
                                                           100), log=True, epoch=epoch, pop_size=50)
    temp1.train()

    print("______BaseHHO_______")
    temp2 = BaseHHO(func, problem_size=10, lb=-100, ub=100,
                    log=True, epoch=epoch, pop_size=50)
    temp2.train()

    print("______BaseFireflyA_______")
    temp2 = BaseFireflyA(func, problem_size=10, lb=-100, ub=100,
                         log=True, epoch=epoch, pop_size=50)
    temp2.train()

    print("______BaseACOR_______")

    temp2 = BaseWOA(func, problem_size=10, domain_range=(-100,
                                                         100), log=True, epoch=epoch, pop_size=50)
    temp2.train()
