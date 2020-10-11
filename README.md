# CSOS - Quick Summary

1. This repository covers a guide to and implementation of CSOS. CSOS, a metaheuristic optimization algorithm based on the Symbiotic Organisms Search (SOS), is mainly designed for high-dimensional and complex problems (e.g., 1000 dimensions). Nonetheless, CSOS also performs well on less complicated problems than SOS and other metaheuristic algorithms.

2. The CSOS method only requires one parameter, which is the population (ecosystem) size. This specialty is a significant advantage of using CSOS, as the user does not need to tune the parameter setting. This specialty is uncommon in metaheuristic algorithms. Other metaheuristic algorithms, such as Genetic Algorithm, Particle Swarm Optimization, Differential Evolutions, and Grey Wolf Optimizer, need two or more parameters. In this case, the user needs to find the best parameter settings before performing the algorithm.

3. As reported in the article below, besides its specialty with only one parameter, CSOS also has a better searching quality and searching efficiency than other metaheuristic algorithms. Our experimental results also show that CSOS can alleviate the dimensionality issues in high-dimensional problems, which exists for most methods in the literature.

4. The CSOS code (written in Python) is available at https://github.com/sutrisnohendri/CSOS/blob/main/CSOSv01.py

Please follow the reference below for further information.
- Chao-Lung Yang and Hendri Sutrisno, (2020). A clustering-based symbiotic organisms search for high dimensional optimization problems, Applied Soft Computing, 106722. DOI: 10.1016/j.asoc.2020.106722. [[LINK]](https://www.sciencedirect.com/science/article/abs/pii/S1568494620306608)
