Large-scale Islanded Microgrids based on Multi-agent Reinforcement Learning Control Methods
===============
- Built by Dong Chen from Michigan State University
- Started on Feb.25, 2020

Overview
-------
We plan to develop a power grid simulation platform with large number of DERs. Instead of the conventional control methods, we want to try multi-agent reinforcement learning algorithms. 

Problems we are targeting at are voltage and frequency stabilization and power sharing among DERs.


Code Structure
--------
- [main.py](main.py): the main function used to run the whole project. In this file, you can choose the DER and system configurations.
- [DER_fn.py](DER_fn.py): This is the graph generation function and will be called by the main.py function.
- [configs/parameters_4.py](configs/parameters_4.py): This is the configuration file for the 4 DER system.
- [configs/parameters_20.py](configs/parameters_20.py): This is the configuration file for the 20 DER system.

In the implementation, we should build a similar configuration file and list the system configurations. We can claim our system design in the main function and keep the graph generation function unchanged.


Experiments & Analysis
---------
- The first experiment is the frequency and voltage control for 4-DER and 20-DER microgrid systems. 

We adapt the DER system proposed in reference [1]. The architecture is given by Fig.1.

<p align="center">
     <img src="docs/DER_4.png" alt="output_example" width="60%" height="60%">
     <br>Fig.1 Single-line diagram of 4-DER microgrid test system.
</p>

<p align="center">
     <img src="results - Vnom/DER_4.png" alt="output_example" width="60%" height="60%">
     <br>Fig.1 Frequency and voltage of 20 DER system
</p>


<p align="center">
     <img src="docs/DER_20.png" alt="output_example" width="60%" height="60%">
     <br>Fig.1 Single-line diagram of 20-DER microgrid test system.
</p>

<p align="center">
     <img src="results - Vnom/DER_20.png" alt="output_example" width="60%" height="60%">
     <br>Fig.2 Frequency and voltage of 20 DER system
</p>

Reference
---------
1. Bidram, Ali, Ali Davoudi, and Frank L. Lewis. "A multiobjective distributed control framework for islanded AC microgrids." IEEE Transactions on industrial informatics 10.3 (2014): 1785-1798.

2. Bidram, Ali, et al. "Distributed cooperative secondary control of microgrids using feedback linearization." IEEE Transactions on Power Systems 28.3 (2013): 3462-3470.

3. Mustafa, Aquib, et al. "Detection and Mitigation of Data Manipulation Attacks in AC Microgrids." IEEE Transactions on Smart Grid (2019).
