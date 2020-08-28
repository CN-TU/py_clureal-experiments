# py_clureal-experiments
Algorithm for Interpretability and Refinement of Clustering. CN contact: Félix Iglesias 

Aug 2020, FIV

**Material for the experiment replicability of scientific paper...**

Iglesias, F., Zseby, T., & Zimek, A. (2020). *Interpretability and Refinement of Clustering*. The 7th DSAA 2020 - IEEE International Conference on Data Science and Advanced Analytics (to be presented and published).


**Replication instructions**

1. CluReAL quick example:

        $ python3 clureal.py


2. 2D experiments:

        $ bash run2d.sh

    - It can take some minutes.
    - Log report is saved in the "results/2d-report.txt" file.


3. Multi-dimensional experiments:

        $ bash runMd.sh

    - It can take some hours (even a day). Parallelization is recommended.
    - Results are saved in the [results] folder.


4. Check python package dependencies

        $ python3 install-dep.py


5. Folders [results-2d-Jun2020] and [results-multidim-Jun2020] contain the results used in the paper

