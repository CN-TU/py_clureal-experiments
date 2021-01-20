<<<<<<< HEAD
# Refinement of Clustering (CluReAL.v2)
### TU Wien, CN group
### FIV, Jan 2021

## Dependencies

1. Check python dependencies (*requirements.txt*).

        $ python3 install-dep.py

## CluReAL.v2 

2. CluReAL refinement and SK ideograms quick example.

        $ python3 toy_example.py

## Material for experiment replicability 

Experiments consist on comparing the performance of (a) the best clustering obtained from parameter search around ideal parameters (according to the GT), with (b) CluReAL refining a suboptimal clustering obtained with arbitrary parameters. 

3. **Datasets**. Datasets used in the experiments that were not generated with MDCGen (https://github.com/CN-TU/mdcgen-matlab) must be downloaded from the author's website: 

    http://cs.uef.fi/sipu/datasets/

We provide here only the corresponding GT (labels). Please download the datasets and prepare the files as comma-separated-values without headers, e.g., (for a 2D-dataset):

        53920,42968,8
        52019,42206,8
        52570,42476,8
        ...

Later, place them in the corresponding folder before running the scripts. In the following table missing datasets are listed with their corresponding folder, GT_file name, expected final name and the direct link to the original source.

|  Folder  -------    |  GT file name  -------       |  Final name  ------- |  Link to the orig. source  -------------------------   |
|-:-:-------- |-:-:---------------- | -:-:------------ | -:-:--------------------------------------------|
| *data2d/*  | *a2_GT.csv*         | *a2.csv*         | http://cs.uef.fi/sipu/datasets/a2.txt           |
| *data2d/*  | *a3_GT.csv*         | *a3.csv*         | http://cs.uef.fi/sipu/datasets/a3.txt           |
| *data2d/*  | *s1_GT.csv*         | *s1.csv*         | http://cs.uef.fi/sipu/datasets/s1.txt           |
| *data2d/*  | *s2_GT.csv*         | *s2.csv*         | http://cs.uef.fi/sipu/datasets/s2.txt           |
| *data2d/*  | *s3_GT.csv*         | *s3.csv*         | http://cs.uef.fi/sipu/datasets/s3.txt           |
| *data2d/*  | *unbalance_GT.csv*  | *unbalance.csv*  | http://cs.uef.fi/sipu/datasets/unbalance.txt    |
| *dataMd/*  | *multidim_0002_GT*  | *multidim_0002*  | http://cs.uef.fi/sipu/datasets/data_dim_txt.zip |
| *dataMd/*  | *multidim_0003_GT*  | *multidim_0003*  | http://cs.uef.fi/sipu/datasets/data_dim_txt.zip |
| *dataMd/*  | *multidim_0005_GT*  | *multidim_0005*  | http://cs.uef.fi/sipu/datasets/data_dim_txt.zip |
| *dataMd/*  | *multidim_0010_GT*  | *multidim_0010*  | http://cs.uef.fi/sipu/datasets/data_dim_txt.zip |
| *dataMd/*  | *multidim_0015_GT*  | *multidim_0015*  | http://cs.uef.fi/sipu/datasets/data_dim_txt.zip |
| *dataMd/*  | *multidim_0032_GT*  | *multidim_0032*  | http://cs.uef.fi/sipu/datasets/dim032.txt       |
| *dataMd/*  | *multidim_0064_GT*  | *multidim_0064*  | http://cs.uef.fi/sipu/datasets/dim064.txt       |
| *dataMd/*  | *multidim_0256_GT*  | *multidim_0256*  | http://cs.uef.fi/sipu/datasets/dim256.txt       |
| *dataMd/*  | *multidim_0512_GT*  | *multidim_0512*  | http://cs.uef.fi/sipu/datasets/dim512.txt       |
| *dataMd/*  | *multidim_1024_GT*  | *multidim_1024*  | http://cs.uef.fi/sipu/datasets/dim1024.txt      |

Alternatively, you can remove these datasets from the scripts. 
   
4. **CluReAL vs k-Sweep (2D, k-means)**. Comparison of clustering optimization methods with 2D-data and k-means algorithm. Plots are saved in the [plots] folder.

        $ python3 2d_comparison_kmk.py

4. **CluReAL vs k-Sweep (multiD, partitional clust.)**. Comparison of clustering optimization methods with Multi-dimensional data and partitional algorithms. Results are saved in the [results] folder.

        $ python3 Md_comparison_k.py

4. **CluReAL vs Random Search (multiD, density-based clust.)**. Comparison of clustering optimization methods with Multi-dimensional data and density-based algorithms. Results are saved in the [results] folder.

        $ python3 Md_comparison_n.py


=======
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
>>>>>>> c182d4069ead97e8078f49ef0493d0ff9a1ea816
