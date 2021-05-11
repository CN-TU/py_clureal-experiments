# Refinement of Clustering (CluReAL.v2)
### TU Wien, CN group
### FIV, May 2021

<br>

## Dependencies

Check python dependencies (*requirements.txt*).

        $ python3 install-dep.py

<br>

## CluReAL.v2 

CluReAL refinement and SK ideograms quick example.

        $ python3 toy_example.py

<br>

## Material for experiment replicability 

Experiments consist on comparing the performance of (a) the best clustering obtained from parameter search around ideal parameters (according to the GT), with (b) CluReAL refining a suboptimal clustering obtained with arbitrary parameters. 

<br>

- **Datasets**. Datasets used in the experiments that were not generated with MDCGen (https://github.com/CN-TU/mdcgen-matlab) must be downloaded from the author's website: 

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

**Real datasets** (i.e., real_1, real_2, real_3, real_4) are obtained from the scikit.learn package and transformed with tSNE. The script to extract such datasets are in the [extra] folder:

        $ python3 extract_real.py 

<br>
   
- **CluReAL vs k-Sweep (2D, k-means)**. Comparison of clustering optimization methods with 2D-data and k-means algorithm. Plots are saved in the [plots] folder.

        $ python3 2d_comparison_kmk.py

<br>

- **CluReAL vs k-Sweep (multiD, partitional clust.)**. Comparison of clustering optimization methods with Multi-dimensional data and partitional algorithms. Results are saved in the [results] folder.

        $ python3 Md_comparison_k.py

<br>

- **CluReAL vs Random Search (multiD, density-based clust.)**. Comparison of clustering optimization methods with Multi-dimensional data and density-based algorithms. Results are saved in the [results] folder.

        $ python3 Md_comparison_n.py

<br>

- **CluReAL.v1 vs CluReAL.v2 (multiD, k-means)**. Comparison of CluReAL.v1 and CluReAL.v2 for refining suboptimal k-means clustering with Multi-dimensional data. Results are saved in the [results] folder.

        $ python3 clureal_v1vs2_comparison.py

<br>

- **Sensitivity analysis (number of samples), CluReAL vs k-Sweep**. Comparison of runtime requirements of clustering optimization methods with Multi-dimensional datasets of different sizes (n={500,1000,2500,5000,10000,25000}) and partitional algorithms. Results are saved in the [results] folder.

        $ python3 runtime_comparison_k.py

<br>

- **Sensitivity analysis (number of samples), CluReAL vs Random Search**. Comparison of runtime requirements of clustering optimization methods with Multi-dimensional datasets of different sizes (n={500,1000,2500,5000,10000,25000}) and density-based  algorithms. Results are saved in the [results] folder.

        $ python3 runtime_comparison_n.py

<br>

- **CluReAL options vs high-overlap**. Study of CluReAL alternatives to deal with high overlap: (a) more strict kinship-based edge-pruning during refinement, (b) using coresets. Plots are saved in the [plots] folder.

        $ python3 high-overlap.py
