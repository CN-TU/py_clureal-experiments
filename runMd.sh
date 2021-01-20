
python3 install-dep.py

python3 -u multidim_cluster_comparison.py separated bench 0 | tee results/separated_bench.txt
python3 -u multidim_cluster_comparison.py separated mkm 0 | tee results/separated_mkm.txt
python3 -u multidim_cluster_comparison.py separated ahc 0 | tee results/separated_ahc.txt
python3 -u multidim_cluster_comparison.py separated gmm 0 | tee results/separated_gmm.txt

python3 -u multidim_cluster_comparison.py low-noise bench 0 | tee results/low-noise_bench.txt
python3 -u multidim_cluster_comparison.py low-noise mkm 0 | tee results/low-noise_mkm.txt
python3 -u multidim_cluster_comparison.py low-noise ahc 0 | tee results/low-noise_ahc.txt
python3 -u multidim_cluster_comparison.py low-noise gmm 0 | tee results/low-noise_gmm.txt

python3 -u multidim_cluster_comparison.py close bench 0 | tee results/close_bench.txt
python3 -u multidim_cluster_comparison.py close mkm 0 | tee results/close_mkm.txt
python3 -u multidim_cluster_comparison.py close ahc 0 | tee results/close_ahc.txt
python3 -u multidim_cluster_comparison.py close gmm 0 | tee results/close_gmm.txt

python3 -u multidim_cluster_comparison.py high-noise bench 0 | tee results/high-noise_bench.txt
python3 -u multidim_cluster_comparison.py high-noise mkm 0 | tee results/high-noise_mkm.txt
python3 -u multidim_cluster_comparison.py high-noise ahc 0 | tee results/high-noise_ahc.txt
python3 -u multidim_cluster_comparison.py high-noise gmm 0 | tee results/high-noise_gmm.txt

python3 -u multidim_cluster_comparison.py complex bench 0 | tee results/complex_bench.txt
python3 -u multidim_cluster_comparison.py complex mkm 0 | tee results/complex_mkm.txt
python3 -u multidim_cluster_comparison.py complex ahc 0 | tee results/complex_ahc.txt
python3 -u multidim_cluster_comparison.py complex gmm 0 | tee results/complex_gmm.txt

python3 -u multidim_cluster_comparison.py dens-diff bench 0 | tee results/dens-diff_bench.txt
python3 -u multidim_cluster_comparison.py dens-diff mkm 0 | tee results/dens-diff_mkm.txt
python3 -u multidim_cluster_comparison.py dens-diff ahc 0 | tee results/dens-diff_ahc.txt
python3 -u multidim_cluster_comparison.py dens-diff gmm 0 | tee results/dens-diff_gmm.txt

