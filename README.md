# Alpha-Tri

Alpha-Tri is a deep neural network to score the intensity similarity using all possible fragment ions, resulting in the improvement in peptide detections.

## Hardware

A [GPU with CUDA support](https://developer.nvidia.com/cuda-gpus)

## Package

- [PyTorch](https://pytorch.org/get-started/locally/#windows-anaconda) 1.0.0+
- [Pyteomics](https://pyteomics.readthedocs.io/en/latest/)
- [numba](http://numba.pydata.org/)

## Tutorial

1. Compile the modified DIA-NN: 
    ```shell script
    cd 'Alpha-Tri/diann_1.8.0_plus_alpha_tri/mstoolkit'
    make
    ```
2. Make a workspace folder containing:
    - diann-alpha-tri.exe compiled by step 1
    - *.mzML, 
    - lib.tsv (the spectral library)
    - prosit.pkl (the predicted result by [Prosit](https://github.com/kusterlab/prosit), containing the headers of ['pr_id', 'spectrum_pred']. 
    The numpy array of spectrum_pred is ordered as y3+, y3++, b3+, b3++ ... y29+, y29++, b29+, b29++.)
    
3. Run DIA-NN:
    ```shell script
   cd workspace
    ./diann-alpha-tri.exe --f *.mzML --lib lib.tsv --out diann_out.tsv --threads 4 --qvalue 0.01
    ```
   Meanwhile, the modified DIA-NN will generate the scores file in workspace.

4. Run Alpha-Tri:
    ```shell script
    cd 'Alpha-Tri/' 
    python main.py workspace_dir
    ```