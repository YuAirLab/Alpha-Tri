# Alpha-Tri

Alpha-Tri is a deep neural network to score the intensity similarity using all possible fragment ions, resulting in the improvement in peptide detections.

## Hardware

A [GPU with CUDA support](https://developer.nvidia.com/cuda-gpus)

## Package
- [PyTorch](https://pytorch.org/get-started/locally/#windows-anaconda) 1.0.0+
- [Pyteomics](https://pyteomics.readthedocs.io/en/latest/)
- [numba](http://numba.pydata.org/)
- [Prosit](https://github.com/kusterlab/prosit)

## Example

1. Compile the modified DIA-NN: 
    ```shell script
    cd Alpha-Tri/DIA-NN_v1.7.12/mstoolkit
    make
    ```
   This will generate diann-alpha.exe in the same path.
   
2. Make a workspace folder containing:
    - diann-alpha.exe compiled by step 1
    - HeLa-1h.mzML, (test data could be downloaded from [figshare](https://figshare.com/projects/Alpha-Tri/128000) 
    or [PXD005573](https://www.ebi.ac.uk/pride/archive/projects/PXD005573))
    - lib.tsv (this spectral library could be downloaded from [figshare](https://figshare.com/projects/Alpha-Tri/128000)
    or [Pan-Human library, SAL00023](https://db.systemsbiology.net/sbeams/cgi/PeptideAtlas/GetDIALibs))
   
3. Run Prosit to predict the MS2 for each precursor in lib
    ```shell script
    cd Alpha-Tri/Prosit
    python prosit.py --lib wsorkspace_dir/lib.tsv
    ``` 
    This will append the predicted MS2 to each precursor and store the result to lib.pkl.
    
4. Run DIA-NN:
    ```shell script
    cd workspace
    ./diann-alpha.exe --f *.mzML --lib lib.tsv --out diann_out.tsv --threads 4 --qvalue 0.01
    ```
   Meanwhile, the modified DIA-NN will generate the scores file in workspace.

5. Run Alpha-Tri:
    ```shell script
    cd Alpha-Tri/Alpha-Tri
    python main.py workspace_dir --tri (post-scoring only by Alpha-Tri)
    python main.py workspace_dir --xic (post-scoring only by Alpha-XIC)
    python main.py workspace_dir --tri --xic (post-scoring by Alpha-Tri & Alpha-XIC)
    ```
    Finally, we get the identification and quantitative result, alpha_out.tsv, in the workspace folder.
    