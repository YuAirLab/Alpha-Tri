# Alpha-Tri

Alpha-Tri is a deep neural network to score the intensity similarity using all possible fragment ions, resulting in the improvement in peptide detections.

## Hardware

A [GPU with CUDA support](https://developer.nvidia.com/cuda-gpus)

## Main Packages
- [PyTorch](https://pytorch.org/get-started/locally/#windows-anaconda)
- [Pyteomics](https://pyteomics.readthedocs.io/en/latest/)
- [numba](http://numba.pydata.org/)
- [Prosit](https://github.com/kusterlab/prosit)
- [tensorflow](https://www.tensorflow.org/install)

## Example on a local PC (win10, NVIDIA GPU 1060)

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

3. Configure the operating environment by conda 
    ```shell script
    conda create -n alpha python=3.6 numpy=1.18 pandas=1.0 numba scikit-learn --yes
    conda activate alpha 
    conda install -c bioconda pyteomics python=3.6 --yes
    conda install tensorflow-gpu=1.11 keras=2.2.4 pytorch=1.1.0 cudatoolkit=9.0 -c pytorch --yes
    ```
    Of note, as Prosit is trained by tf1.11, advanced version of tf and CUDA may not be compatible. 
    
4. Run Prosit to predict the MS2 for each precursor in lib
    ```shell script
    cd Alpha-Tri/Prosit
    python prosit.py --lib workspace_dir/lib.tsv
    ``` 
    This will append the predicted MS2 to each precursor and store the result to lib.pkl.
    
4. Run DIA-NN:
    ```shell script
    cd workspace
    ./diann-alpha.exe --f *.mzML --lib lib.tsv --out diann_out.tsv --threads 4 --qvalue 0.01
    or
    diann-alpha.exe --f *.mzML --lib lib.tsv --out diann_out.tsv --threads 4 --qvalue 0.01 
    ```
   Meanwhile, the modified DIA-NN will generate the scores file in workspace.

5. Run Alpha-Tri:
    ```shell script
    cd Alpha-Tri/Alpha-Tri
    python main.py -ws workspace_dir --tri (post-scoring only by Alpha-Tri)
    python main.py -ws workspace_dir --xic (post-scoring only by Alpha-XIC)
    python main.py -ws workspace_dir --tri --xic (post-scoring by Alpha-Tri & Alpha-XIC)
    ```
    Finally, we get the identification and quantitative result, alpha_out.tsv, in the workspace folder.

## Example on Google Colab Pro (linux, NVIDIA GPU P100)
see [alpha_tri_on_Colab.ipynb](https://github.com/YuAirLab/Alpha-Tri/blob/master/alpha_tri_on_Colab.ipynb)

## Troubleshooting
```shell script
1. AttributeError: 'str' object has no attribute 'decode'
```

```shell script
2. TypeError: add_weight() got multiple values for argument 'name'
```

These may be raised by the version incompatibility of Keras in Prosit (see the discussions in [https://github.com/keras-team/keras/issues/14265](https://github.com/keras-team/keras/issues/14265) and [https://github.com/keras-team/keras/issues/13540](https://github.com/keras-team/keras/issues/13540)). You may have to install the same version of Keras: 
```shell script
conda install tensorflow-gpu=1.11 keras=2.2.4 pytorch=1.1.0 cudatoolkit=9.0 -c pytorch --yes
```




    
    