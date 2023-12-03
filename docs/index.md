Installation
============
We recommend that you use the following steps to install the dependencies.
1. Install `Conda`

    1.1 Install Miniconda or Anaconda, please refer to: https://mirror.tuna.tsinghua.edu.cn/help/anaconda/

    1.2 Create Conda env:
    ``` sh
    conda create -n wekws python=3.8
    conda activate wekws
    ```
2. Install `pytorch`, `torchaudio` and `cudatoolkit`
    ``` sh
    conda install pytorch=1.10.0 torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    ```
3. Clone the repo
    ``` sh
    git clone https://github.com/wenet-e2e/wekws.git
    ```
4. Install requirements
    ``` sh
    cd wekws
    pip install -r requirements.txt
    ```
    
