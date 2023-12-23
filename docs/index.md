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
    
Data preparation
============
Considering that the source data may be organized differently, you should first prepare them into the desired formulas and then run the unified `make_list.py` script
1. Prepare `dict`
   the `dict` file contains all the target keywords and the label of each keyword, the example of `hi xiaowen` likes:
    ```
    keyword1 0
    keyword2 1
    ```
3. Prepare `wav.scp` and `text`

    1.1 `wav.scp` contains the id and path of all the training wavs, it likes:
    ```
    id_of_wav1 /path/to/wav1
    id_of_wav2 /path/to/wav2
    id_of_wav3 /path/to/wav3
    ```

    1.2 `text` contains the id and label of all the training wavs, for the wav that contain keyword, it's label is 1 it likes:
    ```
    conda create -n wekws python=3.8
    conda activate wekws
    ```
4. Install `pytorch`, `torchaudio` and `cudatoolkit`
    ``` sh
    conda install pytorch=1.10.0 torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    ```
5. Clone the repo
    ``` sh
    git clone https://github.com/wenet-e2e/wekws.git
    ```
6. Install requirements
    ``` sh
    cd wekws
    pip install -r requirements.txt
    ```
