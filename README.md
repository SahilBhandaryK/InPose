# InPose

This is the official implementation of the ICLR 2026 paper: Zero-shot Human Pose Estimation using Diffusion-based Inverse solvers.<br>

## Dependencies and Installation

- Python >= 3.10 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch == 2.0.1  TorchVision == 0.15.2](https://pytorch.org/)
- NVIDIA GPU + [CUDA v11.8](https://developer.nvidia.com/cuda-downloads)

1. Clone repo

    ```bash
    git clone https://github.com/SahilBhandaryK/InPose
    ```

2. Install dependent packages

    ```bash
    cd InPose
    conda env create -f env.yaml
    ```


## Dataset Preparation

- Please refer to [this repo](https://github.com/eth-siplab/AvatarPoser#datasets) for details about the dataset organization.
- For Testing, we use HumanEva and Transitions_mocap.
- You also need to download the SMPL model files.
- Your downloaded AMASS dataset and SMPL models folders should look as follows:
  
  .
├── data/
│   ├── dataset_raw/
│   │   └── AMASS
│   │   │   └── HumanEva/
│   │   │   └── Transitions_mocap/
├── support_data/
│   ├── body_models/
│   │   └── smplh/
│   │   │   └── male/
│   │   │   │   └── model.npz
│   │   │   └── female/
│   │   │   │   └── model.npz
│   │   └── dmpls/
│   │   │   └── male/
│   │   │   │   └── model.npz
│   │   │   └── female/
│   │   │   │   └── model.npz
└── README.md

- Run the following scipt to convert the AMASS data to the required form:

    ```bash
    python prepare_data.py
    ```
    Note separate flags for preparing the training and testing datasets.
- Please note that the training dataset preperation uses global joint rotations, and testing uses local joint rotations.
- Options are provided in the respective .json files in the options/ folder.

## Train

- **Training command**: 

    ```bash
    python train.py
    ```

- **Pre-trained SR model**: Find the pre-trained SR model at [Drive](https://drive.google.com/drive/folders/1Bih3uOU9ZeQoxyIgcaH7rrugSDYw7xax?usp=sharing), and place it in the InPose_best/ folder.

## Test

- **Running Experiments**:
- Experiments can be run using the following command:
  
    ```bash
    python test.py
    ```
    Please modify the bone lengths and scale multiplier in the script as necessary. Joint names are also provided in the script.

- To Compile the Results run the following scipt with the experiments and the algorithm as arguments:
  
    ```bash
    python Compile_results.py
    ```

## Visualization

- Use the Visualize.ipynb jupyter notebook to Visualize the results using Open3d.
- To Install Open3D for GPU, use the following resources:
    [Open3D](https://www.open3d.org/docs/latest/compilation.html)

## Citations

If InPose helps your research, please consider citing us.<br>

``` latex
@inproceedings{bhandary2026InPose,
  title={Zero-shot Human Pose Estimation using Diffusion-based Inverse solvers},
  author={Bhandary Karnoor, Sahil and Roy Choudhury, Romit},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year={2026}
}
```

Find other resources in our [webpage](https://iclrinpose-crypto.github.io/ICLRInPose/).

## License and Acknowledgement

This project borrows heavily from [BoDIffusion](https://github.com/openai/guided-diffusion) and [Human Body Prior](https://pypi.org/project/human-body-prior/), we thank the authors for their contributions to the community.<br>

## Contact

If you have any question, please email `sahilb5@illinois.edu`.