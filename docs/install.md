### Set up the Python environment

Initialize python environment by running:

```shell
conda create -n instant-nvr python=3.9
conda activate instant-nvr
```

Then, install pytorch3d=0.7.2 according to the [instructions here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

Finally, install other packages by running:

```shell
pip install -r requirements.txt
```

### Set up datasets

For both datasets, we refine the camera parameters. See below for further details.

#### ZJU-MoCap dataset

Since the dataset is licensed, we require the users to agree with some policies before obtaining the dataset. Anyone who wants access to the dataset can fill out this [form](https://forms.gle/rBSXkXpuSsHep4R26) to obtain the instructions to download the data. You can also alternatively fill in this [agreement](https://pengsida.net/project_page_assets/files/Refined_ZJU-MoCap_Agreement.pdf) and email it to [Chen Geng](mailto:chen.geng@cs.stanford.edu) with cc to [Sida Peng](mailto:pengsida@zju.edu.cn) and [Xiaowei Zhou](mailto:xwzhou@zju.edu.cn) to obtain the access.

**Please note, even if you have previously downloaded the ZJU-MoCap dataset from our [previous work](https://github.com/zju3dv/neuralbody), it is essential to re-download it. We have refined the dataset to include more accurate camera parameters and additional auxiliary files that are crucial for running our code.**

> If you've sent an email and have not received a response within three days, there's a possibility that your email may have been overlooked. We kindly request you resend the email as a reminder.

After acquiring the link, set up the dataset by:

```shell
ROOT=/path/to/instant-nvr
mkdir -p $ROOT/data
cd $ROOT/data
ln -s /path/to/my-zjumocap zju-mocap
```

#### MonoCap dataset

Following [animatable_nerf](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md#monocap-dataset), the dataset is composed by [DeepCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/) and [DynaCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/), which forbids further distribution. 

Please download the raw data [here](https://gvv-assets.mpi-inf.mpg.de/) and email [Chen Geng](mailto:chen.geng@cs.stanford.edu) with cc to [Sida Peng](mailto:pengsida@zju.edu.cn) for instructions on how to process this dataset.

After successfully obtaining the dataset, set up it by:

```shell
ROOT=/path/to/instant-nvr
mkdir -p $ROOT/data
cd $ROOT/data
ln -s /path/to/monocap monocap
```

