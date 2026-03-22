# Project 3

**Student:** Andre Barle 

**Class:** CS7180 - Advanced Computer Vision  

**Project:** Project 3 - Supervised Monocular Depth Estimation on a Single Image via Semantic Object Conditioning and a Shared Transformer Backbone

**OS:** macOS (Apple Silicon M1/M2/M3) with MPS support  

## Hardware
Apple M4 Max OSX Sequoia 15.6, 64 GB RAM

## Project Overview

This project implements monocular depth estimation via a shared transformer backbone between a semantic specific decoder and a depth specific decoder and shared refinement head to perform depth estimation from a single sRGB image transformed into a pseudo-log chroma historgram and reconstructed as output. Unlike the previous project it works better and only needs an sRGB image for prediction. 

### Required Packages
you can copy this into a .yml file for an immediate environment using your environment manager. I am using conda.
name: tri-transformer
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.11
  - pytorch=2.4.1
  - torchvision=0.19.1
  - torchaudio=2.4.1
  - numpy=1.26.4
  - pillow=10.4.0
  - matplotlib=3.9.2
  - tqdm=4.66.5

you might need this depending on your Pillow installation as well (for the demo file if you want to test a .heif image):
conda install conda-forge::pillow-heif


in your shell:
conda env create -f tri-transformer.yml
conda activate tri-transformer

Full list of depenencies (not needed but its exhaustive):

  # Name                    Version                   Build  Channel
aom                       3.9.1                h7bae524_0    conda-forge
appnope                   0.1.4              pyhd8ed1ab_1    conda-forge
asttokens                 3.0.0              pyhd8ed1ab_1    conda-forge
blas                      1.0                    openblas  
bzip2                     1.0.8                h80987f9_6  
ca-certificates           2025.12.2            hca03da5_0  
cairo                     1.18.4               h191e429_0  
comm                      0.2.3              pyhe01879c_0    conda-forge
contourpy                 1.3.3                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
dav1d                     1.2.1                h80987f9_0  
debugpy                   1.8.16          py311h0962b89_0  
decorator                 5.2.1              pyhd8ed1ab_0    conda-forge
exceptiongroup            1.3.0              pyhd8ed1ab_0    conda-forge
executing                 2.2.1              pyhd8ed1ab_0    conda-forge
expat                     2.7.1                h313beb8_0  
filelock                  3.13.1                   pypi_0    pypi
fontconfig                2.15.0               h29935d0_0  
fonttools                 4.60.0                   pypi_0    pypi
freetype                  2.13.3               h47d26ad_0  
fribidi                   1.0.10               h1a28f6b_0  
fsspec                    2024.6.1                 pypi_0    pypi
gettext                   0.21.0               hbdbcc25_2  
graphite2                 1.3.14               hc377ac9_1  
harfbuzz                  10.2.0               he637ebf_1  
icu                       73.1                 h313beb8_0  
importlib-metadata        8.7.0              pyhe01879c_1    conda-forge
ipykernel                 6.30.1             pyh92f572d_0    conda-forge
ipython                   9.5.0              pyhfa0c392_0    conda-forge
ipython_pygments_lexers   1.1.1              pyhd8ed1ab_0    conda-forge
jedi                      0.19.2             pyhd8ed1ab_1    conda-forge
jinja2                    3.1.4                    pypi_0    pypi
jpeg                      9f                   h2f69dba_0  
jupyter_client            8.6.3              pyhd8ed1ab_1    conda-forge
jupyter_core              5.8.1              pyh31011fe_0    conda-forge
kiwisolver                1.4.9                    pypi_0    pypi
krb5                      1.21.3               h237132a_0    conda-forge
lcms2                     2.17                 h7418793_0  
lerc                      4.0.0                h313beb8_0  
libavif                   1.3.0                hb06b76e_2    conda-forge
libavif16                 1.3.0                hb06b76e_2    conda-forge
libcxx                    20.1.8               h8869778_0  
libde265                  1.0.15               h2ffa867_0    conda-forge
libdeflate                1.22                 h80987f9_0  
libedit                   3.1.20250104    pl5321hafb1f1b_0    conda-forge
libexpat                  2.7.1                hec049ff_0    conda-forge
libffi                    3.4.4                hca03da5_1  
libgfortran5              15.2.0               hb654fa1_1  
libglib                   2.84.4               h7a3292d_0  
libheif                   1.19.7          gpl_h79e6334_100    conda-forge
libiconv                  1.16                 h80987f9_3  
liblzma                   5.8.1                h39f12f2_2    conda-forge
liblzma-devel             5.8.1                h39f12f2_2    conda-forge
libopenblas               0.3.30               hf2bb037_2  
libopenjpeg               2.5.4                haa24f5a_1  
libpng                    1.6.50               h5c318fc_0  
libsodium                 1.0.20               h99b78c6_0    conda-forge
libsqlite                 3.51.1               h9a5124b_0    conda-forge
libtiff                   4.7.1                h367c460_0  
libwebp-base              1.6.0                h92b2d59_0  
libxml2                   2.13.9               h528a072_0  
libzlib                   1.3.1                h5f15de7_0  
llvm-openmp               20.1.8               he822017_0  
lz4-c                     1.9.4                h313beb8_1  
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.10.6                   pypi_0    pypi
matplotlib-inline         0.1.7              pyhd8ed1ab_1    conda-forge
mpmath                    1.3.0                    pypi_0    pypi
ncurses                   6.5                  hee39554_0  
nest-asyncio              1.6.0              pyhd8ed1ab_1    conda-forge
networkx                  3.3                      pypi_0    pypi
numpy                     2.3.3                    pypi_0    pypi
numpy-base                2.3.5           py311h23175f9_0  
openssl                   3.5.3                h5503f6c_0    conda-forge
packaging                 25.0               pyh29332c3_1    conda-forge
pandas                    2.3.3           py311hdb8e4fa_2    conda-forge
parso                     0.8.5              pyhcf101f3_0    conda-forge
pcre2                     10.46                h1dacb4a_0  
pexpect                   4.9.0              pyhd8ed1ab_1    conda-forge
pickleshare               0.7.5           pyhd8ed1ab_1004    conda-forge
pillow                    11.3.0                   pypi_0    pypi
pillow-heif               1.1.1           py311habe3797_0    conda-forge
pip                       25.2               pyhc872135_0  
pixman                    0.46.4               h09dc60e_0  
platformdirs              4.4.0              pyhcf101f3_0    conda-forge
prompt-toolkit            3.0.52             pyha770c72_0    conda-forge
psutil                    7.0.0           py311h254cc4a_0  
ptyprocess                0.7.0              pyhd8ed1ab_1    conda-forge
pure_eval                 0.2.3              pyhd8ed1ab_1    conda-forge
pygments                  2.19.2             pyhd8ed1ab_0    conda-forge
pyparsing                 3.2.4                    pypi_0    pypi
python                    3.11.11         hc22306f_2_cpython    conda-forge
python-dateutil           2.9.0.post0        pyhe01879c_2    conda-forge
python-tzdata             2025.2             pyhd3eb1b0_0  
python_abi                3.11                    3_cp311  
pytz                      2025.2          py311hca03da5_0  
pyzmq                     27.1.0          py311h854a7ef_0  
rav1e                     0.7.1                h0716509_3    conda-forge
readline                  8.3                  h0b18652_0  
setuptools                78.1.1          py311hca03da5_0  
six                       1.17.0             pyhe01879c_1    conda-forge
sqlite                    3.50.2               h79febb2_1  
stack_data                0.6.3              pyhd8ed1ab_1    conda-forge
svt-av1                   3.1.2                h12ba402_0    conda-forge
sympy                     1.13.3                   pypi_0    pypi
tk                        8.6.15               hcd8a7d5_0  
torch                     2.8.0                    pypi_0    pypi
torchvision               0.23.0                   pypi_0    pypi
tornado                   6.5.1           py311h80987f9_0  
tqdm                      4.67.1                   pypi_0    pypi
traitlets                 5.14.3             pyhd8ed1ab_1    conda-forge
typing-extensions         4.12.2                   pypi_0    pypi
typing_extensions         4.15.0             pyhcf101f3_0    conda-forge
tzdata                    2025b                h04d1e81_0  
wcwidth                   0.2.13             pyhd8ed1ab_1    conda-forge
wheel                     0.45.1          py311hca03da5_0  
x265                      3.5                  hbc6ce65_3    conda-forge
xz                        5.8.1                h9a6d368_2    conda-forge
xz-gpl-tools              5.8.1                h9a6d368_2    conda-forge
xz-tools                  5.8.1                h39f12f2_2    conda-forge
zeromq                    4.3.5                h888dc83_9    conda-forge
zipp                      3.23.0             pyhd8ed1ab_0    conda-forge
zlib                      1.3.1                h5f15de7_0  
zstd                      1.5.7                h817c040_0  


Supplementary material:
I only include 102 samples per each category for the upload since it is very large data. So you will only see 102 samples in each subcategory of train/test/val per each dataset as the full dataset is 60,000+ files. This amount is sufficient to run the demo script and to run the model, but the model will have less data to work with. 

Link to processed_data and checkpoints (for model epochs) and final presentation:
https://drive.google.com/drive/folders/1fhYYf2zJZb1ePlW8mNlEFHJZpOfVN22S?usp=sharing

/checkpoints
  2 model checkpoints epoch 150 and 151, these are the trained model
/processed_data
  all data already preprocessed for convenience
final_presentation.mp4 - the final presentation video
final_presentation.pdf

File Structure:
PROJECT_ROOT/
  processed_all/(download) or render yourself with final_preprocess_data.py (1.5 hour runtime)
  checkpoints/
    tri_transformer_depth_logchroma_ep151.pth
    ...
  final_model.py
  final_demo.py
  final_preprocess_data.py
  predictions_test (will be generated by final_demo.py)
  raw_data/
    archive/
      Cityscape Dataset/
        leftImg8bit/
          train/<city>/*.png
          val/<city>/*.png
          test/<city>/*.png
      Fine Annotations/
        gtFine/
          train/<city>/*_gtFine_labelIds.png
          val/<city>/*_gtFine_labelIds.png
          test/<city>/*_gtFine_labelIds.png

    data/
      train/
        image/*.npy
        depth/*.npy
      val/
        image/*.npy
        depth/*.npy

    nyu_data/
      data/
        nyu2_train.csv
        nyu2_test.csv
        ...

    pix2pix-depth/
      pix2pix-depth/
        training/*.png
        validation/*.png
        testing/*.png

## Usage
Download the data here:
1. Run final_preprocess_data.py first to generate processed data from the raw data.
2. Run final_model.py to verify the model training loop works. It trained for 2 full days and got to 150 epochs so shorten the epochs if you want to try a run.
3. Run final_demo.py - currently it is hardcoded to run from epoch 151, taken from the saved model checkpoints, you can run whichever checkpoint you want and get summary stats for it. Here are some viable ways to run this script. First ensure you downloaded all the data and you have the preprocessed data as well as the model checkpoints. You can select any of the three datasets cityscapes, pix2pix, or nyu. You can run this as follows:


default:
python final_demo.py

python final_demo.py \
  --checkpoint checkpoints/tri_transformer_depth_logchroma_ep151.pth \
  --data-root processed_all \
  --dataset cityscapes \
  --num-images 50 \
  --save-dir predictions_test

  python final_demo.py \
  --dataset cityscapes \
  --num-images 10 \
  --save-dir predictions_cityscapes

  python final_demo.py \
  --dataset cityscapes \
  --eval-metrics \
  --save-dir predictions_cityscapes_full

  python final_demo.py \
  --dataset cityscapes \
  --num-images 5 \
  --save-dir predictions_cityscapes_speed

  python final_demo.py \
  --use-my-image

## References

Dijk, T.V. and Croon, G.D., 2019. How do neural networks see depth in single images?. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 2183-2191).

Maxwell, B.A., Singhania, S., Patel, A., Kumar, R., Fryling, H., Li, S., Sun, H., He, P. and Li, Z., 2024. Logarithmic lenses: Exploring log rgb data for image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 17470-17479).

Zhang, W., Liu, H., Li, B., He, J., Qi, Z., Wang, Y., Zhao, S., Yu, X., Zeng, W. and Jin, X., 2025. Hybrid-grained Feature Aggregation with Coarse-to-fine Language Guidance for Self-supervised Monocular Depth Estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 6678-6692).

Masoumian, A., Rashwan, H.A., Cristiano, J., Asif, M.S. and Puig, D., 2022. Monocular depth estimation using deep learning: A review. Sensors, 22(14), p.5353.

Liu, L., Song, X., Wang, M., Liu, Y. and Zhang, L., 2021. Self-supervised monocular depth estimation for all day images using domain separation. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 12737-12746).

Cheng, Z., Zhang, Y. and Tang, C., 2021. Swin-depth: Using transformers and multi-scale fusion for monocular-based depth estimation. IEEE Sensors Journal, 21(23), pp.26912-26920.

Link to Original Dataset Website
https://www.cityscapes-dataset.com/dataset-overview/

Youtube:
https://youtu.be/5IImLps1ayw?si=JKAvPswbtrKQaLAJ
https://youtu.be/sz30TDttIBA?si=2wWhX0tQvLv9ROON
https://youtu.be/egBNsSCajDg?si=lsDrsW0hIZIinUoh

Project Data such as model and checkpoints:
Datasets: download and place all in the same directory called /raw_data/ next to the code files
  https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation
  https://www.kaggle.com/datasets/electraawais/cityscape-dataset
  https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2
  https://www.kaggle.com/datasets/greg115/pix2pix-depth

Link to processed_data and checkpoints (for model epochs) and final presentation:
I only include 102 samples per each category for the upload since it is very large data. So you will only see 102 samples in each subcategory of train/test/val per each dataset as the full dataset is 60,000+ files. This amount is sufficient to run the demo script and to run the model, but the model will have less data to work with. 

https://drive.google.com/drive/folders/1fhYYf2zJZb1ePlW8mNlEFHJZpOfVN22S?usp=sharing

/checkpoints
  2 model checkpoints epoch 150 and 151, these are the trained model
/processed_data
  all data already preprocessed for convenience
final_presentation.mp4 - the final presentation video
final_presentation.pdf


## License

This project is for educational purposes as part of CS7180 Advanced Computer Vision course.

## Travel Days

I am using 0 travel days for this project.