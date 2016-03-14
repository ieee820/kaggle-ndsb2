# Second Annual Data Science Bowl

Code for Second Annual Data Science Bowl.

# Sumamry

A Hybrid Deep CNN/MLP based approach.
It is used only 3 sax slices to predict real volume, is not used segmentation technique, is not needed hand-labeling.



## Developer Environment

- Ubuntu 14.04
- 16GB RAM 
- GPU & CUDA (I used EC2 g2.2xlarge instance)
- [Torch7](http://torch.ch/)
- Ruby
- dicom (rubygems)
- graphicsmagick (luarocks)

## Installation

Install CUDA, Torch7 first.

```
sudo apt-get install libgraphicsmagick-dev ruby rubygems
sudo gem install dicom
luarocks install graphicsmagick
```

## For validation set

    ./run_all.sh

## For test set

    ./run_all_test.sh

NOTICE: I used 8 g2.xlarge instances to execute this script. See comments in `./run_all_test.sh`
