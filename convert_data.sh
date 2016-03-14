#!/bin/sh

ruby dicom-tag.rb --dir ./data/train
ruby dicom-tag.rb --dir ./data/validate

th convert_data.lua -dir ./data/train -label ./data/train.csv -prefix train_sax -outputdir ./data -image_size 64 -calibration_mergin 12
th convert_data.lua -dir ./data/train -label ./data/train.csv -prefix train_sax -outputdir ./data -image_size 64 -calibration_mergin 16

th convert_data.lua -dir ./data/validate -prefix validate_sax -outputdir ./data -image_size 64 -calibration_mergin 12
th convert_data.lua -dir ./data/validate -prefix validate_sax -outputdir ./data -image_size 64 -calibration_mergin 16
