#!/bin/sh

ruby dicom-tag.rb --dir ./data/train
ruby dicom-tag.rb --dir ./data/validate
ruby dicom-tag.rb --dir ./data/test

th convert_data.lua -dir ./data/train -label ./data/train.csv -prefix train_sax -outputdir ./data -image_size 64 -calibration_mergin 12
th convert_data.lua -dir ./data/train -label ./data/train.csv -prefix train_sax -outputdir ./data -image_size 64 -calibration_mergin 16
th convert_data.lua -dir ./data/validate -label ./data/validate.csv -prefix validate_sax -outputdir ./data -image_size 64 -calibration_mergin 12
th convert_data.lua -dir ./data/validate -label ./data/validate.csv -prefix validate_sax -outputdir ./data -image_size 64 -calibration_mergin 16

# merge ./train and ./validate
th merge_data.lua -a data/train_sax_id.t7 -b data/validate_sax_id.t7 -o data/train_sax_id.t7
th merge_data.lua -a data/train_sax_x_64_12.t7 -b data/validate_sax_x_64_12.t7 -o data/train_sax_x_64_12.t7
th merge_data.lua -a data/train_sax_x_64_16.t7 -b data/validate_sax_x_64_16.t7 -o data/train_sax_x_64_16.t7
th merge_data.lua -a data/train_sax_tag_64_12.t7 -b data/validate_sax_tag_64_12.t7 -o data/train_sax_tag_64_12.t7
th merge_data.lua -a data/train_sax_tag_64_16.t7 -b data/validate_sax_tag_64_16.t7 -o data/train_sax_tag_64_16.t7
th merge_data.lua -a data/train_sax_y_64_12.t7 -b data/validate_sax_y_64_12.t7 -o data/train_sax_y_64_12.t7
th merge_data.lua -a data/train_sax_y_64_16.t7 -b data/validate_sax_y_64_16.t7 -o data/train_sax_y_64_16.t7

# test set (but prefix is `validate_sax`)
th convert_data.lua -dir ./data/test -prefix validate_sax -outputdir ./data -image_size 64 -calibration_mergin 12
th convert_data.lua -dir ./data/test -prefix validate_sax -outputdir ./data -image_size 64 -calibration_mergin 16
