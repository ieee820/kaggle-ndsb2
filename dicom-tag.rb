require 'dicom'
require 'optparse'
require 'json'
require 'pp'

# A DICOM Tag dumper, DICOM.dcm -> tag.JSON.
# 
# Installation: 
#   gem install dicom
#
# Run:
#   ruby dicom-tag.rb --dir ./data_dir
#
# Options:
#             --dir   : data dir

params = ARGV.getopts(nil, "dir:./data/")
pp params
DICOM.logger.level = Logger::ERROR

Dir.glob(File.join(params["dir"], "**", "*.dcm")).each do |file|
  json_file = file.gsub(/\.dcm$/, ".json")
  dcm = DICOM::DObject.read(file)
  img = dcm.image
  puts "write : #{json_file}"
  File.write(json_file, dcm.to_hash.to_json)
end
