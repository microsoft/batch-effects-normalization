import os
import argparse
from PIL import Image
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("archive_path", help="Path of the COOS h5py archive to unpack", type=str)
parser.add_argument("out_path", help="Directory to save images to", type=str)
args = parser.parse_args()
archive_name = args.archive_path
out_folder = args.out_path

# create output directory if necessary
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# open COOS archive
print ("Unpacking", archive_name)
archive = h5py.File(archive_name, "r")
images = archive['data']
labels = archive['labels']

# save all images to output directory by class
for i in range (0, images.shape[0]):
    if i % 1000 == 0:
        print ("Unpacked", i, "of", images.shape[0], "images")
    current_label = labels[i]
    current_protein = Image.fromarray(images[i, 0])
    current_nucleus = Image.fromarray(images[i, 1])

    current_directory = os.sep.join((out_folder, str(current_label)))
    if not os.path.exists(current_directory):
        os.makedirs(current_directory)

    current_protein.save(os.sep.join((current_directory, "COOSv1cell_" + str(i) + "_protein.tif")))
    current_nucleus.save(os.sep.join((current_directory, "COOSv1cell_" + str(i) + "_nucleus.tif")))