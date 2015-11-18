# compute openFace represenations for stored images
from pathlib import Path
import reps
import argparse
import datetime
from PIL import Image
import pickle

dataPath = '/nli_images/'
inputDictFname = 'faceReps'
outputDictFname = 'faceReps'



def iterate_folders(ie_folder_root_path):

    """ computes representations of the stored faces"""

    curOutputDictFname = outputDictFname + datetime.datetime.now().isoformat() + '.json'
    #curInputDictFname = inputDictFname + ".json"

    inputDict = {}
    outputDict = inputDict.copy()

    ie_folders = Path(ie_folder_root_path)
    try:
        with (ie_folders / curOutputDictFname).open('w') as f:
            for folder in ie_folders.iterdir():
                if folder.is_file():
                    continue

                print("processing: ", folder.parts[-2])

                for item in folder.iterdir():
                    if item.is_file() and item.suffix == ".jpg":

                        imgKey = "{0}/{1}".format(item.paths[-2], item.paths[-1])

                        if imgKey not in outputDict.keys():

                            try:
                                img = Image.open(item)
                                img.load()
                                rep = reps.getRep(item)
                                outputDict[imgKey] = rep


                            except (Exception) as e:
                                print(e)

                pickle.dump( outputDict, open(outputDictFname, "wb"))




    finally:
        print("Done!")
