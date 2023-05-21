import imagej
ij = imagej.init()
print(ij.getVersion())

import jpype

# Start the JVM

# Open a file using a string file path

ij = imagej.init('C:\Fiji.app')

inputDirectory = "C:/Users/amityu/Gel_Sheet_Data/150721/C1";
outputDirectory = "C:/Users/amityu/Gel_Sheet_Data/150721/tif/";


#get file list in input directory
dir = jpype.java.io.File(inputDirectory)
output_dir = str(jpype.java.io.File(outputDirectory).getAbsolutePath()) +'\\'

#read files from input directory
icsFiles = [str(f) for f in dir.listFiles() if f.isFile() and str(f).endswith(".ics")]

for file in icsFiles:
    #read the file but don't show it
    image = ij.io().open(file)
    ij.io().save(image, output_dir + file.replace(".ics", ".tif").split('\\')[-1])


    print(file)


#%%
