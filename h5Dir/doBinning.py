import os
import optparse
import shutil

def fitting_options():
    parser = optparse.OptionParser()
    parser.add_option("-l", "--sampleLoc", dest="sampleLoc", default="/uscms/home/sbrightt/nobackup/CASE/analysisOutput/data_forStats_mjjDecorrelate/", help="sample file location, default is /uscms/home/sbrightt/nobackup/CASE/analysisOutput/data_forStats_mjjDecorrelate/")
    parser.add_option("-t", "--sigTrain", dest="sigTrain", default="DUMMYDIR", help="signal was trained on? Example: sigTrainXYY_X3000_Y80_UL17_bkgTrainQCDBKG")
    parser.add_option("-i", "--sigInject", dest="sigInject", default="DUMMYDIR", help="signal you want to inject? Example: eval_XYY_X3000_Y80_UL17")
    parser.add_option("-x", "--crossSection", dest="crossSection", type=int, default=-1, help="how much signal do you want to inject in fb?")
    parser.add_option("-n", "--nBkgFiles", dest="nBkgFiles", type=int, default=-1, help="how many background files to use?")
    parser.add_option("-b", "--numBins", dest="numBins", type=int, default=-1, help="how many bins to use per loss (total number of bins will be numBins^2)?")
    parser.add_option("-c", "--consideredFrac", dest="consideredFrac", type=float, default=-1., help="what fraction of event will be considered (e.g entering .1 would consider the 10% least background-like events)?")
    return parser

if __name__ == "__main__":
    parser = fitting_options()
    (options, args) = parser.parse_args()

    sampleLoc = options.sampleLoc
    if(not os.path.isdir(sampleLoc)):
        print("Sample directory does not exist!")
        sampleLoc = str(input('Directory with samples (default /uscms/home/sbrightt/nobackup/CASE/analysisOutput/data_forStats_mjjDecorrelate/) : ') or "/uscms/home/sbrightt/nobackup/CASE/analysisOutput/data_forStats_mjjDecorrelate/")
        if(not os.path.isdir(sampleLoc)):
            print("Sample directory still does not exist!")
            exit()
    print("using "+sampleLoc)

    sigTrain = options.sigTrain
    if(sigTrain == "DUMMYDIR"):
        sigTrain = str(input('sigTrain type (e.g. sigTrainXYY_X3000_Y80_UL17_bkgTrainQCDBKG) : ') or "DUMMYDIR")
    stTries = 0
    while(not os.path.isdir(sampleLoc+'/'+sigTrain) and stTries < 5):
        print("sigTraining not available.  Please choose from: ")
        print(os.listdir(sampleLoc))
        sigTrain = str(input('sigTrain type (e.g. sigTrainXYY_X3000_Y80_UL17_bkgTrainQCDBKG) : ') or "DUMMYDIR")
        stTries += 1
    if(not os.path.isdir(sampleLoc+'/'+sigTrain)):
        print("you had five tries to get it right...")
        exit()

    fullSigTrainPath = sampleLoc+'/'+sigTrain+"/"

    sigInject = options.sigInject
    if(sigInject == "DUMMYDIR"):
        sigInject = str(input('what signal do you want to inject? (e.g. XYY_X3000_Y80_UL17) : ') or "DUMMYDIR")
    stTries = 0
    while(not os.path.isfile(fullSigTrainPath+'eval_'+sigInject+".npy") and stTries < 5):
        print(fullSigTrainPath+'eval_'+sigInject+".npy")
        print("Can't inject that signal right now.  Please choose from: ")
        print(os.listdir(fullSigTrainPath))
        print("but please drop the eval_ and the .npy")
        sigInject = str(input('what signal do you want to inject? (e.g. XYY_X3000_Y80_UL17) : ') or "DUMMYDIR")
        stTries += 1
    if(not os.path.isfile(fullSigTrainPath+'eval_'+sigInject+".npy")):
        print("you had five tries to get it right...")
        exit()

    injectionFile = fullSigTrainPath+'eval_'+sigInject+".npy"
    print("injecting from "+injectionFile)

    crossSection = options.crossSection
    if(crossSection < 0):
        crossSection = int(input("how much signal to inject (fb)? "))

    nBkgFiles = options.nBkgFiles
    if(nBkgFiles < 0):
        nBkgFiles = int(input("how many background files do you want to use? "))

    numBins = options.numBins
    if(numBins < 0):
        numBins = int(input("how many bins to use per loss (total number of bins will be numBins^2)? "))

    consideredFrac = options.consideredFrac
    if(consideredFrac < 0):
        consideredFrac = float(input("what fraction of event will be considered (e.g entering .1 would consider the 10% least background-like events)? "))

    newDirName = sigTrain+"_INJECT_"+sigInject+"_XS_"+str(crossSection)+"fb_"+str(nBkgFiles)+"BkgFiles_"+(str(int(100*consideredFrac)))+"percConsidered_"+str(numBins)+"Bins"
    print(newDirName)

    if(os.path.isdir("./"+newDirName)):
        redo = input("Directory already exists.  Do you want to overwrite? [y/n] ")
        if(redo == "y" or redo == "Y"):
            shutil.rmtree("./"+newDirName)
            os.system("mkdir "+newDirName)
            os.chdir("./"+newDirName)
        else:
            exit()
    else:
        os.system("mkdir "+newDirName)
        os.chdir("./"+newDirName)

    print(os.getcwd())

    os.system("ln -s ../quakSpaceBinner.py .")
    os.system("ln -s ../makeRootFile.py .")
    os.system("ln -s ../significance_TEMPLATE.sh .")

    cmd1 = "python3 makeRootFile.py "+fullSigTrainPath+" "+sigInject+" 1000000 1 signalQUAKSpace.root"
    print(cmd1)
    os.system(cmd1)

    cmd2 = "python3 makeRootFile.py "+fullSigTrainPath+" "+sigInject+" "+str(crossSection)+" "+str(nBkgFiles)+" myOutput.root"
    print(cmd2)
    os.system(cmd2)

    cmd3 = "python3 quakSpaceBinner.py -f myOutput.root -c True -frac "+str(consideredFrac)+" -b "+str(numBins)
    print(cmd3)
    os.system(cmd3)
