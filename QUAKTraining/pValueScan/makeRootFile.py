import numpy as np
import matplotlib.pyplot as plt

import ROOT as r
from array import array
import significance_fit_new as sf
import sys
import os

inDir = sys.argv[1]
files = [f for f in os.listdir(inDir) if ".npy" in f]

for f in files:

    inFile = inDir+"/"+f

    outFile = "".join(inFile.split("/")[-1].split(".")[:-1])+".root"

    npf = np.load(inFile)

    test_mass = npf[0]
    sigtr_test_loss = npf[1]
    bkgtr_test_loss = npf[2]
    test_label = npf[3]

    lFile = r.TFile("root_files/"+outFile,"RECREATE")
    lTree = r.TTree("output","output")
    mass  =  array('f', [ 1.5 ])
    loss1 =  array('f', [ 1.5 ])
    loss2 =  array('f', [ 1.5 ])
    label =  array('f', [ 1.5 ])
    lTree.Branch('mass',   mass, 'mass/F')
    lTree.Branch('loss1', loss1, 'loss1/F')
    lTree.Branch('loss2', loss2, 'loss2/F')
    lTree.Branch('label', label, 'label/F')

    for i0 in range(len(test_mass)):
        mass [0]=test_mass[i0]
        loss1[0]=sigtr_test_loss[i0]
        loss2[0]=bkgtr_test_loss[i0]
        label[0]=test_label[i0]
        lTree.Fill()

    lTree.Write()
    lFile.Close()