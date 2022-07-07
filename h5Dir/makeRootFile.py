import numpy as np
import matplotlib.pyplot as plt

import ROOT as r
from array import array
import sys
import os
import json

inDir = sys.argv[1]
signal = sys.argv[2]
xsec = float(sys.argv[3])
n_bkg_files = int(sys.argv[4])
outfile = sys.argv[5]

if len(sys.argv) < 6:
    print("Not enough arguments!")
    print("Usage: makeRootFile.py directory_with_npy_files signal_to_inject desired_xsec(fb) n_bkg_files_to_load outFileName")
    exit

sig_file = inDir+"/eval_{0}.npy".format(signal)
sig_arr = np.load(sig_file)
nsig_tot = sig_arr.shape[1]

bkg_arr = None
for i in range(n_bkg_files):
    bkg_file = inDir+"/eval_QCDBKG_{0}.npy".format(i)
    if i == 0:
        bkg_arr = np.load(bkg_file)
    else:
        bkg_arr = np.concatenate((bkg_arr,np.load(bkg_file)),axis=1)

with open("/uscms/home/sbrightt/nobackup/CASE/analysisOutput/data_forStats_mjjDecorrelate/info.json","r") as f1:
    info = json.load(f1)

bkg_tot_lumi = 26.81 #1/fb
n_bkg = bkg_arr.shape[1]
n_bkg_tot = info['n_bkg_tot']
eff_lumi = bkg_tot_lumi * (n_bkg/n_bkg_tot)
print("{} total bkg events, using {}".format(n_bkg_tot,n_bkg))
print("corresponds to an effective lumi of {}".format(eff_lumi))

sig_presel_eff = info[signal]
nsig_keep = int(xsec*eff_lumi*sig_presel_eff)
sig_arr = sig_arr[:,:nsig_keep]
print("injecting {} signal events".format(nsig_keep))

total_arr = np.concatenate((sig_arr,bkg_arr),axis=1)
test_mass = total_arr[0,:]
sigtr_loss = total_arr[1,:]
bkgtr_loss = total_arr[2,:]
test_label = total_arr[3,:]

lFile = r.TFile(outfile,"RECREATE")
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
    loss1[0]=sigtr_loss[i0]
    loss2[0]=bkgtr_loss[i0]
    label[0]=test_label[i0]
    lTree.Fill()

lTree.Write()
lFile.Close()
