# CASE_QUAK

MassScan folder for performing bump hunt with Combine

QUAKTraining folder for training and testing models


# Updated Instructions July 7

The setup to do the QUAK fitting is a little "suboptimal" right now, but hopefully these instructions can get you going.  The current scheme uses the main fitting code used by the other CASE groups, which is why this is a little convoluted.  The main framework relies on having h5 files with your mjj spectra, so the QUAK fitting scheme consists of two parts:

1. h5 file making.  This is the step that divvies up the QUAK space, and then a separate h5 file is made for each QUAK bin

2. Fitting.  This takes in the h5 files from the previous step and does a unified fit to extract significance at this point

## 0th Steps - Getting Set Up

A complication here is that the h5 file making step uses python 3 and a version of ROOT with RDataFrames, while the fitting relies on combine, which requires python2 and an older version of ROOT that doesn't have RDataFrames

So the setup that I have can be found here: /uscms_data/d1/wmccorma/CASE_ALL/.  It's not the smartest setup, but that directory has two CMSSW releases in it: CMSSW_12_4_0_pre2 and CMSSW_10_2_13.  CMSSW_10_2_13 is the recommended release for combine, and CMSSW_12_4_0_pre2 is just a more up-to-date release that has an environment I wanted for the h5 file making.  You could also choose to use a conda environment or something like that.

If you choose to sourse an up-to-date CMSSW release, like CMSSW_12_4_0_pre2, then you might end up with a directory like /uscms_data/d1/wmccorma/CASE_ALL/CMSSW_12_4_0_pre2/src/.  Within this directory, you can clone https://github.com/wpmccormack/CASE_QUAK.  The important directory for our purposes is "h5Dir"

I personally set up combine with the instructions here: https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#cc7-release-cmssw_10_2_x-recommended-version.  Then within /uscms_data/d1/wmccorma/CASE_ALL/CMSSW_10_2_13/src/ I cloned my fork of the main CASE fitting software: https://github.com/wpmccormack/CASEUtils/tree/master.  You'll need to switch to the QUAK_branch_pullInMain_v2 branch (there is a slightly older QUAK_branch branch, that doesn't have some code updates from Oz pulled in; sometimes the performance in the older branch looks a tiny bit better).  The fitting code is in the "fitting" directory.


## 1st step: H5 File Making

First you need to be in the python3+modern ROOT environment

To do the h5 file making you need to use the doBinning.py code in h5Dir.  You can run this with a command like:

python3 doBinning.py -t sigTrainXYY_X3000_Y80_UL17_bkgTrainQCDBKG -i XYY_X3000_Y80_UL17 -x 100 -n 15 -b 3 -c .3

You can check the various arguments in doBinning, and it will also walk you through the inputs if you don't want to specify them in the command line.  One thing to be aware of is that /uscms/home/sbrightt/nobackup/CASE/analysisOutput/data_forStats_mjjDecorrelate/ is taken as the default sample location.  You can change this with an argument, but you won't be prompted to change it if you don't specify it.

To be clear about the example command above, that would use a signal loss trained on XYY and background trained on QCD.  It would inject 100 fb of XYY signal into 15 background files.  Then the QUAK bins would use the 30% least background-like events and break those events into 3x3 bins.

When you run the code, a directory will be created which contains the h5 files, some templates for performing significance scans, and some combine commands in text files that will be needed in the next step.  In this case, it would be called sigTrainXYY_X3000_Y80_UL17_bkgTrainQCDBKG_INJECT_XYY_X3000_Y80_UL17_XS_100fb_15BkgFiles_30percConsidered_3Bins

## 2nd step: Fitting

Here, you need to be in the combine+python2+old ROOT environment

The fitting is mostly controlled by QUAK_Space_SignalTester.py, which is in CASEUtils/fitting/.  An example of how I run this code in condor is in CASEUtils/fitting/runCASE_example.sh, but please don't run this script directly.  I don't think you have permission to overwrite some of the stuff in my eos space, but still, best not to risk it.  The main thing to look at is the bash for loop:

for m in {1800..2700..100}
do
    python QUAK_Space_SignalTester.py -d ./h5s_mjjFlat_35BkgFiles_newSigFracs/${1} --masspoint ${m} -t ${2} -l $((${m}-$((800+25*(${m}-2200)/100)))) -u $((${m}+$((800+25*(${m}-2200)/100))))
done

Quick note: the ${1} argument here is a directory name that contains H5 files.  E.g. sigTrainQstar2000_W400_UL17-and-Wp3000_B400_UL17-and-XYY_X3000_Y80_UL17_bkgTrainQCDBKG_mjjFlat_mjj800_ptCut300_INJECT_XYY_X3000_Y80_UL17_XS_1fb_35BkgFiles_10percConsidered_9Bins.  The ${2} argument is a directory that contains signal template files.  E.g. sigTemplateMakerWithInterpolation_XToYYprimeTo4Q_MY80_MYprime170

 - The -d flag here is the directory with the H5 files.  You can see my tarball /store/user/wmccorma/CASE_H5Files_mjjFlat_35BkgFiles_9Bins_newSigFracs_Oct06.tgz to see exactly how that's configured.  I created it using the code from Step 1 above.
 - The --masspoint flag is the mass hypothesis for the signal
 - The -t flag is a directory that contains signal templates
 - the -l argument is the minimum of the range for the mjj fit (l stands for lower, here)
 - the -u flag is the maximum of the range for the mjj fit (u stands for upper here)

Another very important thing to note is that the main fitting code relies on *signal templates*.  To derive the signal templates, you need to follow the instrucitons like: https://github.com/case-team/CASEUtils/blob/master/fitting/fit_signalshape_template.sh.  I have templates which you can use at

/uscms_data/d1/wmccorma/CASE_ALL/CMSSW_10_2_13/src/CASEUtils/fitting/sigTemplateMakerWithInterpolation (this is graviton signal)

/uscms_data/d1/wmccorma/CASE_ALL/CMSSW_10_2_13/src/CASEUtils/fitting/sigTemplateMakerWithInterpolation_QstarToQW_mW_400

/uscms_data/d1/wmccorma/CASE_ALL/CMSSW_10_2_13/src/CASEUtils/fitting/sigTemplateMakerWithInterpolation_WpToBpT_Bp400_Top170

/uscms_data/d1/wmccorma/CASE_ALL/CMSSW_10_2_13/src/CASEUtils/fitting/sigTemplateMakerWithInterpolation_XToYYprimeTo4Q_MY80_MYprime170

You can just copy those directories to whereever you want to use them.  Right now, it is hard-coded that these directories will be within your "fitting" directory.

This will run the full f-test fit for each QUAK bin, generate combine cards, and then run a combined significance and p-val fits, and run Asymptotic Limits.  Root files with the significance (or p-value) are generated, as are a bunch of plots.  Everything will be put in the H5 directory that you specify with the -d flag


### Older instructions.  Do not use unless you know what you're doing

The fitting is controlled by combineScan.py.  An example command is:

python combineScan.py -d sigTrainXYY_X3000_Y80_UL17_bkgTrainQCDBKG_INJECT_XYY_X3000_Y80_UL17_XS_100fb_15BkgFiles_30percConsidered_3Bins -t sigTemplateMakerWithInterpolation_WpToBpT_Bp400_Top170 --minMscan 3000 --maxMscan 3000 --massgaps 100

A very important thing to note first here, is that a default h5 directory is taken to be: /uscms_data/d1/wmccorma/CASE_ALL/CMSSW_12_4_0_pre2/src/h5Dir.  You'll probably want to change that within the code to whereever your files are.  It can be changed with an argument, but it'd be a pain to have to specify it every time.

Of course, you can see what the various arguments are by looking in combineScan.py.  The -d argument specifies the name of the directory where your h5 files are, which is sigTrainXYY_X3000_Y80_UL17_bkgTrainQCDBKG_INJECT_XYY_X3000_Y80_UL17_XS_100fb_15BkgFiles_30percConsidered_3Bins (continuing the example from the above h5 file making block).  You can adjust the mass ranges considered in a signficance scan.

 