# Bump Hunt Code

For this code to work, you must set up and source Higgs Combine https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/.  I did the recommended 10_2_X version.  Setup instructions here: https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#cc7-release-cmssw_10_2_x-recommended-version.  Make sure to setup cms environment and cmsenv in the release's src/ folder where you've pulled combine before running the bump hunt code

Important note: You need to copy the files in the scriptsForCombine/ directory here into the correct places in Combine.  Or rather, there are places in the files in scriptsForCombine/ that do gSystem->Load for a .so file.  This ia custom function.  You'll need to change the hard-coded locations of the .so files to the correct places in your environment and recompile the Combine code

The bump hunt works with a command like

source sigScanner_testNewFunction_nofit.sh descriptiveFolderName etacutNone_NSQUAD.root

where descriptiveFolderName is a folder that will be created and where everything will run, and where etacutNone_NSQUAD.root can be replaced with whatever input root file you want to run on (main tree must be named ``output'' with  mass, loss2, and loss1 branches)

Note, depending on the name of your input root file, you may have to add a line to sigScanner_testNewFunction_nofit.sh.  There needs to be a symbolic link to the root file in descriptiveFolderName.  When I was developing things, all the root file names started with etacutNone_NSQUAD, hence that line in sigScanner_testNewFunction_nofit.sh.  Either follow that naming convention, or just add an ln -s line.  Your root files need to be in the MassScan/ directory.