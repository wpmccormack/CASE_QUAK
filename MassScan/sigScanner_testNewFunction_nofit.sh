#!/bin/bash

#s=20
#b=15

mkdir $1
cd $1
ln -s ../etacutNone_NSQUAD* .
ln -s ../rooWorkSpaceMaker*.py .
ln -s ../test_BINTEMPLATE.txt .
ln -s ../combine_TEMPLATE.txt .
ln -s ../massScanner_TEMPLATE.sh .
ln -s ../significance_TEMPLATE.sh .
ln -s ../single_mass.sh .
ln -s ../residual_TEMPLATE.py .
ln -s ../simpleDrawer_TEMPLATE.py .
ln -s ../Roo* .
python rooWorkSpaceMaker_NewDiJetFunction.py -f $2
source combineCommand.txt
text2workspace.py fullCard.txt
source significance.sh
python resid.py
#source autoMassScan.sh
#python fullPlotter.py

#combine -M ChannelCompatibilityCheck fullCard.root -m 2500 --preFitValue 0 --rMax 1000 --setParameterRanges sigma=75,75:sig_rate_bkgL_190_10000_sigL_389_10000=1,1:sig_rate_bkgL_162_190_sigL_299_10000=1,1:sig_rate_bkgL_162_190_sigL_0_299=1,1:sig_rate_bkgL_190_10000_sigL_0_389=1,1 --saveFitResult

#for s in {20,25}
#do
#    for b in {10,15}
#    do
#	mkdir -p test_${b}_${s}
#	
#	text2workspace.py test_bkgL_${b}_sigL_${s}.txt
#	
#	for i in {1500..3000..100}
#	do
#	    echo $i
#	    combine -M Significance -m $i test_bkgL_${b}_sigL_${s}.root
#	    mv higgsCombineTest.Significance.mH${i}.root test_${b}_${s}/
#	done
#	
#	hadd test_${b}_${s}/tester.root test_${b}_${s}/*
#	
#	python sigPlotter.py test_${b}_${s}
#    done
#done
#combine -M Significance -
