#!/bin/bash

for mass in {MIN..MAX..50}
do
    for sig in {SIGMAMIN..SIGMAMAX..SIGMAGAP}
    do
	COMMAND
	mv higgsCombineTest.ChannelCompatibilityCheck.mH${mass}.root higgsCombineTest.ChannelCompatibilityCheck.mH${mass}_sigma${sig}.root
    done
done

for sig in {SIGMAMIN..SIGMAMAX..SIGMAGAP}
do
    hadd fullMassScan_sigma${sig}.root higgsCombineTest.ChannelCompatibilityCheck.mH*_sigma${sig}.root
done
