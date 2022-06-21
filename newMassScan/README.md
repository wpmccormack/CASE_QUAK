quakSpaceBinner makes h5 files to work with the main CASE plotting code, and fills out the significance_TEMPLATE

Oddity here: you need to run quakSpaceBinner in a place with a version of ROOT that has RDataFrames, ie v6.14 or higher I think.  This is a newer version than what you get if you set up the CMSENV-based version of combine

You can copy the dijetfitter and DataCardMaker into the CASEUtils fitter for now: https://github.com/case-team/CASEUtils/tree/master/fitting

run quakSpaceBinner with e.g. python3 quakSpaceBinner.py -f etacutNone_NSQUAD.root -c True

The -c option makes it so that you only consider the bottom right 4 bins in the QUAK space

Two versions of the datacards get made:
- One with a floating signal normalization for all but the corner bin.  This can be used for a generic signal scan.  Bins with 0 signal don't drag down the significance
- One with a gaussian nuisance parameter for the expected signal contribution in non-corner bins.  This is for limit-setting.  I need to write up some code to actually get the ratio of expected signals in the bins, because right now the nuisance parameter is centered on equivalent signal contributions for all bins, which is not correct