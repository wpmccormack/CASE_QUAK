fi=$1
VAR1=""
VAR2="sigma="${3}","${3}":"
VAR3="sigma="${3}","${3}":"
VAR4="sigma="${3}","${3}":"
VAR5=""
VAR6=""
for f in "${fi[@]}"
do
    for v in 0 -1 1 -10 10
    do
        NCLIENTS_RUNNING=$(combine -M MultiDimFit test_${f}.root -m ${2} --rMax 1000 --setParameterRanges sigma=${3},${3}:sig_rate_${f}=1,1 --setParameters p1_${f}=${v} | grep WARNING: | wc | awk {'print $1'})
        if [ ${NCLIENTS_RUNNING} -lt 1 ]
        then
            #combine -M MultiDimFit test_${f}.root -m ${2} --rMax 1000 --setParameterRanges sigma=${3},${3}:sig_rate_${f}=1,1 --setParameters p1_${f}=${v} | grep 'r : '
            #combine -M Significance test_${f}.root -m ${2} --rMax 1000 --setParameterRanges sigma=${3},${3}:sig_rate_${f}=1,1 --setParameters p1_${f}=${v} | grep 'Significance: '
            VAR1+="p1_${f}=${v},"
            VAR2+="sig_rate_${f}=0.0001,0.0001:"
            VAR3+="sig_rate_${f}=0.0001,0.0001:"
            VAR4+="sig_rate_${f}=0,1000:"
            VAR5+="-P sig_rate_${f} -P bkg_rate_${f} -P p1_${f} -P p2_${f} -P p3_${f} "
            VAR6+="bkg_rate_${f},p1_${f},p2_${f},p3_${f},"
            break
        fi
    done
done

combine -M MultiDimFit fullCard.root -m ${2} --rMax 1 --rMin 1 --setParameterRanges ${VAR2%?} --setParameters ${VAR1%?} -v 10 -n NoSig --keepFailures --saveWorkspace --saveNLL &> initialFit${2}_${3}.txt
mv higgsCombineNoSig.MultiDimFit.mH${2}.root higgsCombineNoSig.MultiDimFit.mH${2}_sigma${3}.root

combine -M MultiDimFit fullCard.root -m ${2} --rMax 1 --rMin 1 --setParameterRanges ${VAR4%?} ${VAR5} --setParameters ${VAR1%?} -v 10 -n Sig --keepFailures --saveWorkspace --saveNLL &> SigFitInit${2}_${3}.txt
mv higgsCombineSig.MultiDimFit.mH${2}.root higgsCombineSig.MultiDimFit.mH${2}_sigma${3}.root
