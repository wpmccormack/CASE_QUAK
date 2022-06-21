fi=(LISTOFFILES)

echo "${#fi[@]}"

for m in {MASSMIN..MASSMAX..GAPS}
do
    if [ ${#fi[@]} -eq 4 ]
    then
	val=0
	for f in "${fi[@]}"
	do
	    if [ $val -eq 2 ]
            then
		python dijetfit.py -i ${f}.h5 -M ${m} --sig_shape ./sigTemplateMakerWithInterpolation/graviton_interpolation_M${m}.0.root --dcb-model -t ${f} -p plots_M${m}_${f} -t ${f} -c True
	    else
		python dijetfit.py -i ${f}.h5 -M ${m} --sig_shape ./sigTemplateMakerWithInterpolation/graviton_interpolation_M${m}.0.root --dcb-model -t ${f} -p plots_M${m}_${f} -t ${f}
	    fi
	    val=$((val+1))
	done
    else
	for f in "${fi[@]}"
	do
	    python dijetfit.py -i ${f}.h5 -M ${m} --sig_shape ./sigTemplateMakerWithInterpolation/graviton_interpolation_M${m}.0.root --dcb-model -t ${f} -p plots_M${m}_${f} -t ${f}
	done
    fi
    source combineCommand.sh
    mv fullCard.txt fullCard_M${m}.txt
    text2workspace.py fullCard_M${m}.txt
    combine -M Significance fullCard_M${m}.root -m ${m} -n significance_COMBO
    combine -M Significance fullCard_M${m}.root -m ${m} --pvalue -n pval_COMBO
done
