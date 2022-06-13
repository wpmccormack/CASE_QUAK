fi=(LISTOFFILES)

echo "${#fi[@]}"

for sigm in {SIGMAMIN..SIGMAMAX..SIGMAGAP}
do
    for m in {MASSMIN..MASSMAX..GAPS}
    do
	source single_mass.sh ${fi} ${m} ${sigm} &
    done
    expectedNumFiles=$((((MASSMAX-MASSMIN)/GAPS+1)*2))
    while true
    do
	NCLIENTS_RUNNING=$(ls -lt ./ | grep sigma${sigm} | wc | awk {'print $1'})
	if [ $expectedNumFiles == $NCLIENTS_RUNNING ]
        then
            echo all jobs are masses ready
            break
	fi
	echo not ready - sleeping $expectedNumFiles $NCLIENTS_RUNNING sigma${sigm}
	sleep 30
    done
done
