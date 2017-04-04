find ~/ori -depth -name "*\.tif" | rename 's/ *\.tif/\.tif/'

grep "OIR\ Inj\ P12/P17/" tmp.txt | sort  > P17.txt
ls ~/oir/CNTF/OIR\ Inj\ P12/P17/*quantified* | sort > p17true.txt
