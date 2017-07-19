find . -depth -name "*\.tif" | wc -l
find . -depth -regextype posix-extended -regex ".*(q|Q)(uantif)?.*\.tif" | wc -l
find . -depth -name "*[Yy]ellow*\.tif" | wc -l
find . -depth -name "*[Rr]ed*\.tif" | wc -l
find . -depth -name "*[Yy]ellow*[Rr]ed*\.tif" | wc -l

find . -depth -name "*((\b[qQ]uantif\b)|(\b[Qq]\b))*\.tif" | wc -l

find /data/oir/ -depth -name "*\.tif" | wc -l
find /data/oir/ -depth -regextype posix-extended -regex ".*/[^/]*((y|Y)ellow|quant|quantified|_Q|QUANTIFIED|QUANT)[^/]*\.tif" | wc -l
find . -depth -regextype posix-extended -regex ".*/[^/]*((((y|Y)ellow)((r|R)ed)?)|quant|quantified|_Q|QUANTIFIED|QUANT)[^/]*\.tif" | wc -l
find . -depth -name "*\.tif" | wc -l
find . -depth -regextype posix-extended -regex ".*/[^/]*(quant|quantified|_Q|QUANTIFIED|QUANT)[^/]*\.tif" | wc -l
find . -depth -name "*[Yy]ellow*\.tif" | wc -l
find . -depth -name "*[Rr]ed*\.tif" | wc -l
find . -depth -name "*[Yy]ellow*[Rr]ed*\.tif" | wc -l
find . -depth -name "*[Yy]ellow*\.tif" | wc -l
find . -depth -name "*[Rr]ed*\.tif" | wc -l

find . -depth -name "*[qQ]uantif*\.tif" | wc -l

