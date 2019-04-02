## Compute performance measures of classification results

Prediction results must be given as a table in the following format:
```sh
$ head README.table 
prediction label
0.612547843484208 1
0.364270970690995 1
0.432136141695082 0
0.140291077783331 0
0.384895941475406 0
0.244415489258245 1
0.970641299150884 1
0.890172811923549 1
0.78178137098439 1
```

Compute the precision-recall curve:
```sh
$ classifierPerformance --header precision-recall README.table | head
recall precision
0.989247 0.462312
0.989247 0.464646
0.978495 0.461929
0.967742 0.459184
0.967742 0.461538
0.967742 0.463918
0.967742 0.466321
0.967742 0.468750
0.967742 0.471204
```

Print thresholds columns:
```sh
$ classifierPerformance --header --print-thresholds precision-recall README.table | head
recall precision threshold
0.989247 0.462312 0.005423
0.989247 0.464646 0.007746
0.978495 0.461929 0.012654
0.967742 0.459184 0.014824
0.967742 0.461538 0.016694
0.967742 0.463918 0.018528
0.967742 0.466321 0.030235
0.967742 0.468750 0.035815
0.967742 0.471204 0.040907
```

Plot precision recall curve and save it as *Rplots.pdf*:
```sh
classifierPerformance --header precision-recall README.table | Rscript -e 't <- read.table(file("stdin"), header=T); plot(precision ~ recall, t, type="l")'
```

Compute ROC curve:
```sh
$ classifierPerformance --header --print-thresholds roc README.table | head
FPR TPR threshold
1.000000 0.989247 0.005423
0.990654 0.989247 0.007746
0.990654 0.978495 0.012654
0.990654 0.967742 0.014824
0.981308 0.967742 0.016694
0.971963 0.967742 0.018528
0.962617 0.967742 0.030235
0.953271 0.967742 0.035815
0.943925 0.967742 0.040907
```

Identify an optimal threshold by maximizing precision and recall:
```sh
$ classifierPerformance --header optimal-precision-recall README.table
recall=0.849462 precision=0.831579 threshold=0.499788
```
