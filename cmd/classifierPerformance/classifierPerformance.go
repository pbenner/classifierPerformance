/* Copyright (C) 2019 Philipp Benner
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package main

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "bufio"
import   "io"
import   "log"
import   "os"
import   "strconv"
import   "strings"

import . "github.com/pbenner/classifierPerformance/pkg/classifierPerformance"
import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

type Config struct {
  NormalizePrecision bool
  PrintHeader        bool
  PrintThresholds    bool
  Verbose            int
}

/* -------------------------------------------------------------------------- */

func PrintStderr(config Config, level int, format string, args ...interface{}) {
  if config.Verbose >= level {
    fmt.Fprintf(os.Stderr, format, args...)
  }
}

/* -------------------------------------------------------------------------- */

func export_table2(config Config, writer io.Writer, x, y []float64, name_x, name_y string) {
  if config.PrintHeader {
    fmt.Fprintf(writer, "%s %s\n", name_x, name_y)
  }
  for i := 0; i < len(x); i++ {
    fmt.Fprintf(writer, "%f %f\n", x[i], y[i])
  }
}

func export_table3(config Config, writer io.Writer, x, y, z []float64, name_x, name_y, name_z string) {
  if config.PrintHeader {
    fmt.Fprintf(writer, "%s %s %s\n", name_x, name_y, name_z)
  }
  for i := 0; i < len(x); i++ {
    fmt.Fprintf(writer, "%f %f %f\n", x[i], y[i], z[i])
  }
}

/* -------------------------------------------------------------------------- */

func read_predictions(config Config, reader io.Reader) ([]float64, []int, error) {
  scanner := bufio.NewScanner(reader)

  i_predictions := -1
  i_labels      := -1

  values := []float64{}
  labels := []int{}

  if scanner.Scan() {
    fields := strings.Fields(scanner.Text())
    if len(fields) != 2 {
      return nil, nil, fmt.Errorf("invalid predictions table")
    }
    for i := 0; i < 2; i++ {
      if fields[i] == "predictions" || fields[i] == "prediction" {
        i_predictions = i
      }
    }
    for i := 0; i < 2; i++ {
      if fields[i] == "labels" || fields[i] == "label" {
        i_labels = i
      }
    }
    if i_predictions == -1 {
      return nil, nil, fmt.Errorf("no column called `predictions' found")
    }
    if i_labels == -1 {
      return nil, nil, fmt.Errorf("no column called `labels' found")
    }
  }

  // read header
  for scanner.Scan() {
    fields := strings.Fields(scanner.Text())
    label, err := strconv.ParseInt(fields[i_labels], 10, 64); if err != nil {
      return nil, nil, err
    }
    value, err := strconv.ParseFloat(fields[i_predictions], 64); if err != nil {
      return nil, nil, err
    }
    if label != 0 && label != 1 {
      return nil, nil, fmt.Errorf("invalid label `%d' observed", label)
    }
    values = append(values, value)
    labels = append(labels, int(label))
  }
  return values, labels, nil
}

func import_predictions(config Config, filename string) ([]float64, []int) {
  var reader io.Reader
  if filename == "" {
    reader = os.Stdin
  } else {
    PrintStderr(config, 1, "Reading predictions from `%s'... ", filename)
    f, err := os.Open(filename)
    if err != nil {
      PrintStderr(config, 1, "failed\n")
      log.Fatal(err)
    }
    defer f.Close()
    reader = f
  }
  if values, labels, err := read_predictions(config, reader); err != nil {
    if filename != "" {
      PrintStderr(config, 1, "failed\n")
    }
    log.Fatal(err)
  } else {
    if filename != "" {
      PrintStderr(config, 1, "done\n")
    }
    return values, labels
  }
  return nil, nil
}

/* -------------------------------------------------------------------------- */

func classifier_performance(config Config, filename, target string) {
  values, labels := import_predictions(config, filename)
  if len(values) == 0 {
    log.Fatalf("table `%s' is empty", filename)
  }
  perf, err := EvalPerformance(values, labels); if err != nil {
    log.Fatal(err)
  }

  switch strings.ToLower(target) {
  case "precision-recall":
    recall, precision := PrecisionRecall(perf, config.NormalizePrecision)
    if config.PrintThresholds {
      export_table3(config, os.Stdout, recall, precision, perf.Tr, "recall", "precision", "threshold")
    } else {
      export_table2(config, os.Stdout, recall, precision, "recall", "precision")
    }
  case "precision-recall-auc":
    recall, precision := PrecisionRecall(perf, config.NormalizePrecision)
    fmt.Println(AUC(recall, precision))
  case "roc":
    fpr, tpr := Roc(perf)
    if config.PrintThresholds {
      export_table3(config, os.Stdout, fpr, tpr, perf.Tr, "FPR", "TPR", "threshold")
    } else {
      export_table2(config, os.Stdout, fpr, tpr, "FPR", "TPR")
    }
  case "roc-auc":
    fpr, tpr := Roc(perf)
    fmt.Println(AUC(fpr, tpr))
  case "optimal-precision-recall":
    recall, precision := PrecisionRecall(perf, config.NormalizePrecision)
    i        := Optimum(perf.Tr, recall, precision)
    if config.PrintHeader {
      fmt.Printf("recall=%f precision=%f threshold=%f\n", recall[i], precision[i], perf.Tr[i])
    } else {
      fmt.Printf("%f %f %f\n", recall[i], precision[i], perf.Tr[i])
    }
  case "optimal-roc":
    fpr, tpr := Roc(perf)
    fpr_inv  := make([]float64, len(fpr))
    for i := 0; i < len(fpr); i++ {
      fpr_inv[i] = 1.0 - fpr[i]
    }
    i := Optimum(perf.Tr, fpr_inv, tpr)
    if config.PrintHeader {
      fmt.Printf("fpr=%f tpr=%f threshold=%f\n", fpr[i], tpr[i], perf.Tr[i])
    } else {
      fmt.Printf("%f %f %f\n", fpr[i], tpr[i], perf.Tr[i])
    }
  default:
    log.Fatalf("invalid target: %s", target)
  }
}

/* -------------------------------------------------------------------------- */

func main() {
  log.SetFlags(0)

  config  := Config{}
  options := getopt.New()

  optNormalizePrec := options.   BoolLong("normalize-precision",  0,    "normalize precision to the interval [0,1]")
  optPrintHeader   := options.   BoolLong("print-header",         0,    "print header")
  optPrintThr      := options.   BoolLong("print-thresholds",     0,    "print addition column with thresholds")
  optVerbose       := options.CounterLong("verbose",             'v',   "verbose level [-v or -vv]")
  optHelp          := options.   BoolLong("help",                'h',   "print help")

  options.SetParameters("<TARGET> [<PREDICTIONS.table>]\n\n" +
    "TARGETS:\n" +
    " -> precision-recall\n" +
    " -> precision-recall-auc\n" +
    " -> roc\n" +
    " -> roc-auc\n" +
    " -> optimal-precision-recall\n" +
    " -> optimal-roc\n")
  options.Parse(os.Args)

  // parse options
  //////////////////////////////////////////////////////////////////////////////
  if *optHelp {
    options.PrintUsage(os.Stdout)
    os.Exit(0)
  }
  if *optVerbose != 0 {
    config.Verbose = *optVerbose
  }
  if len(options.Args()) != 1 && len(options.Args()) != 2 {
    options.PrintUsage(os.Stderr)
    os.Exit(1)
  }
  config.PrintHeader        = *optPrintHeader
  config.PrintThresholds    = *optPrintThr
  config.NormalizePrecision = *optNormalizePrec
  config.PrintThresholds    = *optPrintThr

  target   := options.Args()[0]
  filename := ""
  if len(options.Args()) == 2 {
    filename = options.Args()[1]
  }
  classifier_performance(config, filename, target)
}
