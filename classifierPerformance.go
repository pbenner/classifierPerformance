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
import   "math"
import   "os"
import   "sort"
import   "strconv"
import   "strings"

import   "github.com/pborman/getopt"

/* -------------------------------------------------------------------------- */

type Config struct {
  Header             bool
  NormalizePrecision bool
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
  if config.Header {
    fmt.Fprintf(writer, "%s %s\n", name_x, name_y)
  }
  for i := 0; i < len(x); i++ {
    fmt.Fprintf(writer, "%f %f\n", x[i], y[i])
  }
}

func export_table3(config Config, writer io.Writer, x, y, z []float64, name_x, name_y, name_z string) {
  if config.Header {
    fmt.Fprintf(writer, "%s %s %s\n", name_x, name_y, name_z)
  }
  for i := 0; i < len(x); i++ {
    fmt.Fprintf(writer, "%f %f %f\n", x[i], y[i], z[i])
  }
}

/* -------------------------------------------------------------------------- */

type Predictions struct {
  Values []float64
  Labels []int
}

func (obj Predictions) Len() int {
  return len(obj.Values)
}

func (obj Predictions) Swap(i, j int) {
  obj.Values[i], obj.Values[j] = obj.Values[j], obj.Values[i]
  obj.Labels[i], obj.Labels[j] = obj.Labels[j], obj.Labels[i]
}

func (obj Predictions) Less(i, j int) bool {
  return obj.Values[i] < obj.Values[j]
}

/* -------------------------------------------------------------------------- */

func read_predictions(config Config, reader io.Reader) (Predictions, error) {
  scanner := bufio.NewScanner(reader)

  i_predictions := -1
  i_labels      := -1

  predictions := Predictions{}

  if scanner.Scan() {
    fields := strings.Fields(scanner.Text())
    if len(fields) != 2 {
      return Predictions{}, fmt.Errorf("invalid predictions table")
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
      return Predictions{}, fmt.Errorf("no column called `predictions' found")
    }
    if i_labels == -1 {
      return Predictions{}, fmt.Errorf("no column called `labels' found")
    }
  }

  // read header
  for scanner.Scan() {
    fields := strings.Fields(scanner.Text())
    label, err := strconv.ParseInt(fields[i_labels], 10, 64); if err != nil {
      return Predictions{}, err
    }
    value, err := strconv.ParseFloat(fields[i_predictions], 64); if err != nil {
      return Predictions{}, err
    }
    if label != 0 && label != 1 {
      return Predictions{}, fmt.Errorf("invalid label `%d' observed", label)
    }
    predictions.Values = append(predictions.Values, value)
    predictions.Labels = append(predictions.Labels, int(label))
  }
  sort.Sort(predictions)
  return predictions, nil
}

func import_predictions(config Config, filename string) Predictions {
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
  if r, err := read_predictions(config, reader); err != nil {
    if filename != "" {
      PrintStderr(config, 1, "failed\n")
    }
    log.Fatal(err)
  } else {
    if filename != "" {
      PrintStderr(config, 1, "done\n")
    }
    return r
  }
  return Predictions{}
}

/* -------------------------------------------------------------------------- */

func compute_performance(config Config, predictions Predictions) ([]float64, []int, []int, []int, []int, int, int) {
  n_pos := 0
  n_neg := 0
  n_pos_map := make(map[float64]int)
  n_neg_map := make(map[float64]int)
  for i, _ := range predictions.Values {
    if predictions.Labels[i] == 1 {
      n_pos += 1
    }
    if predictions.Labels[i] == 0 {
      n_neg += 1
    }
    n_pos_map[predictions.Values[i]] = n_pos
    n_neg_map[predictions.Values[i]] = n_neg
  }
  // create a list of unique thresholds
  tr := []float64{}
  for _, key := range predictions.Values {
    tr = append(tr, key)
  }
  sort.Float64s(tr)
  tp := make([]int, len(tr))
  fp := make([]int, len(tr))
  tn := make([]int, len(tr))
  fn := make([]int, len(tr))
  for i, t := range tr {
    tp[i] = n_pos - n_pos_map[t]
    fp[i] = n_neg - n_neg_map[t]
    tn[i] = n_neg_map[t]
    fn[i] = n_pos_map[t]
  }
  return tr, tp, fp, tn, fn, n_pos, n_neg
}

/* -------------------------------------------------------------------------- */

func auc(x, y []float64) float64 {
  n1 := len(x)
  n2 := len(y)
  if n1 != n2 {
    panic("internal error")
  }
  result := 0.0

  for i := 1; i < n1; i++ {
    dx := math.Abs(x[i] - x[i-1])
    dy := (y[i] + y[i-1])/2.0
    result += dx*dy
  }
  return result
}

func compute_precision_recall(config Config, tp, fp, fn []int, n_pos, n_neg int) ([]float64, []float64) {
  precision := make([]float64, len(tp))
  recall    := make([]float64, len(tp))
  for i := 0; i < len(precision); i++ {
    if tp[i] > 0 {
      recall   [i] = float64(tp[i])/float64(tp[i] + fn[i])
      precision[i] = float64(tp[i])/float64(tp[i] + fp[i])
    } else
    if i > 0 {
      precision[i] = precision[i-1]
    }
  }
  if config.NormalizePrecision {
    c := float64(n_pos)/float64(n_pos+n_neg)
    for i := 0; i < len(precision); i++ {
      precision[i] = (precision[i] - c)/(1.0 - c)
    }
  }
  return recall, precision
}

func compute_roc(config Config, tp, fp []int, n_pos, n_neg int) ([]float64, []float64) {
  tpr := make([]float64, len(tp))
  fpr := make([]float64, len(tp))
  for i := 0; i < len(tpr); i++ {
    tpr[i] = float64(tp[i])/float64(n_pos)
    fpr[i] = float64(fp[i])/float64(n_neg)
  }
  return fpr, tpr
}

func compute_optimum(config Config, tr, x, y []float64) int {
  k := 0
  v := math.Inf(-1)
  for i := 0; i < len(tr); i++ {
    if r := x[i]*y[i]; r > v {
      v = r
      k = i
    }
  }
  return k
}

/* -------------------------------------------------------------------------- */

func classifier_performance(config Config, filename, target string) {
  predictions := import_predictions(config, filename)
  if predictions.Len() == 0 {
    log.Fatalf("table `%s' is empty", filename)
  }
  tr, tp, fp, _, fn, n_pos, n_neg := compute_performance(config, predictions)

  switch strings.ToLower(target) {
  case "precision-recall":
    recall, precision := compute_precision_recall(config, tp, fp, fn, n_pos, n_neg)
    if config.PrintThresholds {
      export_table3(config, os.Stdout, recall, precision, tr, "recall", "precision", "threshold")
    } else {
      export_table2(config, os.Stdout, recall, precision, "recall", "precision")
    }
  case "precision-recall-auc":
    recall, precision := compute_precision_recall(config, tp, fp, fn, n_pos, n_neg)
    fmt.Println(auc(recall, precision))
  case "roc":
    fpr, tpr := compute_roc(config, tp, fp, n_pos, n_neg)
    if config.PrintThresholds {
      export_table3(config, os.Stdout, fpr, tpr, tr, "FPR", "TPR", "threshold")
    } else {
      export_table2(config, os.Stdout, fpr, tpr, "FPR", "TPR")
    }
  case "roc-auc":
    fpr, tpr := compute_roc(config, tp, fp, n_pos, n_neg)
    fmt.Println(auc(fpr, tpr))
  case "optimal-precision-recall":
    recall, precision := compute_precision_recall(config, tp, fp, fn, n_pos, n_neg)
    i        := compute_optimum(config, tr, recall, precision)
    if config.Header {
      fmt.Printf("recall=%f precision=%f threshold=%f\n", recall[i], precision[i], tr[i])
    } else {
      fmt.Printf("%f %f %f\n", recall[i], precision[i], tr[i])
    }
  case "optimal-roc":
    fpr, tpr := compute_roc(config, tp, fp, n_pos, n_neg)
    fpr_inv  := make([]float64, len(fpr))
    for i := 0; i < len(fpr); i++ {
      fpr_inv[i] = 1.0 - fpr[i]
    }
    i := compute_optimum(config, tr, fpr_inv, tpr)
    if config.Header {
      fmt.Printf("fpr=%f tpr=%f threshold=%f\n", fpr[i], tpr[i], tr[i])
    } else {
      fmt.Printf("%f %f %f\n", fpr[i], tpr[i], tr[i])
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

  optHeader        := options.   BoolLong("header",               0,    "print header")
  optNormalizePrec := options.   BoolLong("normalize-precision",  0,    "normalize precision to the interval [0,1]")
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
  config.Header             = *optHeader
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
