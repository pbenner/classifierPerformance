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

package classifierPerformance

/* -------------------------------------------------------------------------- */

import   "fmt"
import   "math"
import   "sort"

import   "bufio"
import   "io"
import   "strconv"
import   "strings"

/* -------------------------------------------------------------------------- */

func ReadPredictions(reader io.Reader) ([]float64, []int, error) {
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

type Performance struct {
  Tr []float64
  Tp []int
  Fp []int
  Tn []int
  Fn []int
  P, N int
}

func (obj Performance) Len() int {
  return len(obj.Tr)
}

/* -------------------------------------------------------------------------- */

func EvalPerformance(values []float64, labels []int) (Performance, error) {
  sort.Sort(Predictions{values, labels})
  n_pos := 0
  n_neg := 0
  n_pos_map := make(map[float64]int)
  n_neg_map := make(map[float64]int)
  for i, _ := range values {
    if labels[i] == 1 {
      n_pos += 1
    } else
    if labels[i] == 0 {
      n_neg += 1
    } else {
      return Performance{}, fmt.Errorf("invalid label: %d", labels[i])
    }
    n_pos_map[values[i]] = n_pos
    n_neg_map[values[i]] = n_neg
  }
  // create a list of unique thresholds
  tr := []float64{}
  for v, _ := range n_pos_map {
    tr = append(tr, v)
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
  return Performance{Tr: tr, Tp: tp, Fp: fp, Tn: tn, Fn: fn, P: n_pos, N: n_neg}, nil
}

func EvalPrecisionRecall(values []float64, labels []int, normalize bool) ([]float64, []float64, error) {
  if perf, err := EvalPerformance(values, labels); err != nil {
    return nil, nil, err
  } else {
    r, p := PrecisionRecall(perf, normalize)
    return r, p, nil
  }
}

func EvalRoc(values []float64, labels []int) ([]float64, []float64, error) {
  if perf, err := EvalPerformance(values, labels); err != nil {
    return nil, nil, err
  } else {
    r, p := Roc(perf)
    return r, p, nil
  }
}

/* -------------------------------------------------------------------------- */

func PrecisionRecall(perf Performance, normalize bool) ([]float64, []float64) {
  precision := make([]float64, perf.Len())
  recall    := make([]float64, perf.Len())
  for i := 0; i < len(precision); i++ {
    if perf.Tp[i] > 0 {
      recall   [i] = float64(perf.Tp[i])/float64(perf.Tp[i] + perf.Fn[i])
      precision[i] = float64(perf.Tp[i])/float64(perf.Tp[i] + perf.Fp[i])
    } else
    if i > 0 {
      precision[i] = precision[i-1]
    }
  }
  if normalize {
    c := float64(perf.P)/float64(perf.P+perf.N)
    for i := 0; i < len(precision); i++ {
      precision[i] = (precision[i] - c)/(1.0 - c)
    }
  }
  return recall, precision
}

func Roc(perf Performance) ([]float64, []float64) {
  tpr := make([]float64, perf.Len())
  fpr := make([]float64, perf.Len())
  for i := 0; i < len(tpr); i++ {
    tpr[i] = float64(perf.Tp[i])/float64(perf.P)
    fpr[i] = float64(perf.Fp[i])/float64(perf.N)
  }
  return fpr, tpr
}

/* -------------------------------------------------------------------------- */

func AUC(x, y []float64) float64 {
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

func Optimum(tr, x, y []float64) int {
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
