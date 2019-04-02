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

//import   "fmt"
import   "math"
import   "sort"

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

func ComputePerformance(predictions Predictions) ([]float64, []int, []int, []int, []int, int, int) {
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

func ComputePrecisionRecall(tp, fp, fn []int, n_pos, n_neg int, normalize bool) ([]float64, []float64) {
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
  if normalize {
    c := float64(n_pos)/float64(n_pos+n_neg)
    for i := 0; i < len(precision); i++ {
      precision[i] = (precision[i] - c)/(1.0 - c)
    }
  }
  return recall, precision
}

func ComputeRoc(tp, fp []int, n_pos, n_neg int) ([]float64, []float64) {
  tpr := make([]float64, len(tp))
  fpr := make([]float64, len(tp))
  for i := 0; i < len(tpr); i++ {
    tpr[i] = float64(tp[i])/float64(n_pos)
    fpr[i] = float64(fp[i])/float64(n_neg)
  }
  return fpr, tpr
}

func ComputeOptimum(tr, x, y []float64) int {
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
