package core

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"github.com/c-bata/goptuna"
	"github.com/zhenghaoz/gorse/base"
	"gonum.org/v1/gonum/stat"
	"math"
	"reflect"
)

// ParameterGrid contains candidate for grid search.
type ParameterGrid map[base.ParamName]interface{}

// FromCategorical chooses the hyper-parameter from categorical.
type FromCategorical []interface{}

// ToCategoricalString converts FromCategorical to an array of string.
func (fromCategorical FromCategorical) ToCategoricalString() ([]string, error) {
	values := make([]string, len(fromCategorical))
	for i, value := range fromCategorical {
		switch value.(type) {
		case string:
			values[i] = value.(string)
		default:
			return nil, errors.New("ToCategoricalString: expect string")
		}
	}
	return values, nil
}

// FromUniform the hyper-parameter from uniform distribution.
type FromUniform struct {
	low, high float64
}

// FromLogUniform chooses the hyper-parameter from log uniform distribution.
type FromLogUniform struct {
	low, high float64
}

// FromInt chooses a integer hyper-parameter from uniform distribution.
type FromInt struct {
	low, high int
}

/* Cross Validation */

// CrossValidateResult contains the result of cross validate
type CrossValidateResult struct {
	TestScore []float64 // TestScore is given by evaluators (such as RMSE, Precision, etc.)
	TestCosts []float64 // TestCosts is used for hyper-parameter optimization (lower is better).
}

// MeanAndMargin returns the mean and the margin of cross validation scores.
func (sv CrossValidateResult) MeanAndMargin() (float64, float64) {
	mean := stat.Mean(sv.TestScore, nil)
	margin := 0.0
	for _, score := range sv.TestScore {
		temp := math.Abs(score - mean)
		if temp > margin {
			margin = temp
		}
	}
	return mean, margin
}

// CrossValidate evaluates a model by k-fold cross validation.
func CrossValidate(model ModelInterface, dataSet DataSetInterface, splitter Splitter, seed int64,
	options *base.RuntimeOptions, evaluators ...CrossValidationEvaluator) []CrossValidateResult {
	// Split data set
	trainFolds, testFolds := splitter(dataSet, seed)
	length := len(trainFolds)
	// Cross validation
	scores := make([][]float64, length)
	costs := make([][]float64, length)
	params := model.GetParams()
	base.Parallel(length, options.GetCVJobs(), func(begin, end int) {
		cp := reflect.New(reflect.TypeOf(model).Elem()).Interface().(ModelInterface)
		Copy(cp, model)
		cp.SetParams(params)
		for i := begin; i < end; i++ {
			trainFold := trainFolds[i]
			testFold := testFolds[i]
			cp.Fit(trainFold, options)
			// Evaluate on test set
			for _, evaluator := range evaluators {
				tempScore, tempCost := evaluator(cp, testFold, trainFold)
				scores[i] = append(scores[i], tempScore...)
				costs[i] = append(costs[i], tempCost...)
			}
		}
	})
	// Create return structures
	ret := make([]CrossValidateResult, len(scores[0]))
	for i := 0; i < len(ret); i++ {
		ret[i].TestScore = make([]float64, length)
		ret[i].TestCosts = make([]float64, length)
		for j := range ret[i].TestScore {
			ret[i].TestScore[j] = scores[j][i]
			ret[i].TestCosts[j] = costs[j][i]
		}
	}
	return ret
}

/* Model Selection */

// ModelSelectionResult contains the return of grid search.
type ModelSelectionResult struct {
	BestScore  float64
	BestCost   float64
	BestParams base.Params
	BestIndex  int
	Results    []CrossValidateResult
	Params     []base.Params
}

// HyperParametersOptimizationCV searches hyper-parameters.
func HyperParametersOptimizationCV(estimator ModelInterface, dataSet DataSetInterface, paramGrid ParameterGrid,
	sampler goptuna.Sampler, splitter Splitter, trial int, seed int64, options *base.RuntimeOptions, evaluator CrossValidationEvaluator) (ModelSelectionResult, error) {
	// create objective
	result := ModelSelectionResult{
		BestCost: math.Inf(1),
		Results:  make([]CrossValidateResult, 0, trial),
		Params:   make([]base.Params, 0, trial),
	}
	objective := func(trial goptuna.Trial) (float64, error) {
		// create parameters
		params := base.Params{}
		for paramName, values := range paramGrid {
			switch values.(type) {
			case FromInt:
				params[paramName], _ = trial.SuggestInt(string(paramName), values.(FromInt).low, values.(FromInt).high)
			case FromUniform:
				params[paramName], _ = trial.SuggestUniform(string(paramName), values.(FromUniform).low, values.(FromUniform).high)
			case FromLogUniform:
				params[paramName], _ = trial.SuggestLogUniform(string(paramName), values.(FromLogUniform).low, values.(FromLogUniform).high)
			case FromCategorical:
				choices, err := values.(FromCategorical).ToCategoricalString()
				if err != nil {
					return 0, errors.New(fmt.Sprintf("expect %s to be string", paramName))
				}
				params[paramName], _ = trial.SuggestCategorical(string(paramName), choices)
			}
		}
		estimator.SetParams(estimator.GetParams().Merge(params))
		cvResult := CrossValidate(estimator, dataSet, splitter, seed, options, evaluator)[0]
		result.Params = append(result.Params, params)
		result.Results = append(result.Results, cvResult)
		cost := stat.Mean(cvResult.TestCosts, nil)
		if cost < result.BestCost {
			result.BestCost = cost
			result.BestScore = stat.Mean(cvResult.TestScore, nil)
			result.BestIndex = len(result.Results) - 1
			result.BestParams = params
		}
		return cost, nil
	}
	// create study
	study, err := goptuna.CreateStudy(
		"HyperParametersOptimizationCV",
		goptuna.StudyOptionSampler(sampler),
	)
	if err != nil {
		return result, err
	}
	// optimize study
	if err = study.Optimize(objective, trial); err != nil {
		return result, err
	}
	return result, nil
}

// Copy a object from src to dst.
func Copy(dst, src interface{}) error {
	buffer := new(bytes.Buffer)
	encoder := gob.NewEncoder(buffer)
	if err := encoder.Encode(src); err != nil {
		return err
	}
	decoder := gob.NewDecoder(buffer)
	err := decoder.Decode(dst)
	return err
}
