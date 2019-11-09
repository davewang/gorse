package core

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"github.com/c-bata/goptuna"
	"github.com/c-bata/goptuna/tpe"
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
	TestScore []float64
	TestCosts []float64
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
	CVResults  []CrossValidateResult
	AllParams  []base.Params
}

// GridSearchCV finds the best parameters for a model.
func GridSearchCV(estimator ModelInterface, dataSet DataSetInterface, paramGrid ParameterGrid,
	splitter Splitter, seed int64, options *base.RuntimeOptions, evaluators ...CrossValidationEvaluator) ([]ModelSelectionResult, error) {
	// Retrieve parameter names and length
	paramNames := make([]base.ParamName, 0, len(paramGrid))
	count := 1
	for paramName, values := range paramGrid {
		switch values.(type) {
		case FromCategorical:
			paramNames = append(paramNames, paramName)
			count *= len(values.(FromCategorical))
		default:
			return nil, errors.New(fmt.Sprintf("GridSearchCV: expect %s to be FromCategorical", paramName))
		}
	}
	// Construct DFS procedure
	var results []ModelSelectionResult
	var dfs func(deep int, params base.Params)
	progress := 0
	dfs = func(deep int, params base.Params) {
		if deep == len(paramNames) {
			progress++
			options.Logf("grid search (%v/%v): %v", progress, count, params)
			// Cross validate
			estimator.SetParams(estimator.GetParams().Merge(params))
			cvResults := CrossValidate(estimator, dataSet, splitter, seed, options, evaluators...)
			// Create GridSearch result
			if results == nil {
				results = make([]ModelSelectionResult, len(cvResults))
				for i := range results {
					results[i] = ModelSelectionResult{}
					results[i].BestCost = math.Inf(1)
					results[i].CVResults = make([]CrossValidateResult, 0, count)
					results[i].AllParams = make([]base.Params, 0, count)
				}
			}
			for i := range cvResults {
				results[i].CVResults = append(results[i].CVResults, cvResults[i])
				results[i].AllParams = append(results[i].AllParams, params.Copy())
				cost := stat.Mean(cvResults[i].TestCosts, nil)
				score := stat.Mean(cvResults[i].TestScore, nil)
				if cost < results[i].BestCost {
					results[i].BestScore = score
					results[i].BestCost = cost
					results[i].BestParams = params.Copy()
					results[i].BestIndex = len(results[i].AllParams) - 1
				}
			}
		} else {
			paramName := paramNames[deep]
			values := paramGrid[paramName]
			for _, val := range values.(FromCategorical) {
				params[paramName] = val
				dfs(deep+1, params)
			}
		}
	}
	params := make(map[base.ParamName]interface{})
	dfs(0, params)
	return results, nil
}

// RandomSearchCV searches hyper-parameters by random.
func RandomSearchCV(estimator ModelInterface, dataSet DataSetInterface, paramGrid ParameterGrid,
	splitter Splitter, trial int, seed int64, options *base.RuntimeOptions, evaluators ...CrossValidationEvaluator) ([]ModelSelectionResult, error) {
	rng := base.NewRandomGenerator(seed)
	var results []ModelSelectionResult
	for i := 0; i < trial; i++ {
		// Make parameters
		params := base.Params{}
		for paramName, values := range paramGrid {
			switch values.(type) {
			case FromCategorical:
				index := rng.Intn(len(values.(FromCategorical)))
				value := values.(FromCategorical)[index]
				params[paramName] = value
			default:
				return nil, errors.New(fmt.Sprintf("GridSearchCV: expect %s to be FromCategorical", paramName))
			}
		}
		// Cross validate
		options.Logf("random search (%v/%v): %v", i+1, trial, params)
		estimator.SetParams(estimator.GetParams().Merge(params))
		cvResults := CrossValidate(estimator, dataSet, splitter, seed, options, evaluators...)
		if results == nil {
			results = make([]ModelSelectionResult, len(cvResults))
			for i := range results {
				results[i] = ModelSelectionResult{}
				results[i].BestCost = math.Inf(1)
				results[i].CVResults = make([]CrossValidateResult, trial)
				results[i].AllParams = make([]base.Params, trial)
			}
		}
		for j := range cvResults {
			results[j].CVResults[i] = cvResults[j]
			results[j].AllParams[i] = params.Copy()
			score := stat.Mean(cvResults[j].TestScore, nil)
			cost := stat.Mean(cvResults[j].TestCosts, nil)
			if cost < results[j].BestCost {
				results[j].BestCost = cost
				results[j].BestScore = score
				results[j].BestParams = params.Copy()
				results[j].BestIndex = len(results[j].AllParams) - 1
			}
		}
	}
	return results, nil
}

// BayesianOptimizationCV searches hyper-parameters by bayesian optimization.
func BayesianOptimizationCV(estimator ModelInterface, dataSet DataSetInterface, paramGrid ParameterGrid,
	splitter Splitter, trial int, seed int64, options *base.RuntimeOptions, evaluators ...CrossValidationEvaluator) ([]ModelSelectionResult, error) {
	// check evaluators
	if len(evaluators) == 0 {
		return nil, nil
	} else if len(evaluators) > 1 {
		return nil, errors.New("BayesianOptimizationCV supports only one evaluator")
	}
	// create objective
	var result ModelSelectionResult
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
		cvResult := CrossValidate(estimator, dataSet, splitter, seed, options, evaluators...)[0]
		result.AllParams = append(result.AllParams, params)
		result.CVResults = append(result.CVResults, cvResult)
		return stat.Mean(cvResult.TestCosts, nil), nil
	}
	// create study
	study, err := goptuna.CreateStudy(
		"BayesianOptimizationCV",
		goptuna.StudyOptionSampler(tpe.NewSampler()),
	)
	if err != nil {
		return nil, err
	}
	// optimize study
	if err = study.Optimize(objective, trial); err != nil {
		return nil, err
	}
	// collect result
	v, _ := study.GetBestValue()
	p, _ := study.GetBestParams()
	result.BestCost = v
	result.BestScore = math.NaN()
	result.BestIndex = -1
	result.BestParams = base.Params{}
	for paramName, values := range paramGrid {
		switch values.(type) {
		case FromInt:
			result.BestParams[paramName] = p[string(paramName)].(int)
		case FromUniform:
			result.BestParams[paramName] = p[string(paramName)].(float64)
		case FromLogUniform:
			result.BestParams[paramName] = p[string(paramName)].(float64)
		case FromCategorical:
			result.BestParams[paramName] = p[string(paramName)].(string)
		}
	}
	return []ModelSelectionResult{result}, nil
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
