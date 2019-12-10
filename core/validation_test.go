package core

import (
	"github.com/c-bata/goptuna/tpe"
	"github.com/stretchr/testify/assert"
	"github.com/zhenghaoz/gorse/base"
	"gonum.org/v1/gonum/stat"
	"testing"
)

type CVTestModel struct {
	Params base.Params
}

func (model *CVTestModel) SetParams(params base.Params) {
	model.Params = params
}

func (model *CVTestModel) GetParams() base.Params {
	return model.Params
}

func (model *CVTestModel) Predict(userId, itemId int) float64 {
	panic("Predict() not implemented")
}

func (model *CVTestModel) Fit(trainSet DataSetInterface, setters *base.RuntimeOptions) {}

func CVTestEvaluator(estimator ModelInterface, testSet, excludeSet DataSetInterface) ([]float64, []float64) {
	params := estimator.GetParams()
	a := params.GetFloat64(base.Lr, 0)
	b := params.GetFloat64(base.Reg, 0)
	c := params.GetFloat64(base.Alpha, 0)
	d := params.GetFloat64(base.InitMean, 0)
	return []float64{a + b + c + d}, []float64{a + b + c + d}
}

func TestCrossValidate(t *testing.T) {
	model := new(CVTestModel)
	model.SetParams(base.Params{
		base.Lr:    3,
		base.Reg:   5,
		base.Alpha: 7,
	})
	out := CrossValidate(model, nil, NewKFoldSplitter(5), 0, nil, CVTestEvaluator)
	assert.Equal(t, 15.0, stat.Mean(out[0].TestScore, nil))
}

func TestHyperParametersOptimizationCV(t *testing.T) {
	paramGrid := ParameterGrid{
		base.Lr:    FromInt{2, 6},
		base.Reg:   FromInt{3, 7},
		base.Alpha: FromInt{1, 3},
	}
	model := new(CVTestModel)
	model.SetParams(base.Params{base.InitMean: 10})
	out, err := HyperParametersOptimizationCV(model, nil, paramGrid, tpe.NewSampler(), NewKFoldSplitter(5), 100, 0, nil, CVTestEvaluator)
	if err != nil {
		t.Fatal(err)
	}
	// Check best parameters
	assert.Equal(t, 16.0, out.BestCost)
	assert.Equal(t, 16.0, out.BestScore)
	assert.Equal(t, base.Params{base.Lr: 2, base.Reg: 3, base.Alpha: 1}, out.BestParams)
	assert.Equal(t, out.BestCost, stat.Mean(out.Results[out.BestIndex].TestCosts, nil))
	assert.Equal(t, out.BestScore, stat.Mean(out.Results[out.BestIndex].TestScore, nil))
	assert.Equal(t, out.BestParams, out.Params[out.BestIndex])
	assert.Equal(t, 100, len(out.Params))
	assert.Equal(t, 100, len(out.Results))
}

func TestCrossValidateResult_MeanAndMargin(t *testing.T) {
	out := CrossValidateResult{TestScore: []float64{1, 2, 3, 4, 5}}
	mean, margin := out.MeanAndMargin()
	assert.Equal(t, 3.0, mean)
	assert.Equal(t, 2.0, margin)
}
