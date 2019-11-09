package main

import (
	"fmt"
	"github.com/zhenghaoz/gorse/base"
	"github.com/zhenghaoz/gorse/core"
	"github.com/zhenghaoz/gorse/model"
	"log"
	"runtime"
)

func main() {
	data := core.LoadDataFromCSV("steam-100k.csv", ",", true)
	//data := core.LoadDataFromBuiltIn("ml-100k")
	cv, err := core.GridSearchCV(model.NewBPR(nil), data, core.ParameterGrid{
		base.NFactors:   core.FromCategorical{5, 10, 50},
		base.Reg:        core.FromCategorical{0.005, 0.01, 0.5},
		base.Lr:         core.FromCategorical{0.01, 0.05, 0.1},
		base.NEpochs:    core.FromCategorical{50},
		base.InitMean:   core.FromCategorical{0},
		base.InitStdDev: core.FromCategorical{0.001},
	}, core.NewKFoldSplitter(5), 0, &base.RuntimeOptions{true, 1, runtime.NumCPU()},
		core.NewRankEvaluator(10, core.Precision, core.Recall, core.NDCG))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("=== Precision@10")
	fmt.Printf("The best score is: %.5f\n", cv[0].BestScore)
	fmt.Printf("The best params is: %v\n", cv[0].BestParams)
	fmt.Println("=== Recall@10")
	fmt.Printf("The best score is: %.5f\n", cv[1].BestScore)
	fmt.Printf("The best params is: %v\n", cv[1].BestParams)
	fmt.Println("=== NDCG@10")
	fmt.Printf("The best score is: %.5f\n", cv[2].BestScore)
	fmt.Printf("The best params is: %v\n", cv[2].BestParams)
}
