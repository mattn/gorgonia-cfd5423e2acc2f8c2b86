package main

import (
	crand "crypto/rand"
	"fmt"
	"log"
	"math"
	"math/big"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"time"

	nnmodel "github.com/mattn/gorgonia-cfd5423e2acc2f8c2b86/twolayernn"
	"github.com/pkg/errors"
	"github.com/seehuhn/mt19937"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const (
	// 次元数（入力）
	dimensionNum = 2
	// クラス数（出力）
	classNum = 3
	// サンプル数
	sampleNum = 100
	// 隠れ層のサイズ
	hiddenSize = 10
	// エポック数
	maxEpoch = 300000
	// バッチサイズ
	batchSize = 30
)

func main() {
	seed, err := crand.Int(crand.Reader, big.NewInt(math.MaxInt64))
	if err != nil {
		log.Fatal(err)
	}
	//乱数の生成にはメルセンヌ・ツイスタを使用
	rng := rand.New(mt19937.New())
	rand.Seed(seed.Int64())

	g := gorgonia.NewGraph()

	//2層ニューラルネットワークを構築
	model := nnmodel.New(g, dimensionNum, hiddenSize, classNum, sampleNum*classNum)

	//学習前の正答率を確認
	err = checkAccuracy(g, model, dimensionNum, classNum, sampleNum, rng)
	if err != nil {
		log.Fatal(err)
	}

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	//学習
	err = train(g, model, dimensionNum, classNum, sampleNum, maxEpoch, batchSize, rng)
	if err != nil {
		log.Fatal(err)
	}

	//学習後の正答率を確認
	err = checkAccuracy(g, model, dimensionNum, classNum, sampleNum, rng)
	if err != nil {
		log.Fatal(err)
	}
}

//入力されたニューラルネットワーク用いtestCount回推論、正答率を計算する
func checkAccuracy(g *gorgonia.ExprGraph, model *nnmodel.TwoLayerNeuralNetworkModel, dimensionNum, classNum, testCount int, rng *rand.Rand) error {
	vm := gorgonia.NewTapeMachine(g)
	//テストデータを生成
	inputDense, expectedDense := createTestData(dimensionNum, classNum, testCount, rng)

	//入力ノードにテストデータを注入
	if err := gorgonia.Let(model.Input, inputDense); err != nil {
		return errors.WithStack(err)
	}

	//実行
	if err := vm.RunAll(); err != nil {
		return errors.WithStack(err)
	}

	var t tensor.Tensor
	var ok bool
	if t, ok = model.Output.Value().(tensor.Tensor); !ok {
		return errors.New("expects a tensor")
	}
	//推論結果を比較しやすい形に変形
	actual, err := tensor.Argmax(t, 1)
	if err != nil {
		panic(err)
	}
	actualData := actual.Data().([]int)

	//期待される結果を比較しやすい形に変形
	expected, err := tensor.Argmax(expectedDense, 1)
	if err != nil {
		return errors.WithStack(err)
	}
	expectedData := expected.Data().([]int)
	var correctAnswerNum int
	for i := 0; i < testCount; i++ {
		//推論結果と期待される結果を比較
		if actualData[i] == expectedData[i] {
			correctAnswerNum++
		}
	}
	fmt.Printf("%d問中 %d問正答 正答率 %.5f\n", testCount, correctAnswerNum, float64(correctAnswerNum)/float64(testCount))
	return nil
}

//学習
func train(g *gorgonia.ExprGraph, model *nnmodel.TwoLayerNeuralNetworkModel, dimensionNum int, classNum int, sampleNum int, maxEpoch int, batchSize int, rng *rand.Rand) error {
	//最適化には基本的な最適化手法を利用
	solver := gorgonia.NewVanillaSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(1))
	//勾配を計算するためのノードを登録
	if err := model.RegistryCalcGradNode(); err != nil {
		return errors.WithStack(err)
	}

	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(model.Learnables()...))

	//トレーニング用のデータを生成
	inputDense, expectedDense := createTrainingData(dimensionNum, classNum, sampleNum, rng)
	starttime := time.Now()
	for i := 0; i < maxEpoch; i++ {
		//入力のノードにトレーニング用のデータを注入
		if err := gorgonia.Let(model.Input, inputDense); err != nil {
			return errors.WithStack(err)
		}
		//期待される結果のためのノード（勾配計算のためのノード）にトレーニング用のデータを注入
		if err := gorgonia.Let(model.Expected, expectedDense); err != nil {
			return errors.WithStack(err)
		}

		if err := vm.RunAll(); err != nil {
			return errors.WithStack(err)
		}

		//勾配から重みとバイアスを更新
		if err := solver.Step(gorgonia.NodesToValueGrads(model.Learnables())); err != nil {
			return errors.WithStack(err)
		}

		vm.Reset()

		//1万回に1回進捗を出力
		if i%10000 == 0 {
			log.Printf("loop:%d, cost:%v, time taken %v\n", i, model.CostValue, time.Since(starttime))
			starttime = time.Now()
		}

	}

	return nil
}

//トレーニング用データの生成
func createTrainingData(dimensionNum int, classNum int, sampleNum int, rng *rand.Rand) (*tensor.Dense, *tensor.Dense) {
	inputArray := make([]float64, sampleNum*classNum*dimensionNum)
	outputArray := make([]float64, sampleNum*classNum*classNum)
	for j := 0; j < classNum; j++ {
		for i := 0; i < sampleNum; i++ {
			rate := float64(i) / float64(sampleNum)
			theta := float64(j)*4.0 + 4.0*rate + rng.NormFloat64()*0.2

			ix := sampleNum*j + i
			inputArray[ix*dimensionNum] = rate * math.Sin(theta)
			inputArray[ix*dimensionNum+1] = rate * math.Cos(theta)
			outputArray[ix*classNum+j] = 1
		}
	}
	input := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(sampleNum*classNum, dimensionNum), tensor.WithBacking(inputArray))
	output := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(sampleNum*classNum, classNum), tensor.WithBacking(outputArray))
	return input, output
}

//テスト用データを生成
func createTestData(dimensionNum int, classNum int, sampleNum int, rng *rand.Rand) (*tensor.Dense, *tensor.Dense) {
	inputArray := make([]float64, sampleNum*classNum*dimensionNum)
	outputArray := make([]float64, sampleNum*classNum*classNum)
	for i := 0; i < sampleNum*classNum; i++ {
		class := rng.Intn(classNum)
		rate := rng.Float64()
		theta := float64(class)*4.0 + 4.0*rate + rng.NormFloat64()*0.2

		inputArray[i*dimensionNum] = rate * math.Sin(theta)
		inputArray[i*dimensionNum+1] = rate * math.Cos(theta)
		outputArray[i*classNum+class] = 1
	}
	input := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(sampleNum*classNum, dimensionNum), tensor.WithBacking(inputArray))
	output := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(sampleNum*classNum, classNum), tensor.WithBacking(outputArray))
	return input, output
}
