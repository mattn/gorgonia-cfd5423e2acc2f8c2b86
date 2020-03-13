package twolayernn

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

//2層ニューラルネットワーク
type TwoLayerNeuralNetworkModel struct {
	inputSize, hiddenSize, outputSize, sampleNum int
	// グラフ
	graph *gorgonia.ExprGraph
	//重み
	weight1, weight2 *gorgonia.Node
	//バイアス
	bias1, bias2 *gorgonia.Node
	//推論用
	Input, Output *gorgonia.Node
	OutputValue   gorgonia.Value
	//勾配計算用
	Expected  *gorgonia.Node
	CostValue gorgonia.Value
}

//各層のノードを生成し、モデルに設定して返す
func New(graph *gorgonia.ExprGraph, inputSize, hiddenSize, outputSize, sampleNum int) *TwoLayerNeuralNetworkModel {
	weight1 := gorgonia.NewMatrix(graph, tensor.Float64, gorgonia.WithShape(inputSize, hiddenSize), gorgonia.WithName("Weight1"), gorgonia.WithInit(gorgonia.Gaussian(0, 1)))
	weight2 := gorgonia.NewMatrix(graph, tensor.Float64, gorgonia.WithShape(hiddenSize, outputSize), gorgonia.WithName("Weight2"), gorgonia.WithInit(gorgonia.Gaussian(0, 1)))
	bias1 := gorgonia.NewMatrix(graph, tensor.Float64, gorgonia.WithShape(1, hiddenSize), gorgonia.WithName("Bias1"), gorgonia.WithInit(gorgonia.Zeroes()))
	bias2 := gorgonia.NewMatrix(graph, tensor.Float64, gorgonia.WithShape(1, outputSize), gorgonia.WithName("Bias2"), gorgonia.WithInit(gorgonia.Zeroes()))

	input := gorgonia.NewTensor(graph, tensor.Float64, 2, gorgonia.WithShape(sampleNum, inputSize), gorgonia.WithName("Input"))

	model := &TwoLayerNeuralNetworkModel{
		graph:      graph,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
		sampleNum:  sampleNum,
		Input:      input,
		weight1:    weight1,
		weight2:    weight2,
		bias1:      bias1,
		bias2:      bias2,
	}

	model.RegistryForwardNode()

	return model
}

//推論のためのノードを登録
func (m *TwoLayerNeuralNetworkModel) RegistryForwardNode() {
	layer1Node1 := gorgonia.Must(gorgonia.Mul(m.Input, m.weight1))
	layer1Node2 := gorgonia.Must(gorgonia.BroadcastAdd(layer1Node1, m.bias1, nil, []byte{0}))
	layer1Result := gorgonia.Must(gorgonia.Sigmoid(layer1Node2))

	layer2Node1 := gorgonia.Must(gorgonia.Mul(layer1Result, m.weight2))
	layer2Result := gorgonia.Must(gorgonia.BroadcastAdd(layer2Node1, m.bias2, nil, []byte{0}))

	m.Output = layer2Result
	gorgonia.Read(m.Output, &m.OutputValue)
}

//勾配計算のためのノードを登録
func (m *TwoLayerNeuralNetworkModel) RegistryCalcGradNode() error {
	expectedNode := gorgonia.NewTensor(m.graph, tensor.Float64, 2, gorgonia.WithShape(m.sampleNum, m.outputSize), gorgonia.WithName("Expected"))

	softMaxNode := gorgonia.Must(gorgonia.SoftMax(m.Output))
	logNode := gorgonia.Must(gorgonia.Log(softMaxNode))
	lossesNode := gorgonia.Must(gorgonia.HadamardProd(expectedNode, logNode))
	sumNode := gorgonia.Must(gorgonia.Sum(lossesNode, 1))
	costNode := gorgonia.Must(gorgonia.Mean(sumNode))
	costNode = gorgonia.Must(gorgonia.Neg(costNode))

	if _, err := gorgonia.Grad(costNode, m.Learnables()...); err != nil {
		return errors.WithStack(err)
	}

	m.Expected = expectedNode
	gorgonia.Read(costNode, &m.CostValue)

	return nil
}

//最適化対象のノードを返す
func (m *TwoLayerNeuralNetworkModel) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.weight1, m.weight2, m.bias1, m.bias2}
}
