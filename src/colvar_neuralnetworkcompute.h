// -*- c++ -*-

// This file is part of the Collective Variables module (Colvars).
// The original version of Colvars and its updates are located at:
// https://github.com/Colvars/colvars
// Please update all Colvars source files before making any changes.
// If you wish to distribute your changes, please submit them to the
// Colvars repository at GitHub.

#if (__cplusplus >= 201103L)
#ifndef NEURALNETWORKCOMPUTE_H
#define NEURALNETWORKCOMPUTE_H

#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <memory>
#include <map>

#ifdef LEPTON
#include "Lepton.h"
#endif

namespace neuralnetworkCV {
/// mapping from a string to the activation function and its derivative
extern std::map<std::string, std::pair<std::function<double(double)>, std::function<double(double)>>> activation_function_map;

#ifdef LEPTON
// allow to define a custom activation function
class CustomActivationFunction {
public:
    /// empty constructor
    CustomActivationFunction();
    /// construct by an mathematical expression
    CustomActivationFunction(const std::string& expression_string);
    /// copy constructor
    CustomActivationFunction(const CustomActivationFunction& source);
    /// overload assignment operator
    CustomActivationFunction& operator=(const CustomActivationFunction& source);
    /// setter for the custom expression
    void setExpression(const std::string& expression_string);
    /// getter for the custom expression
    std::string getExpression() const;
    /// evaluate the value of an expression
    double evaluate(double x) const;
    /// evaluate the gradient of an expression
    double derivative(double x) const;
private:
    std::string expression;
    std::unique_ptr<Lepton::CompiledExpression> value_evaluator;
    std::unique_ptr<Lepton::CompiledExpression> gradient_evaluator;
    double* input_reference;
    double* derivative_reference;
};
#endif

// abstract interface of all layers
class LayerBase {
public:
    LayerBase(const std::vector<std::string>& config) {}
    virtual ~LayerBase() {}
    /// get the input size
    virtual size_t getInputSize() const = 0;
    /// get the output size
    virtual size_t getOutputSize() const = 0;
    /// compute the value of this layer
    virtual void compute(const std::vector<double>& input, std::vector<double>& output) const = 0;
    /// compute the gradient of i-th output wrt j-th input
    virtual double computeGradientElement(const std::vector<double>& input, const size_t i, const size_t j) const = 0;
    /// output[i][j] is the gradient of i-th output wrt j-th input
    virtual void computeGradient(const std::vector<double>& input, std::vector<std::vector<double>>& output_grad) const = 0;
    /// get the type of the layer
    virtual std::string layerType() const = 0;
};

class DenseLayer: public LayerBase {
private:
    size_t m_input_size;
    size_t m_output_size;
    std::function<double(double)> m_activation_function;
    std::function<double(double)> m_activation_function_derivative;
#ifdef LEPTON
    bool m_use_custom_activation;
    CustomActivationFunction m_custom_activation_function;
#else
    static const bool m_use_custom_activation = false;
#endif
    /// weights[i][j] is the weight of the i-th output and the j-th input
    std::vector<std::vector<double>> m_weights;
    /// bias of each node
    std::vector<double> m_biases;
public:
    /// constructor with a vector with 5 strings
    DenseLayer(const std::vector<std::string>& config);
    /// read weights and biases from file
    void readFromFile(const std::string& weights_file, const std::string& biases_file);
    /// setup activation function
    void setActivationFunction(const std::function<double(double)>& f, const std::function<double(double)>& df);
    /// compute the value of this layer
    void compute(const std::vector<double>& input, std::vector<double>& output) const override;
    /// compute the gradient of i-th output wrt j-th input
    double computeGradientElement(const std::vector<double>& input, const size_t i, const size_t j) const override;
    /// output[i][j] is the gradient of i-th output wrt j-th input
    void computeGradient(const std::vector<double>& input, std::vector<std::vector<double>>& output_grad) const override;
    /// get the input size
    size_t getInputSize() const override {
        return m_input_size;
    }
    /// get the output size
    size_t getOutputSize() const override {
        return m_output_size;
    }
    /// get the weights
    double getWeight(size_t i, size_t j) const {
        return m_weights[i][j];
    }
    /// get the biases
    double getBias(size_t i) const {
        return m_biases[i];
    }
    /// get the type of the layer
    std::string layerType() const override {
        return "DenseLayer";
    }
    ~DenseLayer() {}
};

class CircularToLinearLayer: public LayerBase {
private:
    size_t m_order;
    size_t m_input_size;
    size_t m_output_size;
    std::function<double(double)> m_activation_function;
    std::function<double(double)> m_activation_function_derivative;
#ifdef LEPTON
    bool m_use_custom_activation;
    CustomActivationFunction m_custom_activation_function;
#else
    static const bool m_use_custom_activation = false;
#endif
    std::vector<double> m_circular_weights;
    std::vector<std::vector<double>> m_circular_biases;
    std::vector<std::vector<double>> m_linear_weights;
    std::vector<double> m_linear_biases;
public:
    /// constructor with a vector with 7 strings
    CircularToLinearLayer(const std::vector<std::string>& config);
    /// read weights and biases from file
    void readFromFile(const std::string& circular_weights_file, const std::string& circular_biases_file,
                      const std::string& linear_weights_file, const std::string& linear_biases_file);
    /// get the input size
    size_t getInputSize() const override {
        return m_input_size;
    }
    /// get the output size
    size_t getOutputSize() const override {
        return m_output_size;
    }
    /// compute the value of this layer
    void compute(const std::vector<double>& input, std::vector<double>& output) const override;
    /// compute the gradient of i-th output wrt j-th input
    double computeGradientElement(const std::vector<double>& input, const size_t i, const size_t j) const override;
    /// output[i][j] is the gradient of i-th output wrt j-th input
    void computeGradient(const std::vector<double>& input, std::vector<std::vector<double>>& output_grad) const override;
    /// get the type of the layer
    std::string layerType() const override {
        return "CircularToLinearLayer";
    }
    ~CircularToLinearLayer() {}
};

class neuralNetworkCompute {
private:
    std::vector<std::unique_ptr<LayerBase>> m_layers;
    std::vector<double> m_input;
    /// temporary output for each layer, useful to speedup the gradients' calculation
    std::vector<std::vector<double>> m_layers_output;
    std::vector<std::vector<std::vector<double>>> m_grads_tmp;
    std::vector<std::vector<double>> m_chained_grad;
private:
    /// helper function: multiply two matrix constructed from 2D vector
    static std::vector<std::vector<double>> multiply_matrix(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
public:
    neuralNetworkCompute(): m_layers(0), m_layers_output(0) {}
    neuralNetworkCompute(std::vector<std::unique_ptr<LayerBase>> dense_layers);
    bool addLayer(std::unique_ptr<LayerBase> layer);
    // for faster computation
    const std::vector<double>& input() const {return m_input;}
    std::vector<double>& input() {return m_input;}
    /// compute the values and the gradients of all output nodes
    void compute();
    double getOutput(const size_t i) const {return m_layers_output.back()[i];}
    double getGradient(const size_t i, const size_t j) const {return m_chained_grad[i][j];}
    /// get a specified layer
    const std::unique_ptr<LayerBase>& getLayer(const size_t i) const {return m_layers[i];}
    /// get the number of layers
    size_t getNumberOfLayers() const {return m_layers.size();}
};

extern std::map<std::string, std::function<std::unique_ptr<LayerBase>(const std::vector<std::string>& config)>> available_layer_map;

/// factory function for creating a new layer
std::unique_ptr<LayerBase> createLayer(const std::vector<std::string>& config);

template <typename>
struct is_std_vector: std::false_type {};

template <typename T, typename... Ts>
struct is_std_vector<std::vector<T, Ts...>> : std::true_type {};

/// helper functions to read space-separeted text files into a 1D or 2D vector
void readSpaceSeparatedFileToVector(const std::string& filename, std::vector<double>& vec);
void readSpaceSeparatedFileToVector(const std::string& filename, std::vector<std::vector<double>>& vec);

}
#endif
#endif
