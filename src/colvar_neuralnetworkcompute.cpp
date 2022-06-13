// -*- Mode:c++; c-basic-offset: 4; -*-

// This file is part of the Collective Variables module (Colvars).
// The original version of Colvars and its updates are located at:
// https://github.com/Colvars/colvars
// Please update all Colvars source files before making any changes.
// If you wish to distribute your changes, please submit them to the
// Colvars repository at GitHub.

#include <iostream>
#include <fstream>

#if (__cplusplus >= 201103L)
#include "colvar_neuralnetworkcompute.h"
#include "colvarparse.h"
#include "colvarproxy.h"

namespace neuralnetworkCV {
std::map<std::string, std::pair<std::function<double(double)>, std::function<double(double)>>> activation_function_map
{
    {"tanh",     {[](double x){return std::tanh(x);},
                  [](double x){return 1.0 - std::tanh(x) * std::tanh(x);}}},
    {"sigmoid",  {[](double x){return 1.0 / (1.0 + std::exp(-x));},
                  [](double x){return std::exp(-x) / ((1.0 + std::exp(-x)) * (1.0 + std::exp(-x)));}}},
    {"linear",   {[](double x){return x;},
                  [](double /*x*/){return 1.0;}}},
    {"relu",     {[](double x){return x < 0. ? 0. : x;},
                  [](double x){return x < 0. ? 0. : 1.;}}},
    {"lrelu100", {[](double x){return x < 0. ? 0.01 * x : x;},
                  [](double x){return x < 0. ? 0.01     : 1.;}}},
    {"elu",      {[](double x){return x < 0. ? std::exp(x)-1. : x;},
                  [](double x){return x < 0. ? std::exp(x)    : 1.;}}}
};

std::map<std::string, std::function<std::unique_ptr<LayerBase>(const std::vector<std::string>& config)>> available_layer_map
{
    {"DenseLayer",              [](const std::vector<std::string>& config){return std::unique_ptr<DenseLayer>(new DenseLayer(config));}},
    {"CircularToLinearLayer",   [](const std::vector<std::string>& config){return std::unique_ptr<CircularToLinearLayer>(new CircularToLinearLayer(config));}},
};

#ifdef LEPTON
CustomActivationFunction::CustomActivationFunction():
expression(), value_evaluator(nullptr), gradient_evaluator(nullptr),
input_reference(nullptr), derivative_reference(nullptr) {}

CustomActivationFunction::CustomActivationFunction(const std::string& expression_string):
expression(), value_evaluator(nullptr), gradient_evaluator(nullptr),
input_reference(nullptr), derivative_reference(nullptr) {
    setExpression(expression_string);
}

CustomActivationFunction::CustomActivationFunction(const CustomActivationFunction& source):
expression(), value_evaluator(nullptr), gradient_evaluator(nullptr),
input_reference(nullptr), derivative_reference(nullptr) {
    // check if the source object is initialized
    if (source.value_evaluator != nullptr) {
        this->setExpression(source.expression);
    }
}

CustomActivationFunction& CustomActivationFunction::operator=(const CustomActivationFunction& source) {
    if (source.value_evaluator != nullptr) {
        this->setExpression(source.expression);
    } else {
        expression = std::string();
        value_evaluator = nullptr;
        gradient_evaluator = nullptr;
        input_reference = nullptr;
        derivative_reference = nullptr;
    }
    return *this;
}

void CustomActivationFunction::setExpression(const std::string& expression_string) {
    expression = expression_string;
    Lepton::ParsedExpression parsed_expression;
    // the variable must be "x" for the input of an activation function
    const std::string activation_input_variable{"x"};
    // parse the expression
    try {
        parsed_expression = Lepton::Parser::parse(expression);
    } catch (...) {
        cvm::error("Error parsing or compiling expression \"" + expression + "\".\n", COLVARS_INPUT_ERROR);
    }
    // compile the expression
    try {
        value_evaluator = std::unique_ptr<Lepton::CompiledExpression>(new Lepton::CompiledExpression(parsed_expression.createCompiledExpression()));
    } catch (...) {
        cvm::error("Error compiling expression \"" + expression + "\".\n", COLVARS_INPUT_ERROR);
    }
    // create a compiled expression for the derivative
    try {
        gradient_evaluator = std::unique_ptr<Lepton::CompiledExpression>(new Lepton::CompiledExpression(parsed_expression.differentiate(activation_input_variable).createCompiledExpression()));
    } catch (...) {
        cvm::error("Error creating compiled expression for variable \"" + activation_input_variable + "\".\n", COLVARS_INPUT_ERROR);
    }
    // get the reference to the input variable in the compiled expression
    try {
        input_reference = &(value_evaluator->getVariableReference(activation_input_variable));
    } catch (...) {
        cvm::error("Error on getting the reference to variable \"" + activation_input_variable + "\" in the compiled expression.\n", COLVARS_INPUT_ERROR);
    }
    // get the reference to the input variable in the compiled derivative expression
    try {
        derivative_reference = &(gradient_evaluator->getVariableReference(activation_input_variable));
    } catch (...) {
        cvm::error("Error on getting the reference to variable \"" + activation_input_variable + "\" in the compiled derivative exprssion.\n", COLVARS_INPUT_ERROR);
    }
}

std::string CustomActivationFunction::getExpression() const {
    return expression;
}

double CustomActivationFunction::evaluate(double x) const {
    *input_reference = x;
    return value_evaluator->evaluate();
}

double CustomActivationFunction::derivative(double x) const {
    *derivative_reference = x;
    return gradient_evaluator->evaluate();
}
#endif


DenseLayer::DenseLayer(const std::vector<std::string>& config): LayerBase(config) {
    const std::string& weights_file = config.at(1);
    const std::string& biases_file = config.at(2);
    const std::string& activation_type = config.at(3);
    const std::string& activation_str = config.at(4);
    if (activation_type == "custom") {
#ifdef LEPTON
        m_use_custom_activation = true;
        m_custom_activation_function = CustomActivationFunction(activation_str);
        readFromFile(weights_file, biases_file);
#else
        throw std::runtime_error("Lepton is required for custom activation function \"" + activation_str + "\", but it is not compiled.");
#endif
    } else if (activation_type == "builtin") {
#ifdef LEPTON
        m_use_custom_activation = false;
#endif
        auto search_builtin_activation_map = activation_function_map.find(activation_str);
        if (search_builtin_activation_map != activation_function_map.end()) {
            m_activation_function = search_builtin_activation_map->second.first;
            m_activation_function_derivative = search_builtin_activation_map->second.second;
            readFromFile(weights_file, biases_file);
        } else {
            throw std::runtime_error("Unkown activation function \"" + activation_str + "\".");
        }
    } else {
        throw std::runtime_error("Unknown activation type " + activation_type);
    }
}

void DenseLayer::readFromFile(const std::string& weights_file, const std::string& biases_file) {
    // parse weights file
    readSpaceSeparatedFileToVector(weights_file, m_weights);
    // parse biases file
    readSpaceSeparatedFileToVector(biases_file, m_biases);
    m_input_size = m_weights[0].size();
    m_output_size = m_weights.size();
}

void DenseLayer::setActivationFunction(const std::function<double(double)>& f, const std::function<double(double)>& df) {
    m_activation_function = f;
    m_activation_function_derivative = df;
}

void DenseLayer::compute(const std::vector<double>& input, std::vector<double>& output) const {
    for (size_t i = 0; i < m_output_size; ++i) {
        output[i] = 0;
        for (size_t j = 0; j < m_input_size; ++j) {
            output[i] += input[j] * m_weights[i][j];
        }
        output[i] += m_biases[i];
#ifdef LEPTON
        if (m_use_custom_activation) {
            output[i] = m_custom_activation_function.evaluate(output[i]);
        } else {
#endif
            output[i] = m_activation_function(output[i]);
#ifdef LEPTON
        }
#endif
    }
}

double DenseLayer::computeGradientElement(const std::vector<double>& input, const size_t i, const size_t j) const {
    double sum_with_bias = 0;
    for (size_t j_in = 0; j_in < m_input_size; ++j_in) {
        sum_with_bias += input[j_in] * m_weights[i][j_in];
    }
    sum_with_bias += m_biases[i];
#ifdef LEPTON
    if (m_use_custom_activation) {
        const double grad_ij = m_custom_activation_function.derivative(sum_with_bias) * m_weights[i][j];
        return grad_ij;
    } else {
#endif
        const double grad_ij = m_activation_function_derivative(sum_with_bias) * m_weights[i][j];
        return grad_ij;
#ifdef LEPTON
    }
#endif
}

void DenseLayer::computeGradient(const std::vector<double>& input, std::vector<std::vector<double>>& output_grad) const {
    for (size_t j = 0; j < m_input_size; ++j) {
        for (size_t i = 0; i < m_output_size; ++i) {
            output_grad[i][j] = computeGradientElement(input, i, j);
        }
    }
}

CircularToLinearLayer::CircularToLinearLayer(const std::vector<std::string>& config): LayerBase(config) {
    const std::string& circular_weights_file = config.at(1);
    const std::string& circular_biases_file = config.at(2);
    const std::string& linear_weights_file = config.at(3);
    const std::string& linear_biases_file = config.at(4);
    const std::string& activation_type = config.at(5);
    const std::string& activation_str = config.at(6);
    readFromFile(circular_weights_file, circular_biases_file,
                 linear_weights_file, linear_biases_file);
    if (activation_type == "custom") {
#ifdef LEPTON
        m_use_custom_activation = true;
        m_custom_activation_function = CustomActivationFunction(activation_str);
#else
        throw std::runtime_error("Lepton is required for custom activation function \"" + activation_str + "\", but it is not compiled.");
#endif
    } else if (activation_type == "builtin") {
#ifdef LEPTON
        m_use_custom_activation = false;
#endif
        auto search_builtin_activation_map = activation_function_map.find(activation_str);
        if (search_builtin_activation_map != activation_function_map.end()) {
            m_activation_function = search_builtin_activation_map->second.first;
            m_activation_function_derivative = search_builtin_activation_map->second.second;
        } else {
            throw std::runtime_error("Unkown activation function \"" + activation_str + "\".");
        }
    } else {
        throw std::runtime_error("Unknown activation type " + activation_type);
    }
}

void CircularToLinearLayer::readFromFile(const std::string& circular_weights_file, const std::string& circular_biases_file,
                                         const std::string& linear_weights_file, const std::string& linear_biases_file) {
    readSpaceSeparatedFileToVector(circular_weights_file, m_circular_weights);
    readSpaceSeparatedFileToVector(circular_biases_file, m_circular_biases);
    readSpaceSeparatedFileToVector(linear_weights_file, m_linear_weights);
    readSpaceSeparatedFileToVector(linear_biases_file, m_linear_biases);
    m_input_size = m_circular_weights.size();
    m_order = m_circular_biases.size();
    m_output_size = m_input_size;
    // some sanity checks
    if (m_circular_weights.size() == 0) throw std::runtime_error("Failed to read circular weights.");
    if (m_linear_biases.size() != m_input_size) throw std::runtime_error("Inconsistent number of linear biases.");
    if (m_circular_biases.size() == 0) throw std::runtime_error("Failed to read circular biases.");
    if (m_circular_biases.size() != m_linear_weights.size())
        throw std::runtime_error(
            "Inconsistent order (" + std::to_string(m_circular_biases.size()) +
            " circular biases) but (" + std::to_string(m_linear_weights.size()) +
            " linear weights.");
    if (m_circular_biases[0].size() != m_linear_weights[0].size()) throw std::runtime_error("Inconsistent number of input units.");
}

void CircularToLinearLayer::compute(const std::vector<double>& input, std::vector<double>& output) const {
    for (size_t i = 0; i < m_input_size; ++i) {
        output[i] = m_linear_biases[i];
        for (size_t j = 0; j < m_order; ++j) {
            output[i] += m_linear_weights[j][i] * std::cos((double)(j+1) * m_circular_weights[i] * input[i] - m_circular_biases[j][i]);
        }
#ifdef LEPTON
        if (m_use_custom_activation) {
            output[i] = m_custom_activation_function.evaluate(output[i]);
        } else {
#endif
            output[i] = m_activation_function(output[i]);
#ifdef LEPTON
        }
#endif
    }
}

double CircularToLinearLayer::computeGradientElement(const std::vector<double>& input, const size_t i, const size_t j) const {
    if (i != j) return 0.0;
    double grad = 0.0;
    double sum = m_linear_biases[i];
    for (size_t k = 0; k < m_order; ++k) {
        sum += m_linear_weights[k][i] * std::cos((double)(k+1) * m_circular_weights[i] * input[i] - m_circular_biases[k][i]);
        grad += -1.0 * m_linear_weights[k][i] * (double)(k+1) * m_circular_weights[i] * std::sin((double)(k+1) * m_circular_weights[i] * input[i] - m_circular_biases[k][i]);
    }
#ifdef LEPTON
    if (m_use_custom_activation) {
        grad *= m_custom_activation_function.derivative(sum);
    } else {
#endif
        grad *= m_activation_function_derivative(sum);
#ifdef LEPTON
    }
#endif
    return grad;
}

void CircularToLinearLayer::computeGradient(const std::vector<double>& input, std::vector<std::vector<double>>& output_grad) const {
    for (size_t j = 0; j < m_input_size; ++j) {
        for (size_t i = 0; i < m_output_size; ++i) {
            output_grad[i][j] = computeGradientElement(input, i, j);
        }
    }
}

neuralNetworkCompute::neuralNetworkCompute(std::vector<std::unique_ptr<LayerBase>> dense_layers): m_layers(std::move(dense_layers)) {
    m_layers_output.resize(m_layers.size());
    m_grads_tmp.resize(m_layers.size());
    for (size_t i_layer = 0; i_layer < m_layers_output.size(); ++i_layer) {
        m_layers_output[i_layer].assign(m_layers[i_layer]->getOutputSize(), 0);
        m_grads_tmp[i_layer].assign(m_layers[i_layer]->getOutputSize(), std::vector<double>(m_layers[i_layer]->getInputSize(), 0));
    }
}

size_t neuralNetworkCompute::getInputSize() const {
    if (m_layers.empty()) return 0;
    else return m_layers[0]->getInputSize();
}

bool neuralNetworkCompute::addLayer(std::unique_ptr<LayerBase> layer) {
    if (m_layers.empty()) {
        // add layer to this ann directly if m_layers is empty
        m_layers_output.push_back(std::vector<double>(layer->getOutputSize()));
        m_grads_tmp.push_back(std::vector<std::vector<double>>(layer->getOutputSize(), std::vector<double>(layer->getInputSize(), 0)));
        m_layers.push_back(std::move(layer));
        return true;
    } else {
        // otherwise, we need to check if the output of last layer in m_layers matches the input of layer to be added
        if (m_layers.back()->getOutputSize() == layer->getInputSize()) {
            m_layers_output.push_back(std::vector<double>(layer->getOutputSize()));
            m_grads_tmp.push_back(std::vector<std::vector<double>>(layer->getOutputSize(), std::vector<double>(layer->getInputSize(), 0)));
            m_layers.push_back(std::move(layer));
            return true;
        } else {
            return false;
        }
    }
}

std::vector<std::vector<double>> neuralNetworkCompute::multiply_matrix(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    const size_t m = A.size();
    const size_t n = B.size();
    if (A[0].size() != n) {
        std::cerr << "Error on multiplying matrices!\n";
    }
    const size_t t = B[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(t, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < t; ++j) {
            for (size_t k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

void neuralNetworkCompute::compute() {
    if (m_layers.empty()) {
        return;
    }
    size_t i_layer;
    m_layers[0]->compute(m_input, m_layers_output[0]);
    for (i_layer = 1; i_layer < m_layers.size(); ++i_layer) {
        m_layers[i_layer]->compute(m_layers_output[i_layer - 1], m_layers_output[i_layer]);
    }
    // gradients of each layer
    m_layers[0]->computeGradient(m_input, m_grads_tmp[0]);
    for (i_layer = 1; i_layer < m_layers.size(); ++i_layer) {
        m_layers[i_layer]->computeGradient(m_layers_output[i_layer - 1], m_grads_tmp[i_layer]);
    }
    // chain rule
    if (m_layers.size() > 1) {
        m_chained_grad = multiply_matrix(m_grads_tmp[1], m_grads_tmp[0]);
        for (i_layer = 2; i_layer < m_layers.size(); ++i_layer) {
            m_chained_grad = multiply_matrix(m_grads_tmp[i_layer], m_chained_grad);
        }
    } else {
        m_chained_grad = m_grads_tmp[0];
    }
}

std::unique_ptr<LayerBase> createLayer(const std::vector<std::string>& config) {
    auto search = available_layer_map.find(config.at(0));
    if (search != available_layer_map.end()) {
        return search->second(config);
    } else {
        throw std::runtime_error("Failed to create a new layer of type \"" + config.at(0) + "\"");
        return nullptr;
    }
}

void readSpaceSeparatedFileToVector(const std::string& filename, std::vector<std::vector<double>>& vec) {
    // 2D case: for a text file has multiple columns. If the content of the file is:
    // 1.0 2.0
    // -2.0 3.0
    // then vec will be {{1.0, 2.0}, {-2.0, 3.0}}
    vec.clear();
    std::string line;
    std::ifstream ifs(filename.c_str());
    if (!ifs) throw std::runtime_error("Cannot open file " + filename);
    while (std::getline(ifs, line)) {
        if (ifs.bad()) throw std::runtime_error("I/O error while reading " + filename);
        std::vector<std::string> splitted_data;
        colvarparse::split_string(line, std::string{" "}, splitted_data);
        if (splitted_data.size() > 0) {
            std::vector<double> tmp(splitted_data.size(), 0.0);
            for (size_t i = 0; i < splitted_data.size(); ++i) {
                try {
                    tmp[i] = std::stod(splitted_data[i]);
                } catch (...) {
                    throw std::runtime_error("Cannot convert " + splitted_data[i] + " to a number while reading file " + filename);
                }
            }
            vec.push_back(tmp);
        }
    }
}

void readSpaceSeparatedFileToVector(const std::string& filename, std::vector<double>& vec) {
    // 1D case: for a text file has only a single column. For example:
    // if the file has:
    // 1.0
    // 2.0
    // 3.0
    // this will read a vector as {1.0, 2.0, 3.0}
    vec.clear();
    std::string line;
    std::ifstream ifs(filename.c_str());
    if (!ifs) throw std::runtime_error("Cannot open file " + filename);
    while (std::getline(ifs, line)) {
        if (ifs.bad()) throw std::runtime_error("I/O error while reading " + filename);
        std::vector<std::string> splitted_data;
        colvarparse::split_string(line, std::string{" "}, splitted_data);
        if (splitted_data.size() > 0) {
            double tmp = 0;
            try {
                tmp = std::stod(splitted_data[0]);
            } catch (...) {
                throw std::runtime_error("Cannot convert " + splitted_data[0] + " to a number while reading file " + filename);
            }
            vec.push_back(tmp);
        }
    }
}
}

#endif
