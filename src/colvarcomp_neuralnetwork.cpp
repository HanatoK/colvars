#if (__cplusplus >= 201103L)

#include "colvarmodule.h"
#include "colvarvalue.h"
#include "colvarparse.h"
#include "colvar.h"
#include "colvarcomp.h"
#include "colvar_neuralnetworkcompute.h"

#include <map>

using namespace neuralnetworkCV;

colvar::neuralNetwork::neuralNetwork(std::string const &conf): linearCombination(conf) {
    set_function_type("neuralNetwork");
    // the output of neural network consists of multiple values
    // read "output_component" key to determine it
    get_keyval(conf, "output_component", m_output_index);
    std::map<size_t, std::vector<std::string>> nn_config_map;
    size_t layer_index = 1;
    while (true) {
        bool layer_read_ok = false;
        // lookup dense layer
        const std::string key_dense_weights =
            std::string{"layer"} + cvm::to_str(layer_index) + std::string{"_WeightsFile"};
        if (!layer_read_ok && key_lookup(conf, key_dense_weights.c_str())) {
            // continue to lookup biases file and activation
            const std::string key_dense_biases =
                std::string{"layer"} + cvm::to_str(layer_index) + std::string{"_BiasesFile"};
            const std::string key_dense_activation =
                std::string{"layer"} + cvm::to_str(layer_index) + std::string{"_activation"};
            const std::string key_dense_customactivation =
                std::string{"layer"} + cvm::to_str(layer_index) + std::string{"_custom_activation"};
            std::vector<std::string> config_strings(5);
            config_strings.at(0) = "DenseLayer";
            if (!get_keyval(conf, key_dense_weights.c_str(), config_strings.at(1), std::string(""))) {
                cvm::error("Expect keyword \"" + key_dense_weights + "\".\n");
                return;
            }
            if (!get_keyval(conf, key_dense_biases.c_str(), config_strings.at(2), std::string(""))) {
                cvm::error("Expect keyword \"" + key_dense_biases + "\".\n");
                return;
            }
            if (key_lookup(conf, key_dense_activation.c_str())) {
                config_strings.at(3) = "builtin";
                get_keyval(conf, key_dense_activation.c_str(), config_strings.at(4), std::string(""));
            }
            if (key_lookup(conf, key_dense_customactivation.c_str())) {
                if (!config_strings.at(3).empty()) {
                    cvm::error("The activation function has been already specified by " + key_dense_activation +
                               ", which is in conflict with " + key_dense_customactivation + ". Please keep only"
                               " one of the option.\n");
                    return;
                } else {
                    config_strings.at(3) = "custom";
                    get_keyval(conf, key_dense_customactivation.c_str(), config_strings.at(4), std::string(""));
                }
            }
            if (config_strings.at(3).empty()) {
                cvm::error("Expect an activation function for layer " + cvm::to_str(layer_index) + "\n");
                return;
            }
            nn_config_map[layer_index] = config_strings;
            layer_read_ok = true;
        }
        // lookup special layer: circular_to_linear layer
        const std::string key_c2l_c_weights =
            std::string{"circularToLinear_layer"} + cvm::to_str(layer_index) + std::string{"_CircularWeightsFile"};
        if (!layer_read_ok && key_lookup(conf, key_c2l_c_weights.c_str())) {
            // continue to lookup circular biases file, linear weights/biases files and activation
            const std::string key_c2l_c_biases =
                std::string{"circularToLinear_layer"} + cvm::to_str(layer_index) + std::string{"_CircularBiasesFile"};
            const std::string key_c2l_l_weights =
                std::string{"circularToLinear_layer"} + cvm::to_str(layer_index) + std::string{"_LinearWeightsFile"};
            const std::string key_c2l_l_biases =
                std::string{"circularToLinear_layer"} + cvm::to_str(layer_index) + std::string{"_LinearBiasesFile"};
            const std::string key_c2l_activation =
                std::string{"circularToLinear_layer"} + cvm::to_str(layer_index) + std::string{"_activation"};
            const std::string key_c2l_customactivation =
                std::string{"circularToLinear_layer"} + cvm::to_str(layer_index) + std::string{"_custom_activation"};
            std::vector<std::string> config_strings(7);
            config_strings.at(0) = "CircularToLinearLayer";
            if (!get_keyval(conf, key_c2l_c_weights.c_str(), config_strings.at(1), std::string(""))) {
                cvm::error("Expect keyword \"" + key_c2l_c_weights + "\".\n");
                return;
            }
            if (!get_keyval(conf, key_c2l_c_biases.c_str(), config_strings.at(2), std::string(""))) {
                cvm::error("Expect keyword \"" + key_c2l_c_biases + "\".\n");
                return;
            }
            if (!get_keyval(conf, key_c2l_l_weights.c_str(), config_strings.at(3), std::string(""))) {
                cvm::error("Expect keyword \"" + key_c2l_l_weights + "\".\n");
                return;
            }
            if (!get_keyval(conf, key_c2l_l_biases.c_str(), config_strings.at(4), std::string(""))) {
                cvm::error("Expect keyword \"" + key_c2l_l_biases + "\".\n");
                return;
            }
            if (key_lookup(conf, key_c2l_activation.c_str())) {
                config_strings.at(5) = "builtin";
                get_keyval(conf, key_c2l_activation.c_str(), config_strings.at(6), std::string(""));
            }
            if (key_lookup(conf, key_c2l_customactivation.c_str())) {
                if (!config_strings.at(5).empty()) {
                    cvm::error("The activation function has been already specified by " + key_c2l_activation +
                               ", which is in conflict with " + key_c2l_customactivation + ". Please keep only"
                               " one of the option.\n");
                    return;
                } else {
                    config_strings.at(5) = "custom";
                    get_keyval(conf, key_c2l_customactivation.c_str(), config_strings.at(6), std::string(""));
                }
            }
            if (config_strings.at(5).empty()) {
                cvm::error("Expect an activation function for layer " + cvm::to_str(layer_index) + "\n");
                return;
            }
            nn_config_map[layer_index] = config_strings;
            layer_read_ok = true;
        }
        // lookup special layer: circular_to_linear_skewed layer
        const std::string key_c2l_cskewed_weights =
            std::string{"circularToLinearSkewed_layer"} + cvm::to_str(layer_index) + std::string{"_CircularWeightsFile"};
        if (!layer_read_ok && key_lookup(conf, key_c2l_cskewed_weights.c_str())) {
            const std::string key_c2l_cskewed_biases =
                std::string{"circularToLinearSkewed_layer"} + cvm::to_str(layer_index) + std::string{"_CircularBiasesFile"};
            const std::string key_c2l_cskewed_skewness =
                std::string{"circularToLinearSkewed_layer"} + cvm::to_str(layer_index) + std::string{"_CircularSkewnessFile"};
            const std::string key_c2l_lskewed_weights =
                std::string{"circularToLinearSkewed_layer"} + cvm::to_str(layer_index) + std::string{"_LinearWeightsFile"};
            const std::string key_c2l_lskewed_biases =
                std::string{"circularToLinearSkewed_layer"} + cvm::to_str(layer_index) + std::string{"_LinearBiasesFile"};
            const std::string key_c2l_skewed_activation =
                std::string{"circularToLinearSkewed_layer"} + cvm::to_str(layer_index) + std::string{"_activation"};
            const std::string key_c2l_skewed_customactivation =
                std::string{"circularToLinearSkewed_layer"} + cvm::to_str(layer_index) + std::string{"_custom_activation"};
            std::vector<std::string> config_strings(8);
            config_strings.at(0) = "CircularToLinearLayerSkewed";
            if (!get_keyval(conf, key_c2l_cskewed_weights.c_str(), config_strings.at(1), std::string(""))) {
                cvm::error("Expect keyword \"" + key_c2l_cskewed_weights + "\".\n");
                return;
            }
            if (!get_keyval(conf, key_c2l_cskewed_biases.c_str(), config_strings.at(2), std::string(""))) {
                cvm::error("Expect keyword \"" + key_c2l_cskewed_biases + "\".\n");
                return;
            }
            if (!get_keyval(conf, key_c2l_cskewed_skewness.c_str(), config_strings.at(3), std::string(""))) {
                cvm::error("Expect keyword \"" + key_c2l_cskewed_skewness + "\".\n");
                return;
            }
            if (!get_keyval(conf, key_c2l_lskewed_weights.c_str(), config_strings.at(4), std::string(""))) {
                cvm::error("Expect keyword \"" + key_c2l_lskewed_weights + "\".\n");
                return;
            }
            if (!get_keyval(conf, key_c2l_lskewed_biases.c_str(), config_strings.at(5), std::string(""))) {
                cvm::error("Expect keyword \"" + key_c2l_lskewed_biases + "\".\n");
                return;
            }
            if (key_lookup(conf, key_c2l_skewed_activation.c_str())) {
                config_strings.at(6) = "builtin";
                get_keyval(conf, key_c2l_skewed_activation.c_str(), config_strings.at(7), std::string(""));
            }
            if (key_lookup(conf, key_c2l_skewed_customactivation.c_str())) {
                if (!config_strings.at(6).empty()) {
                    cvm::error("The activation function has been already specified by " + key_c2l_skewed_activation +
                               ", which is in conflict with " + key_c2l_skewed_customactivation + ". Please keep only"
                               " one of the option.\n");
                    return;
                } else {
                    config_strings.at(6) = "custom";
                    get_keyval(conf, key_c2l_skewed_customactivation.c_str(), config_strings.at(7), std::string(""));
                }
            }
            if (config_strings.at(6).empty()) {
                cvm::error("Expect an activation function for layer " + cvm::to_str(layer_index) + "\n");
                return;
            }
            nn_config_map[layer_index] = config_strings;
            layer_read_ok = true;
        }
        // lookup special layer: circular_to_linear_yi layer
        const std::string key_theta_a =
            std::string{"circularToLinearYi_layer"} + cvm::to_str(layer_index) + std::string{"_theta_a"};
        if (!layer_read_ok && key_lookup(conf, key_theta_a.c_str())) {
            // continues to lookup theta b
            const std::string key_theta_b =
                std::string{"circularToLinearYi_layer"} + cvm::to_str(layer_index) + std::string{"_theta_b"};
            const std::string key_activation =
                std::string{"circularToLinearYi_layer"} + cvm::to_str(layer_index) + std::string{"_activation"};
            const std::string key_customactivation =
                std::string{"circularToLinearYi_layer"} + cvm::to_str(layer_index) + std::string{"_custom_activation"};
            std::vector<std::string> config_strings(5);
            config_strings.at(0) = "CircularToLinearLayerYi";
            if (!get_keyval(conf, key_theta_a.c_str(), config_strings.at(1), std::string(""))) {
                cvm::error("Expect keyword \"" + key_theta_a + "\".\n");
                return;
            }
            if (!get_keyval(conf, key_theta_b.c_str(), config_strings.at(2), std::string(""))) {
                cvm::error("Expect keyword \"" + key_theta_b + "\".\n");
                return;
            }
            if (key_lookup(conf, key_activation.c_str())) {
                config_strings.at(3) = "builtin";
                get_keyval(conf, key_activation.c_str(), config_strings.at(4), std::string(""));
            }
            if (key_lookup(conf, key_customactivation.c_str())) {
                if (!config_strings.at(3).empty()) {
                    cvm::error("The activation function has been already specified by " + key_activation +
                               ", which is in conflict with " + key_customactivation + ". Please keep only"
                               " one of the option.\n");
                    return;
                } else {
                    config_strings.at(3) = "custom";
                    get_keyval(conf, key_customactivation.c_str(), config_strings.at(4), std::string(""));
                }
            }
            if (config_strings.at(3).empty()) {
                cvm::error("Expect an activation function for layer " + cvm::to_str(layer_index) + "\n");
                return;
            }
            nn_config_map[layer_index] = config_strings;
            layer_read_ok = true;
        }
        if (layer_read_ok) {
            ++layer_index;
            continue;
        } else {
            break;
        }
    }
    // std::make_unique is only available in C++14
    nn = std::unique_ptr<neuralnetworkCV::neuralNetworkCompute>(new neuralnetworkCV::neuralNetworkCompute());
    for (size_t i_layer = 1; i_layer < nn_config_map.size() + 1; ++i_layer) {
        const std::vector<std::string>& layer_config = nn_config_map[i_layer];
        try {
            std::unique_ptr<LayerBase> layer = neuralnetworkCV::createLayer(layer_config);
            // add a new dense layer to network
            if (nn->addLayer(std::move(layer))) {
                if (cvm::debug()) {
                    const std::unique_ptr<LayerBase>& current_layer = nn->getLayer(i_layer - 1);
                    // show information about the neural network
                    if (current_layer->layerType() == "DenseLayer") {
                        const DenseLayer* d = dynamic_cast<const DenseLayer*>(current_layer.get());
                        cvm::log("Dense layer " + cvm::to_str(i_layer) + " : has " + cvm::to_str(current_layer->getInputSize()) + " input nodes and " + cvm::to_str(current_layer->getOutputSize()) + " output nodes.\n");
                        for (size_t i_output = 0; i_output < current_layer->getOutputSize(); ++i_output) {
                            for (size_t j_input = 0; j_input < current_layer->getInputSize(); ++j_input) {
                                cvm::log("    weights[" + cvm::to_str(i_output) + "][" + cvm::to_str(j_input) + "] = " + cvm::to_str(d->getWeight(i_output, j_input)));
                            }
                            cvm::log("    biases[" + cvm::to_str(i_output) + "] = " + cvm::to_str(d->getBias(i_output)) + "\n");
                        }
                    }
                }
            } else {
                throw std::runtime_error("Error: error on adding a new layer.\n");
            }
        } catch (const std::exception& e) {
            cvm::log(e.what());
            cvm::error("Error: error on creating a new layer.\n");
            return;
        }
    }
    size_t input_size = 0;
    for (size_t i_cv = 0; i_cv < cv.size(); ++i_cv) {
        input_size += cv[i_cv]->value().size();
        cvm::log("CV " + cvm::to_str(i_cv) + " has " + cvm::to_str(cv[i_cv]->value().size()) + " component(s).");
    }
    cvm::log("Number of CVs: " + cvm::to_str(cv.size()));
    cvm::log("Number of input components: " + cvm::to_str(input_size));
    cvm::log("Number of input units of the neural network: " + cvm::to_str(nn->getInputSize()));
    if (input_size != nn->getInputSize()) {
        cvm::error("The total number of input components does not match the required input units of the neural network.");
        return;
    }
    nn->input().resize(input_size);
    x.type(colvarvalue::type_scalar);
}

colvar::neuralNetwork::~neuralNetwork() {
}

void colvar::neuralNetwork::calc_value() {
    x.reset();
    size_t input_index = 0;
    for (size_t i_cv = 0; i_cv < cv.size(); ++i_cv) {
        cv[i_cv]->calc_value();
        const colvarvalue& current_cv_value = cv[i_cv]->value();
        // for current nn implementation we have to assume taht types are always scaler
        if (current_cv_value.type() == colvarvalue::type_scalar) {
            nn->input()[input_index] = cv[i_cv]->sup_coeff * (cvm::pow(current_cv_value.real_value, cv[i_cv]->sup_np));
        } else {
            for (size_t j = 0; j < current_cv_value.size(); ++j) {
                nn->input()[input_index+j] = cv[i_cv]->sup_coeff * (current_cv_value[j]);
            }
        }
        input_index += current_cv_value.size();
    }
    nn->compute();
    x = nn->getOutput(m_output_index);
}

void colvar::neuralNetwork::calc_gradients() {
    size_t input_index = 0;
    for (size_t i_cv = 0; i_cv < cv.size(); ++i_cv) {
        cv[i_cv]->calc_gradients();
        const colvarvalue& current_cv_value = cv[i_cv]->value();
        if (cv[i_cv]->is_enabled(f_cvc_explicit_gradient)) {
            const cvm::real factor_polynomial = getPolynomialFactorOfCVGradient(i_cv);
            for (size_t j = 0; j < current_cv_value.size(); ++j) {
                const cvm::real factor = nn->getGradient(m_output_index, input_index + j);
                // TODO: what should I do for vector CVs with explicit gradients?
                for (size_t k_ag = 0 ; k_ag < cv[i_cv]->atom_groups.size(); ++k_ag) {
                    for (size_t l_atom = 0; l_atom < (cv[i_cv]->atom_groups)[k_ag]->size(); ++l_atom) {
                        (*(cv[i_cv]->atom_groups)[k_ag])[l_atom].grad = factor_polynomial * factor * (*(cv[i_cv]->atom_groups)[k_ag])[l_atom].grad;
                    }
                }
            }
        }
        input_index += current_cv_value.size();
    }
}

void colvar::neuralNetwork::apply_force(colvarvalue const &force) {
    size_t input_index = 0;
    for (size_t i_cv = 0; i_cv < cv.size(); ++i_cv) {
        const colvarvalue& current_cv_value = cv[i_cv]->value();
        // If this CV us explicit gradients, then atomic gradients is already calculated
        // We can apply the force to atom groups directly
        if (cv[i_cv]->is_enabled(f_cvc_explicit_gradient)) {
            for (size_t k_ag = 0 ; k_ag < cv[i_cv]->atom_groups.size(); ++k_ag) {
                (cv[i_cv]->atom_groups)[k_ag]->apply_colvar_force(force.real_value);
            }
        } else {
            // Compute factors for polynomial combinations
            const cvm::real factor_polynomial = getPolynomialFactorOfCVGradient(i_cv);
            // Cannot initialize a colvarvalue with the type directly since the size cannot be initialize in that way
            // colvarvalue cv_force(current_cv_value.type());
            colvarvalue cv_force(current_cv_value);
            cv_force.reset();
            for (size_t j = 0; j < current_cv_value.size(); ++j) {
                const cvm::real factor = nn->getGradient(m_output_index, input_index + j);
                cv_force[j] = force.real_value * factor * factor_polynomial;
            }
            cv[i_cv]->apply_force(cv_force);
            if (cvm::debug()) {
                cvm::log("Apply colvar force " + cvm::to_str(cv_force) + " to sub CV " + cvm::to_str(i_cv));
            }
        }
        input_index += current_cv_value.size();
    }
}

#endif
