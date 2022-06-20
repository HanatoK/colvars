#include "colvarcomp.h"

#if (__cplusplus >= 201103L)

colvar::random_uniform::random_uniform(std::string const& conf)
  :cvc(conf), m_random_device(), m_random_engine(m_random_device()) {
    set_function_type("randomUniform");
    get_keyval(conf, "min", m_min);
    get_keyval(conf, "max", m_max);
    m_distribution = std::uniform_real_distribution<>(m_min, m_max);
    x.type(colvarvalue::type_scalar);
}

colvar::random_uniform::~random_uniform() {}

void colvar::random_uniform::calc_value() {
    x = m_distribution(m_random_engine);
}

void colvar::random_uniform::calc_gradients() {
}

void colvar::random_uniform::apply_force(colvarvalue const& force) {
}

#endif
