import unittest
import jax.numpy as jnp
from jax import random
from nanodl import *

class TestKANLinearVariants(unittest.TestCase):
    def setUp(self):
        self.in_features = 4
        self.out_features = 3
        self.degree = 5

        self.key = random.PRNGKey(0)
        self.x = random.normal(self.key, (10, self.in_features)) 

        self.models = {
            "ChebyKANLinear": ChebyKANLinear(self.in_features, self.out_features, self.degree),
            "LegendreKANLinear": LegendreKANLinear(self.in_features, self.out_features, self.degree),
            "MonomialKANLinear": MonomialKANLinear(self.in_features, self.out_features, self.degree),
            "FourierKANLinear": FourierKANLinear(self.in_features, self.out_features, self.degree),
            "HermiteKANLinear": HermiteKANLinear(self.in_features, self.out_features, self.degree),
        }

    def test_model_outputs(self):
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                variables = model.init(self.key, self.x)
                output = model.apply(variables, self.x)
                self.assertEqual(output.shape, (10, self.out_features))
                self.assertTrue(jnp.all(jnp.isfinite(output)))


if __name__ == "__main__":
    unittest.main()