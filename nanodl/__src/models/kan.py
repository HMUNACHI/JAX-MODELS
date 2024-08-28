from flax import linen as nn
import jax.numpy as jnp
import jax
from jax import random
from typing import Any


class KANLinear(nn.Module):
    """
    A Flax module implementing a B-spline Neural Network layer, where the basis functions are B-splines.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        degree (int): Degree of the B-splines.
    """
    in_features: int
    out_features: int
    degree: int

    def setup(self) -> None:
        assert self.degree > 0, "Degree of the B-splines must be greater than 0"
        mean, std = 0.0, 1 / (self.in_features * (self.degree + 1))
        self.coefficients = self.param(
            "coefficients",
            lambda key, shape: mean + std * random.normal(key, shape),
            (self.in_features, self.out_features, self.degree + 1)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(x)

        knots = jnp.linspace(-1, 1, self.degree + self.in_features + 1)

        b_spline_values = jnp.array([
            self.bspline_basis(x[:, i], self.degree, knots) for i in range(self.in_features)
        ])

        b_spline_values = b_spline_values.transpose((1, 0, 2))

        output = jnp.einsum('bid,ijd->bj', b_spline_values, self.coefficients)
        return output
    
    def bspline_basis(self, x: jnp.ndarray, degree: int, knots: jnp.ndarray) -> jnp.ndarray:

        def cox_de_boor(x, k, i, t):
            if k == 0:
                return jnp.where((t[i] <= x) & (x < t[i + 1]), 1.0, 0.0)
            else:
                denom1 = t[i + k] - t[i]
                denom2 = t[i + k + 1] - t[i + 1]
                
                term1 = jnp.where(denom1 != 0, (x - t[i]) / denom1, 0.0) * cox_de_boor(x, k - 1, i, t)
                term2 = jnp.where(denom2 != 0, (t[i + k + 1] - x) / denom2, 0.0) * cox_de_boor(x, k - 1, i + 1, t)
                
                return term1 + term2

        n_basis = len(knots) - degree - 1
        basis = jnp.array([cox_de_boor(x, degree, i, knots) for i in range(n_basis)])
        return jnp.transpose(basis)


class ChebyKANLinear(nn.Module):
    """
    A Flax module implementing a Chebyshev Neural Network layer, where the basis functions are Chebyshev polynomials.

    Inspired by https://github.com/CG80499/KAN-GPT-2/blob/master/chebykan_layer.py

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        degree (int): Degree of the Chebyshev polynomials.
    """
    in_features: int
    out_features: int
    degree: int

    def setup(self) -> None:
        mean, std = 0.0, 1 / (self.in_features * (self.degree + 1))
        self.coefficients = self.param(
            "coefficients",
            lambda key, shape: mean + std * random.normal(key, shape),
            (self.in_features, self.out_features, self.degree + 1)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        x = jnp.tanh(x)

        cheby_values = jnp.ones((x.shape[0], self.in_features, self.degree + 1))
        cheby_values = cheby_values.at[:, :, 1].set(x)

        for i in range(2, self.degree + 1):
            next_value = 2 * x * cheby_values[:, :, i - 1] - cheby_values[:, :, i - 2]
            cheby_values = cheby_values.at[:, :, i].set(next_value)

        output = jnp.einsum('bid,ijd->bj', cheby_values, self.coefficients)
        return output


class LegendreKANLinear(nn.Module):
    """
    A Flax module implementing a Legendre Neural Network layer, where the basis functions are Legendre polynomials.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        degree (int): Degree of the Legendre polynomials.
    """
    in_features: int
    out_features: int
    degree: int

    def setup(self) -> None:
        assert self.degree > 0, "Degree of the Legendre polynomials must be greater than 0"
        mean, std = 0.0, 1 / (self.in_features * (self.degree + 1))
        self.coefficients = self.param(
            "coefficients",
            lambda key, shape: mean + std * random.normal(key, shape),
            (self.in_features, self.out_features, self.degree + 1)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(x)

        legendre_values = jnp.ones((x.shape[0], self.in_features, self.degree + 1))
        legendre_values = legendre_values.at[:, :, 1].set(x)

        for i in range(2, self.degree + 1):
            next_value = ((2 * i - 1) * x * legendre_values[:, :, i - 1] - (i - 1) * legendre_values[:, :, i - 2]) / i
            legendre_values = legendre_values.at[:, :, i].set(next_value)

        output = jnp.einsum('bid,ijd->bj', legendre_values, self.coefficients)
        return output


class MonomialKANLinear(nn.Module):
    """
    A Flax module implementing a Monomial Neural Network layer, where the basis functions are monomials.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        degree (int): Degree of the monomial basis functions.
    """
    in_features: int
    out_features: int
    degree: int

    def setup(self) -> None:
        assert self.degree > 0, "Degree of the monomial basis functions must be greater than 0"
        mean, std = 0.0, 1 / (self.in_features * (self.degree + 1))
        self.coefficients = self.param(
            "coefficients",
            lambda key, shape: mean + std * random.normal(key, shape),
            (self.in_features, self.out_features, self.degree + 1)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(x)
        monomial_values = jnp.ones((x.shape[0], self.in_features, self.degree + 1))

        for i in range(1, self.degree + 1):
            monomial_values = monomial_values.at[:, :, i].set(x ** i)

        output = jnp.einsum('bid,ijd->bj', monomial_values, self.coefficients)
        return output


class FourierKANLinear(nn.Module):
    """
    A Flax module implementing a Fourier Neural Network layer, where the basis functions are sine and cosine functions.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        degree (int): Degree of the Fourier series (i.e., number of harmonics).
    """
    in_features: int
    out_features: int
    degree: int

    def setup(self) -> None:
        assert self.degree > 0, "Degree of the Fourier series must be greater than 0"
        mean, std = 0.0, 1 / (self.in_features * (2 * self.degree + 1))
        self.sine_coefficients = self.param(
            "sine_coefficients",
            lambda key, shape: mean + std * random.normal(key, shape),
            (self.in_features, self.out_features, self.degree)
        )
        self.cosine_coefficients = self.param(
            "cosine_coefficients",
            lambda key, shape: mean + std * random.normal(key, shape),
            (self.in_features, self.out_features, self.degree)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(x)
        sine_values = jnp.sin(jnp.pi * jnp.arange(1, self.degree + 1) * x[..., None])
        cosine_values = jnp.cos(jnp.pi * jnp.arange(1, self.degree + 1) * x[..., None])

        output = (jnp.einsum('bid,ijd->bj', sine_values, self.sine_coefficients) +
                  jnp.einsum('bid,ijd->bj', cosine_values, self.cosine_coefficients))
        return output


class HermiteKANLinear(nn.Module):
    """
    A Flax module implementing a Hermite Neural Network layer, where the basis functions are Hermite polynomials.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        degree (int): Degree of the Hermite polynomials.
    """
    in_features: int
    out_features: int
    degree: int

    def setup(self) -> None:
        assert self.degree > 0, "Degree of the Hermite polynomials must be greater than 0"
        mean, std = 0.0, 1 / (self.in_features * (self.degree + 1))
        self.coefficients = self.param(
            "coefficients",
            lambda key, shape: mean + std * random.normal(key, shape),
            (self.in_features, self.out_features, self.degree + 1)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(x)

        hermite_values = jnp.ones((x.shape[0], self.in_features, self.degree + 1))

        if self.degree >= 1:
            hermite_values = hermite_values.at[:, :, 1].set(2 * x)
        for i in range(2, self.degree + 1):
            hermite_values = hermite_values.at[:, :, i].set(
                2 * x * hermite_values[:, :, i - 1] - 2 * (i - 1) * hermite_values[:, :, i - 2]
            )

        output = jnp.einsum('bid,ijd->bj', hermite_values, self.coefficients)
        return output
    