"""
Encodes KL divergences between normal distributions
"""

@register_kl(MultivariateNormalFactor)
def _kl_multivariate_normal_factor_multivariate_normal_factor_sum(
        MultivariateNormalFactor, MultivariateNormalFactorSum
):
    pass
