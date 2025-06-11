from torch import isnan, isinf


def validate_probs(probs):
    if isnan(probs).any():
        print("NaN detected in probs!")
    if isinf(probs).any():
        print("Inf detected in probs!")
    if (probs < 0).any():
        print("Negative values in probs!")