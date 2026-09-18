"""Trusted handwritten Hecate examples; tracing only, no execution here.

The Linear uses one input ciphertext with a repeating four-slot period and
two output ciphertexts. Each output row is a true four-element dot product.
Select slot zero of each output. The MLP keeps each hidden scalar broadcast in
its own ciphertext; it never decrypts or repacks hidden values using plaintext.
"""
import argparse
from pathlib import Path
import numpy as np
import hecate as hc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("add", "mul_plain", "linear4x2", "rotate1", "rotate2",
                                        "square", "quartic", "mlp4x4x2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.output / "arrays.npz", allow_pickle=False) as arrays:
        weight = arrays["weight"].copy()
        bias = arrays["bias"].copy()
        if args.case == "mlp4x4x2":
            hidden_weight = arrays["mlp_hidden_weight"].copy()
            hidden_bias = arrays["mlp_hidden_bias"].copy()

    if args.case == "add":
        @hc.func("c")
        def golden(x):
            return x + x
    elif args.case == "mul_plain":
        @hc.func("c")
        def golden(x):
            return x * weight
    elif args.case.startswith("rotate"):
        @hc.func("c")
        def golden(x):
            return x.rotate(int(args.case[-1]))
    elif args.case == "square":
        @hc.func("c")
        def golden(x):
            return x * x
    elif args.case == "quartic":
        @hc.func("c")
        def golden(x):
            squared = x * x
            return squared * squared
    elif args.case == "linear4x2":
        @hc.func("c")
        def golden(x):
            outputs = []
            for row in range(2):
                product = x * weight[row]
                pairs = product + product.rotate(1)
                total = pairs + pairs.rotate(2)
                outputs.append(total + float(bias[row]))
            return outputs
    else:
        @hc.func("c")
        def golden(x):
            hidden = []
            for row in range(4):
                product = x * hidden_weight[row]
                pairs = product + product.rotate(1)
                total = pairs + pairs.rotate(2)
                activation = total + float(hidden_bias[row])
                hidden.append(activation * activation)
            outputs = []
            for row in range(2):
                total = hidden[0] * float(weight[row, 0])
                for column in range(1, 4):
                    total = total + hidden[column] * float(weight[row, column])
                outputs.append(total + float(bias[row]))
            return outputs
    hc.save(str(args.output), str(args.output))


if __name__ == "__main__":
    main()
