import argparse
import os
import sys
import h5py
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model as model_lib


def convert_weights_to_hdf5(weights_path: str, output_path: str, input_size=(128, 128, 3)):
    print("Instantiating base model architecture...")
    model = model_lib.unet(input_size=input_size)

    print(f"Inspecting and reading weights from: {weights_path}")
    with h5py.File(weights_path, "r") as f:
        weight_datasets = {}

        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                weight_datasets[name] = np.array(node)

        f.visititems(visitor)

        transferred = 0
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if not layer_weights:
                continue

            # Match keys in the H5 file containing this layer's name
            matching_keys = sorted([k for k in weight_datasets if layer.name in k])

            # Assign weights if key count and shapes align
            if len(matching_keys) == len(layer_weights):
                new_weights = [weight_datasets[k] for k in matching_keys]
                if all(w1.shape == w2.shape for w1, w2 in zip(layer_weights, new_weights)):
                    layer.set_weights(new_weights)
                    transferred += 1
                elif all(w1.shape == w2.shape for w1, w2 in zip(layer_weights, reversed(new_weights))):
                    layer.set_weights(list(reversed(new_weights)))
                    transferred += 1

    print(f"Transferred weights for {transferred} parameterized layers.")
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving compiled model to: {output_path}")
    model.save(output_path, save_format="h5")
    print(f"Conversion successful! Output saved to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras .weights.h5 files into standalone .hdf5 models.")
    parser.add_argument(
        "--weights_path",
        type=str,
        default="../trained_models/impurities_unet/model.weights.h5",
        help="Path to the source .weights.h5 file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path for the output .hdf5 file. Defaults to replacing extension with .hdf5 in the same folder.",
    )
    args = parser.parse_args()

    # Automatically derive output filename if not explicitly provided
    if not args.output_path:
        base_dir = os.path.dirname(args.weights_path)
        base_name = os.path.basename(args.weights_path).replace(".weights.h5", "").replace(".h5", "")
        args.output_path = os.path.join(base_dir, f"{base_name}.hdf5")

    convert_weights_to_hdf5(args.weights_path, args.output_path)