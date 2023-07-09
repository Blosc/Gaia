import btune_lib as bt
import json
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
import sys


def get_input_data(cspeed):
    probes = [
        'entropy-nofilter-nosplit',
    ]
    categories = [
        'blosclz-nofilter-nosplit-5',
        'blosclz-shuffle-nosplit-5',
        'blosclz-bitshuffle-nosplit-5',
        'blosclz-shuffle-bytedelta-nosplit-5',
        'lz4-nofilter-nosplit-5',
        'lz4-shuffle-nosplit-5',
        'lz4-bitshuffle-nosplit-5',
        'lz4-shuffle-bytedelta-nosplit-5',
        'lz4hc-nofilter-nosplit-5',
        'lz4hc-shuffle-nosplit-5',
        'lz4hc-bitshuffle-nosplit-5',
        'lz4hc-shuffle-bytedelta-nosplit-5',
        'zlib-nofilter-nosplit-5',
        'zlib-shuffle-nosplit-5',
        'zlib-bitshuffle-nosplit-5',
        'zlib-shuffle-bytedelta-nosplit-5',
        'zstd-nofilter-nosplit-1',
        'zstd-shuffle-nosplit-1',
        'zstd-bitshuffle-nosplit-1',
        'zstd-shuffle-bytedelta-nosplit-1',
        'zstd-nofilter-nosplit-3',
        'zstd-shuffle-nosplit-3',
        'zstd-bitshuffle-nosplit-3',
        'zstd-shuffle-bytedelta-nosplit-3',
        'zstd-nofilter-nosplit-6',
        'zstd-shuffle-nosplit-6',
        'zstd-bitshuffle-nosplit-6',
        'zstd-shuffle-bytedelta-nosplit-6',
        'zstd-nofilter-nosplit-9',
        'zstd-shuffle-nosplit-9',
        'zstd-bitshuffle-nosplit-9',
        'zstd-shuffle-bytedelta-nosplit-9',
    ]

    if not cspeed:
        categories += [
                        'blosclz-nofilter-split-5',
                        'blosclz-shuffle-split-5',
                        'blosclz-bitshuffle-split-5',
                        'blosclz-shuffle-bytedelta-split-5',
                        'lz4-nofilter-split-5',
                        'lz4-shuffle-split-5',
                        'lz4-bitshuffle-split-5',
                        'lz4-shuffle-bytedelta-split-5',
                        'lz4hc-nofilter-split-5',
                        'lz4hc-shuffle-split-5',
                        'lz4hc-bitshuffle-split-5',
                        'lz4hc-shuffle-bytedelta-split-5',
                        'zlib-nofilter-split-5',
                        'zlib-shuffle-split-5',
                        'zlib-bitshuffle-split-5',
                        'zlib-shuffle-bytedelta-split-5',
                        'zstd-nofilter-split-1',
                        'zstd-shuffle-split-1',
                        'zstd-bitshuffle-split-1',
                        'zstd-shuffle-bytedelta-split-1',
                        'zstd-nofilter-split-3',
                        'zstd-shuffle-split-3',
                        'zstd-bitshuffle-split-3',
                        'zstd-shuffle-bytedelta-split-3',
                        'zstd-nofilter-split-6',
                        'zstd-shuffle-split-6',
                        'zstd-bitshuffle-split-6',
                        'zstd-shuffle-bytedelta-split-6',
                        'zstd-nofilter-split-9',
                        'zstd-shuffle-split-9',
                        'zstd-bitshuffle-split-9',
                        'zstd-shuffle-bytedelta-split-9',
                    ]

    # Load data as dataframes
    probe_dfs, codecs_dfs = bt.load_data_chunk(probes=probes, categories=categories)
    balances_array = np.linspace(0, 1, 11, dtype="float32")
    bests = bt.get_labels_balances(codecs_dfs, balances_array, cspeed=cspeed)

    # Build input data
    nn_input = bt.get_nn_input(probe_dfs, balances_array)

    # Split train/test data, it may be possible that the test_data contains previously used trainig data
    (train_data, train_labels, train_bests), \
    (test_data, test_labels, test_bests) = bt.split_data(nn_input, bests, len(categories))

    # Get metadata
    # Opening JSON file
    f = open('metadata.json')
    metadata = json.load(f)
    # Closing file
    f.close()

    # Normalize test data set
    test_data = bt.normalize_test(test_data, metadata)

    return test_data, test_labels, test_bests


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in ["c", "d"]:
        cspeed = sys.argv[1] == "c"
    else:
        print("Usage example: python training_chunk.py c[cspeed]/d[dspeed]")
        raise Exception("You can only specify whether to use compression speed (c) or decompression speed (d)")

    suffix = 'comp' if cspeed else 'decomp'
    model_path = f'model_{suffix}.tflite'
    test_data, test_labels, test_three_bests = get_input_data(cspeed)

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']

    pred = []
    expected = []
    n = len(test_data)
    for i in range(n):
        tensor = test_data.iloc[i:i + 1]
        interpreter.set_tensor(input_index, tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)
        output_data = output_data[0]

        pred.append(np.argmax(output_data))
        expected.append(np.argmax(test_labels[i]))

    aux = np.array(pred) - np.array(expected)
    ok = np.count_nonzero(aux == 0)
    score = ok / len(aux)
    print(f'Score by chunks: {score}')

    # Cross tabulation
    table = pd.crosstab(
        pred,
        expected,
        rownames=['pred'],
        colnames=['true'],
    )
    print(table)
