#######################################################################
# Copyright (C) Blosc Development team <blosc@blosc.org>
# All rights reserved.
#######################################################################
import numpy as np

import btune_lib as bt
import pandas as pd
import tensorflow as tf
import sys

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in ["c", "d"]:
        cspeed = sys.argv[1] == "c"
    else:
        print("Usage example: python training_chunk.py c[cspeed]/d[dspeed]")
        raise Exception("You can only specify whether to use compression speed (c) or decompression speed (d)")

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

    # Load data as dataframes
    probes_dfs, codecs_dfs = bt.load_data_chunk(root='../arange_random_data/', files=["b2frame"],  probes=probes, categories=categories)
    balances_array = np.linspace(0, 1, 11, dtype="float32")
    print("Balances = ", balances_array)
    # Bests categories for every data sample
    bests = bt.get_labels_balances(codecs_dfs, balances_array, cspeed=cspeed)

    # Build input data
    nn_input = bt.get_nn_input(probes_dfs, balances_array)

    # Split train/test data
    (train_data, train_labels, train_bests), \
        (test_data, test_labels, test_bests) = bt.split_data(nn_input, bests, len(categories))

    # Normalize train test data sets
    suffix = 'comp' if cspeed else 'decomp'
    meta_path = f'model_{suffix}.json'
    train_data, test_data = bt.normalize_train_test(train_data, test_data, meta_path, categories)

    # Train model
    print()
    print('# Model fit')
    model = bt.get_model(len(categories))
    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        validation_split=0.1,
    )

    # Save model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(f'model_{suffix}.tflite', 'wb').write(tflite_model)

    # Plot
    bt.plot_history(history)

    # Test with train data
    print()
    print('# Prediction with the TRAIN data')
    train_preds = bt.test_prediction(model, train_data, train_bests)

    # Test with test data
    print()
    print('# Prediction with the TEST data')
    test_preds = bt.test_prediction(model, test_data, test_bests)

    # Print most predicted categories for each balance
    print()
    balances = pd.concat([train_data.balance, test_data.balance], axis=0)
    balances = balances.reset_index(drop=True)
    preds = pd.concat([train_preds, test_preds], axis=0).reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    table = bt.most_predicted(preds, balances, categories, codecs_dfs)
    print(table)

    # Print different scores for each balance
    print()
    bests = pd.concat([train_bests, test_bests], axis=0)
    bests = bests.reset_index(drop=True)
    bt.scores_summary(preds, bests, balances)

    # Print legend (index to category name)
    print()
    print('# Legend')
    bt.print_legend(probes, categories)
