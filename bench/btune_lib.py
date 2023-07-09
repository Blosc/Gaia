#######################################################################
# Copyright (C) Blosc Development team <blosc@blosc.org>
# All rights reserved.
#######################################################################
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Optional requirements
try:
    import keras
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def get_df_chunk(paths, name):
    """
    Get a pandas dataframe with the measurements for the `name` category
    in the different `paths`.
    @param paths: list
        Directory paths containing measurements files
        (each directory corresponds to a different dataset).
    @param name: str
        Category name.
    @return: pandas.DataFrame
        Data frame with the `cratio`, `speed`,
        `special_vals` (0 if it is not a special value) and
        `nchunk` as columns.
        It contains one row for each stream.
    """
    df_res = pd.DataFrame()
    nchunk_start = 0
    for i in range(len(paths)):
        filename = name + '.csv.gz'
        df = pd.read_csv(paths[i]/filename, delimiter=',')
        df.columns = df.columns.str.strip()
        if i > 0:
            df.nchunk = df.nchunk + nchunk_start
        df_res = pd.concat([df_res, df], axis=0)
        nchunk_start = df_res.nchunk.max() + 1

    df_res = df_res.reset_index(drop=True)
    df_res = df_res.astype('float32')

    return df_res


def drop_special_vals(df, special_vals):
    """
    Drop all row from a pandas.DataFrame corresponding to a
    special value measurent.
    @param df: pandas.DataFrame
        The data frame to filter.
    @param special_vals: pandas.Series
        The `special_vals` value for each row in the `df`.
    @return: pandas.DataFrame
        The filtered data frame.
    """
    df.special_vals = special_vals          # Replace special values column
    df = df[df.special_vals.isin([0])]      # Remove rows with special values
    df = df.drop(['special_vals'], axis=1)  # Drop the special values column
    df = df.reset_index(drop=True)          # Reindex the dataframe from zero
    return df


def del_nchunk(data_codecs, nchunks):
    """
    Remove measurements nchunks rows from dataframes.
    @param data_codecs: list[pd.DataFrame]
        The list of measurements data frames for each category.
    @param nchunks: list[int]
        The nchunks to remove.
    @return: list[pd.DataFrame]
        The new list of data frames without the special values chunks.

    """
    for i in range(len(data_codecs)):
        data_codecs[i] = data_codecs[i].drop(nchunks)
        data_codecs[i] = data_codecs[i].reset_index(drop=True)
    return data_codecs


def filter_entropy(probe_dfs, codecs_dfs):
    """
    Compute mean for each chunk and delete data corresponding
    to special values chunks.
    @param probe_dfs: list[pd.DataFrame]
        The data frames with the entropy-probe for each stream.
    @param codecs_dfs: list[pd.DataFrame]
        The real data measurements.
    @return: list[pd.DataFrame]
        The new list of entropy-probe data frames by chunk.
    """
    mean_probe_dfs = [pd.DataFrame()] * len(probe_dfs)
    special_chunks = []

    for i in range(len(probe_dfs)):
        cratio_mean = []
        speed_mean = []
        for nchunk in range(len(codecs_dfs[0])):
            df = probe_dfs[i][probe_dfs[i].nchunk.isin([nchunk])]
            if df.empty:
                special_chunks.append(nchunk)
                cratio_mean.append(0)
                speed_mean.append(0)
            else:
                cratio_mean.append(df.cratio.mean())
                speed_mean.append(df.speed.mean())

        mean_probe_dfs[i]["cratio"] = cratio_mean
        mean_probe_dfs[i]["speed"] = speed_mean
        mean_probe_dfs[i] = mean_probe_dfs[i].copy()

    del_nchunk(codecs_dfs, special_chunks)
    del_nchunk(mean_probe_dfs, special_chunks)

    return mean_probe_dfs


def load_data_chunk(
    root: str = '../data/',
    probes: list[str] = ['entropy-nofilter-nosplit'],
    categories: list[str] = None,
    files: list[str] = ['temp', 'precip', 'wind', 'flux', 'pressure', 'snow'],
):
    """
    Get filtered measurements for each category in `categories` and probe in `probes`.
    @param root: str
        The directory where the measurements for each
        `temp`, `wind` and `precip` dataset are.
    @param probes: list[str]
        Which entropy-probe data to use (`entropy-bitshuffle`, `entropy-shuffle`,
        `entropy-nofilter` or `entropy-shuffle-bytedelta`).
    @param categories: list[str]
        Which categories to get.
    @return: list[pandas.DataFrame], list[pandas.DataFrame], pandas.DataFrame
        * The first output contains the data frames for each entropy-probe in `probes`.
        * The second output contains the data frames for each category in `categories`.
    """
    # Verify input parameters
    assert type(categories) is list and len(categories) > 1

    # Paths to the input files
    root = Path(root)

    # Load probe dataframes
    paths = [root / file for file in files]
    probe_dfs = [get_df_chunk(paths, name) for name in probes]

    # Load codec dataframes
    paths = [root / file for file in files]
    codec_dfs = [get_df_chunk(paths, name) for name in categories]
    codec_dfs = [df.drop(['nchunk'], axis=1) for df in codec_dfs]

    # Verify special values match
    # This does not make sense because the special values are not present in chunks anymore
    # dataframes = probe_dfs + codec_dfs
    dataframes = probe_dfs
    special_vals = pd.concat([df[['special_vals']] for df in dataframes], axis=1)
    special_vals = special_vals.any(axis=1)
    special_vals = special_vals.astype('int8')

    # Drop special values from all dataframes
    probe_dfs = [drop_special_vals(df, special_vals) for df in probe_dfs]
    # This does not make sense because the special values are not present in chunks anymore
    # codec_dfs = [drop_special_vals(df, special_vals) for df in codec_dfs]
    probe_dfs = filter_entropy(probe_dfs, codec_dfs)

    return probe_dfs, codec_dfs


def normalize(df):
    """
    Modify data to be comparable with other measurements (like speed and cratio).
    @param df: pandas.DataFrame
        The data to normalize.
    @return: pandas.DataFrame
        An equivalent data frame with normalized values.
    """
    array = df.to_numpy()
    array -= array.mean()
    array /= array.std(ddof=1)  # Calculate std like pandas (with ddof=1)
    
    df = pd.DataFrame(array)
    df = df.astype('float32')
    
    return df


def normalize_train_test(train, test, meta_path=None, categories=None):
    """
    Normalize 2 dataframes using the `train` statistics for normalizing the
    `test` data. The `balance` values are not normalized.
    @param train: pandas.DataFrame
        A data frame with 3 possible columns names: `cratio`, `speed` and `balance`.
    @param test: pandas.DataFrame
        A data frame with 3 possible columns names: `cratio`, `speed` and `balance`.
    @param meta_path: str
        If it is not None, the path to write the metadata for normalization.
    @param categories: list[str]
        If it is not None, list of category names in the format <codec>-<filter>-<split>
    @return: pandas.DataFrame, pandas.DataFrame
        The equivalent `train` and `test` dataframes normalized as described.
    """
    # Use train data to normalize test data
    train_cratio = train.cratio.to_numpy()

    mean_cratio = train_cratio.mean()
    std_cratio = train_cratio.std(ddof=1)
    train_cratio -= mean_cratio
    train_cratio /= std_cratio
    test_cratio = test.cratio.to_numpy()
    test_cratio -= mean_cratio
    test_cratio /= std_cratio

    train_speed = train.speed.to_numpy()
    mean_speed = train_speed.mean()
    std_speed = train_speed.std(ddof=1)
    train_speed -= mean_speed
    train_speed /= std_speed
    test_speed = test.speed.to_numpy()
    test_speed -= mean_speed
    test_speed /= std_speed

    if isinstance(train.cratio, pd.DataFrame):
        cratio_names = train.cratio.columns
        train_cratio = pd.DataFrame(train_cratio, columns=cratio_names)
        test_cratio = pd.DataFrame(test_cratio, columns=cratio_names)
        speed_names = train.speed.columns
        train_speed = pd.DataFrame(train_speed, columns=speed_names)
        test_speed = pd.DataFrame(test_speed, columns=speed_names)
    else:
        train_cratio = pd.DataFrame(train_cratio)
        test_cratio = pd.DataFrame(test_cratio)
        train_speed = pd.DataFrame(train_speed)
        test_speed = pd.DataFrame(test_speed)

    # Save metadata to file
    if meta_path is not None:
        metadata = {
            "cratio": {
                "mean": float(mean_cratio),
                "std": float(std_cratio),
            },
            "speed": {
                "mean": float(mean_speed),
                "std": float(std_speed),
            },
        }

        if categories is not None:
            metadata['categories'] = []
            codecs = ['blosclz', 'lz4', 'lz4hc', None, 'zlib', 'zstd']
            filters = {
                'nofilter': 0,
                'shuffle': 1,
                'bitshuffle': 2,
                'shuffle-bytedelta': 34,
            }
            BLOSC_ALWAYS_SPLIT = 1 # blosc2.SplitMode.ALWAYS_SPLIT
            BLOSC_NEVER_SPLIT = 2  # blosc2.SplitMode.NEVER_SPLIT
            for category in categories:
                rest, split, clevel = category.rsplit('-', 2)
                codec, filter = rest.split('-', 1)
                codec = codecs.index(codec)
                filter = filters[filter]
                split = {'split': BLOSC_ALWAYS_SPLIT, 'nosplit': BLOSC_NEVER_SPLIT}[split]
                metadata['categories'].append([codec, filter, int(clevel), split])

        # Serialize to JSON and write to file
        json_object = json.dumps(metadata)
        with open(meta_path, "w") as outfile:
            outfile.write(json_object)
    train_balance = pd.DataFrame(train.balance).reset_index(drop=True)
    test_balance = pd.DataFrame(test.balance).reset_index(drop=True)

    train_df = pd.concat([train_cratio, train_speed, train_balance], axis=1)
    train_df = train_df.astype('float32')
    test_df = pd.concat([test_cratio, test_speed, test_balance], axis=1)
    test_df = test_df.astype('float32')
    
    return (train_df, test_df)


def normalize_test(test, meta_dict):
    """
    Normalize `test` using the statistics in `meta_dict`.
    @param test: pandas.DataFrame
        A data frame with 3 possible columns names: `cratio`, `speed` and balance.
    @param meta_dict: dict
        The cratio and speed statistics to use in order to normalize.
    @return: pandas.DataFrame
        The normalized dataframe.
    """
    cratio = meta_dict["cratio"]
    test_cratio = test.cratio.to_numpy()
    test_cratio -= cratio["mean"]
    test_cratio /= cratio["std"]

    speed = meta_dict["speed"]
    test_speed = test.speed.to_numpy()
    test_speed -= speed["mean"]
    test_speed /= speed["std"]

    if isinstance(test.cratio, pd.DataFrame):
        cratio_names = test.cratio.columns
        test_cratio = pd.DataFrame(test_cratio, columns=cratio_names)
        speed_names = test.speed.columns
        test_speed = pd.DataFrame(test_speed, columns=speed_names)
    else:
        test_cratio = pd.DataFrame(test_cratio)
        test_speed = pd.DataFrame(test_speed)

    test_balance = pd.DataFrame(test.balance).reset_index(drop=True)
    
    df = pd.concat([test_cratio, test_speed, test_balance], axis=1)
    df = df.astype('float32')
    
    return df


# We need this function even for chunks inference
def get_labels_row(dataframes, balance=0.6, cspeed=True, errors=[0.05, 0.1, 0.2]):
    """
    The input is a list of dataframes, one for every codec. Each dataframe contains
    a list of data points (compression ratio and speed).
    If cspeed is true the score will be computed with the compression
    speed otherwise it will be computed with the decompression speed.
    The `errors` is the list of % range of score error (between 0 and 1) we would like
    to admit when computing the prediction score.

    The output is a dataframe with 3 columns: the best codec, the 2nd best,
    the 3rd best and one column per each error with the list of codecs
    inside the error range for the data sample,
    expressed as the index in the input list for each chunk (row).
    """
    # Normalize compression ratios and compression speeds, so they are comparable
    ratios = pd.concat([df[['cratio']] for df in dataframes], axis=1)
    ratios = normalize(ratios)
    if cspeed:
        speeds = pd.concat([df[['cspeed']] for df in dataframes], axis=1)
        speeds = normalize(speeds)
    else:
        speeds = pd.concat([df[['dspeed']] for df in dataframes], axis=1)
        speeds = normalize(speeds)

    # Reduce compression ratios and speed to a score value, which says how good the codec
    # is for the given data sample
    scores = pd.DataFrame()
    for i in range(ratios.shape[1]):
        scores.insert(i, i, balance * ratios.iloc[:, i] + (1 - balance) * speeds.iloc[:, i])

    # Get ordered list of the 3 best categories for each chunk
    l = [0] * len(scores)
    for i in range(len(scores)):
        df = scores.iloc[i, :]
        # Get 3 bests categories without looking the difference
        l[i] = df.sort_values(ascending=False).iloc[:3].index.values
    bests = pd.DataFrame(l)

    # Get list of the best categories for each  inside error ranges
    errors_lists = [[0] * len(scores)] * len(errors)
    for i in range(len(scores)):
        df = scores.iloc[i, :]
        df_sorted = df.sort_values(ascending=False)
        min = df_sorted.iloc[-1]
        max = df_sorted.iloc[0]
        var = max - min
        for j in range(len(errors)):
            inf_limit = max - var * errors[j]
            # Get bests categories with an error % of `error[j]`
            errors_lists[j][i] = df_sorted[df_sorted >= inf_limit].index.values
    for j in range(len(errors)):
        bests[str(errors[j])] = [[]] * len(bests)
        bests[str(errors[j])] = errors_lists[j]

    return bests


def get_labels_balances(codecs_dfs, balances_array, cspeed=True, errors=[0.05, 0.1, 0.2]):
    """

    @param codecs_dfs: list[pandas.DataFrame]
        The list with the real measurements dataframes.
    @param balances_array: np.array(float)
        The list of balances to get the bests categories.
    @param cspeed: bool
        @see get_labels_row
    @param errors: list[float]
        Values from 0.01 to 1. It indicates which error ranges to take into account
        to decide if one category is not far from the best one.
    @return: pandas.DataFrame
        A dataframe with 3 columns: best, 2nd best, 3rd best categories
        and one additional column for each range in `errors` list with the
        categories inside the range.
    """
    bests_list = [pd.DataFrame()] * len(balances_array)
    for i in range(len(balances_array)):
        bests_list[i] = get_labels_row(codecs_dfs, balances_array[i], cspeed, errors)

    bests = pd.concat(bests_list, axis=0)
    bests = bests.reset_index(drop=True)
    errors_str = [str(e) for e in errors]
    bests.columns = ["best", "2nd best", "3rd best"] + errors_str

    return bests


def get_nn_input(probes_dfs, balances_array):
    """
    Get the neural network input data. This must still be normalized.
    @param probes_dfs: list[pandas.DataFrame]
        The dataframes with the entropy data.
    @param balances_array: np.array(float)
        Balances to train for.
    @return: pandas.DataFrame
        The entropy data to use plus the balance for each row.
    """
    balances = pd.Series(balances_array).repeat(len(probes_dfs[0]))
    balances = balances.reset_index(drop=True)
    balances = balances.rename("balance")

    probes = pd.concat([df for df in probes_dfs], axis=1)
    list_probes = [probes] * len(balances_array)
    nn_input = pd.concat(list_probes, axis=0)
    nn_input = nn_input.reset_index(drop=True)
    nn_input = pd.concat([nn_input, balances], axis=1)

    return nn_input


def split_data(entropy, bests, ncategories):
    """
    Split data between train and test.
    @param entropy: pandas.DataFrame
        It contains the input data for the neural network
        (two columns for each probe: `cratio` and `speed`; plus the `balance`).
    @param bests: pandas.DataFrame
        Data frame with the 3 best categories for each chunk (`best`, `2nd best` and `3rd best`) plus one column
        with the list of good categories for each error range tested.
    @param ncategories: int
        The number of possible categories.
    @return: tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame),
    tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        * The first tuple contains the train data, train labels
        and train bests categories for each chunk (like the `bests` parameter).
        * The first tuple contains the test data, test labels and test
        bests categories for each chunk (like the `bests parameter).
    """

    # Verify data
    best = bests['best']
    categories = best.unique()
    assert len(categories) > 1, 'not enough categories'

    for cat in categories:
        n = len(entropy[best == cat])
        if n < 2000:
            print(f'WARNING only {n} samples for {cat} category')

    # Split 90% samples for training and 10% for testing
    data = pd.concat([entropy, bests], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the rows
    n = int(len(data) * .9)
    train = data.head(n)
    test = data.iloc[n:, :]

    # Separate labels from data points
    train_labels = train['best']
    train_data = train[["cratio", "speed", "balance"]]
    train_three_best = train.drop(['cratio', 'speed', 'balance'], axis=1)
    test_labels = test['best']
    test_data = test[["cratio", "speed", "balance"]]
    test_three_best = test.drop(['cratio', 'speed', 'balance'], axis=1)

    # Labels, change dataframe from 1 column with 0-N values to N columns with 0/1 values
    train_labels = to_categorical(train_labels, num_classes=ncategories)
    test_labels = to_categorical(test_labels, num_classes=ncategories)

    return (train_data, train_labels, train_three_best), (test_data, test_labels, test_three_best)


def test_prediction(model, data, bests):
    """
    Test model.
    @param model: keras.models.Model
        The neural network model to test.
    @param data:
        The input data to pass to the model.
    @param bests: pandas.DataFrame
        The bests labels for each chunk (1st, 2nd, 3rd and a column for each error range).
    @return: pandas.DataFrame
        The predicted categories
    """
    # Get the prediction: a np array with, for each sample, the probability for every
    # codec to be the good one
    prediction = model.predict(data)

    # Array with the predicted for every sample
    pred_categories = np.argmax(prediction, axis=1)

    ranges = bests.columns.values.tolist()[3:]
    scores_ranges = [0] * len(ranges)
    print("Scores inside ranges (error range, score): ", end="")
    for j in range(len(ranges)):
        for i in range(len(pred_categories)):
            if pred_categories[i] in bests[ranges[j]].iloc[i]:
                scores_ranges[j] += 1
        print(tuple([float(ranges[j]), round(scores_ranges[j]/len(pred_categories), 2)]), end=" ")
    print()

    scores = [0] * 3
    for i in range(len(scores)):
        # Produce an array with zeros when the prediction is correct
        aux = pred_categories - bests.iloc[:, i].to_numpy()
        scores[i] = np.count_nonzero(aux == 0) / len(aux)
    fails = 1 - np.array(scores).sum()
    print("1st to 3rd best predictions score ", scores)
    print("Fails: ", fails)

    print("")
    # Cross tabulation
    table = pd.crosstab(
        pred_categories,
        bests['best'],
        rownames=['pred'],
        colnames=['true'],
    )
    # Print
    print(table)
    print("")

    print("Table for 2nd true best category")
    table = pd.crosstab(
        pred_categories,
        bests['2nd best'],
        rownames=['pred'],
        colnames=['2nd best'],
    )
    print(table)

    print("")
    print("Table for 3rd true best category")
    table = pd.crosstab(
        pred_categories,
        bests["3rd best"],
        rownames=['pred'],
        colnames=['3rd best'],
    )
    print(table)

    pred_categories = pd.DataFrame(pred_categories, columns=['best'])

    return pred_categories


def get_model(n_categories):
    """
    Build model.
    @param n_categories: int
        Number of categories to predict.
    @return: keras.models.Model
        The neural network model to train.
    """
    model = keras.models.Sequential([
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(n_categories, activation='softmax'),
    ])

    # sgd, rmsprop, adam, adagrad, etc.
    optimizer = 'adam'

    # CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy,
    # MeanSquaredError, KLDivergence, CosineSimilarity, etc.
    loss = 'categorical_crossentropy'

    # CategoricalAccuracy, SparseCategoricalAccuracy, BinaryAccuracy, AUC, Precision,
    # Recall, etc.
    metrics = ['acc']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def print_legend(probes, categories):
    """
    Print the legend corresponding to the different categories' id.
    @param probes: list[str]
    @param categories: list[str]
    @return: None
    """
    print('Categories:')
    for idx, name in enumerate(categories):
        print(f'- {idx}: {name}')

    print('Probes:')
    for probe in probes:
        print(f'- {probe}')


def plot_history(history):
    """
    Plot performance of model during training.
    @param history: `History` object
    @return: None
    """
    _, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    ax[0].plot(history.history['acc'], 'r')
    ax[0].plot(history.history['val_acc'], 'g')
    ax[0].set_xlabel("Num of Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Training Accuracy vs Validation Accuracy")
    ax[0].legend(['train', 'validation'])
    ax[1].plot(history.history['loss'], 'r')
    ax[1].plot(history.history['val_loss'], 'g')
    ax[1].set_xlabel("Num of Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Training Loss vs Validation Loss")
    ax[1].legend(['train', 'validation'])
    plt.tight_layout()
    plt.show()


def to_categorical(y, num_classes=None, dtype="float32"):
    # It seems like tflite does not support float64, so `dtype`
    # should always be float32
    # This function is a copy of keras.np_utils.to_categorical
    # It's here so, we can run the inference only with tflite-runtime
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def most_predicted(preds, balances, categories, codecs_dfs):
    """
    Print a summary of the 3 most predicted categories
    and compute the mean cratio, cspeed and dspeed for the
    most predicted category for each balance.
    @param preds: pandas.DataFrame
        Predictions done by the neural network.
    @param balances: pandas.Series
        The balance corresponding to each prediction.
    @param categories: list[str]
        List with each category name in the index corresponding to the id used.
    @param codecs_dfs: list[pandas.Series]
        List of real measurements for each category.
    @return: pandas.DataFrame
        Resulting dataframe.
    """
    table = []

    balance_values = balances.unique()
    balance_values.sort()
    for balance in balance_values:
        df = preds[balances == balance]
        row = df.best.value_counts().iloc[:3].index
        real_df = codecs_dfs[row[0]]

        row = [categories[pred] for pred in row]
        row += ['-'] * (3 - len(row))

        # Get most predicted category mean cratio, cspeed and dspeed for each balance
        row.append(round(real_df['cratio'].mean(), 2))
        row.append(round(real_df['cspeed'].mean() / 10**9, 2))
        row.append(round(real_df['dspeed'].mean() / 10**9, 2))

        table.append(np.array(row))

    table = pd.DataFrame(table, columns=["1st most predicted", "2nd most predicted", "3rd most predicted",
                                         "Mean cratio", "Mean cspeed (GB/s)", "Mean dspeed (GB/s)"])
    table.index = balance_values

    return table


def scores_summary(preds, bests, balances):
    """
    Print a score summary for each balance.
    @param preds: pandas.DataFrame
        The predictions done by the nn.
    @param bests: pandas.DataFrame
        It contains best, 2nd best, 3rd best categories plus the list of
        good categories inside an error range (one column per error) for each chunk.
    @param balances: pandas.Series
        The balance corresponding to each row from the other params.
    @return: None
    """
    print("Get score separately for each balance")
    ranges = bests.columns.values.tolist()[3:]

    preds.columns = ["pred"]
    aux_df = pd.concat([preds, bests, balances], axis=1)
    aux_df = aux_df.reset_index(drop=True)
    balance_values = balances.unique()
    balance_values.sort()

    table = []

    for i in range(len(balance_values)):
        df = aux_df[aux_df.balance.isin([balance_values[i]])]
        predictions = df["pred"].to_numpy()

        row = []
        # Score best category
        true_cat = df["best"].to_numpy()
        arr = predictions - true_cat
        row.append(np.count_nonzero(arr == 0) / len(arr))
        # Score 2nd best category
        true_cat = df["2nd best"].to_numpy()
        arr = predictions - true_cat
        row.append(np.count_nonzero(arr == 0) / len(arr))
        # Score 3rd best category
        true_cat = df["3rd best"].to_numpy()
        arr = predictions - true_cat
        row.append(np.count_nonzero(arr == 0) / len(arr))
        # Fails
        scores = row[0] + row[1] + row[2]
        row.append(1 - scores)

        # Scores for different error ranges
        scores_ranges = [0] * len(ranges)
        for j in range(len(ranges)):
            for nchunk in range(len(predictions)):
                if predictions[nchunk] in df[ranges[j]].iloc[nchunk]:
                    scores_ranges[j] += 1
            row.append(scores_ranges[j]/len(predictions))

        table.append(np.array(row))

    res = pd.DataFrame(table)
    res.columns = ["Best", "2nd best", "3rd best", "Fails"] + ranges
    res.index = balance_values
    print(res)
