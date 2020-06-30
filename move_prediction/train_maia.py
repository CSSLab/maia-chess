import argparse
import os
import os.path
import yaml
import sys
import glob
import gzip
import random
import multiprocessing

import tensorflow as tf

import maia_chess_backend
import maia_chess_backend.maia

SKIP = 32

#@maia_chess_backend.logged_main
def main(config_path, name, collection_name):
    output_name = os.path.join('models', collection_name, name + '.txt')

    with open(config_path) as f:
        cfg = yaml.safe_load(f.read())

    maia_chess_backend.printWithDate(yaml.dump(cfg, default_flow_style=False))

    experimental_parser = cfg['dataset'].get('experimental_v4_only_dataset', False)

    train_chunks = get_latest_chunks(cfg['dataset']['input_train'])
    test_chunks = get_latest_chunks(cfg['dataset']['input_test'])

    shuffle_size = cfg['training']['shuffle_size']
    total_batch_size = cfg['training']['batch_size']
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    if total_batch_size % batch_splits != 0:
        raise ValueError('num_batch_splits must divide batch_size evenly')
    split_batch_size = total_batch_size // batch_splits
    # Load data with split batch size, which will be combined to the total batch size in tfprocess.
    maia_chess_backend.maia.ChunkParser.BATCH_SIZE = split_batch_size

    root_dir = os.path.join('models', collection_name, name)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    tfprocess = maia_chess_backend.maia.TFProcess(cfg, name, collection_name)

    if experimental_parser:
        train_dataset = tf.data.Dataset.from_tensor_slices(train_chunks).shuffle(len(train_chunks)).repeat()\
                         .interleave(lambda x: tf.data.FixedLengthRecordDataset(x, 8292, compression_type='GZIP', num_parallel_reads=1).filter(sample), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                         .shuffle(shuffle_size)\
                         .batch(split_batch_size).map(extract_inputs_outputs).prefetch(4)
    else:
        train_parser = maia_chess_backend.maia.ChunkParser(FileDataSrc(train_chunks.copy()),
                shuffle_size=shuffle_size, sample=SKIP, batch_size=maia_chess_backend.maia.ChunkParser.BATCH_SIZE)
        train_dataset = tf.data.Dataset.from_generator(
            train_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
        train_dataset = train_dataset.map(maia_chess_backend.maia.ChunkParser.parse_function)
        train_dataset = train_dataset.prefetch(4)

    shuffle_size = int(shuffle_size)
    if experimental_parser:
        test_dataset = tf.data.Dataset.from_tensor_slices(test_chunks).shuffle(len(test_chunks)).repeat()\
                         .interleave(lambda x: tf.data.FixedLengthRecordDataset(x, 8292, compression_type='GZIP', num_parallel_reads=1).filter(sample), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                         .shuffle(shuffle_size)\
                         .batch(split_batch_size).map(extract_inputs_outputs).prefetch(4)
    else:
        test_parser = maia_chess_backend.maia.ChunkParser(FileDataSrc(test_chunks),
                shuffle_size=shuffle_size, sample=SKIP, batch_size=maia_chess_backend.maia.ChunkParser.BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_generator(
            test_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
        test_dataset = test_dataset.map(maia_chess_backend.maia.ChunkParser.parse_function)
        test_dataset = test_dataset.prefetch(4)

    tfprocess.init_v2(train_dataset, test_dataset)

    tfprocess.restore_v2()

    # If number of test positions is not given
    # sweeps through all test chunks statistically
    # Assumes average of 10 samples per test game.
    # For simplicity, testing can use the split batch size instead of total batch size.
    # This does not affect results, because test results are simple averages that are independent of batch size.
    num_evals = cfg['training'].get('num_test_positions', len(test_chunks) * 10)
    num_evals = max(1, num_evals // maia_chess_backend.maia.ChunkParser.BATCH_SIZE)
    print("Using {} evaluation batches".format(num_evals))

    tfprocess.process_loop_v2(total_batch_size, num_evals, batch_splits=batch_splits)

    if cfg['training'].get('swa_output', False):
        tfprocess.save_swa_weights_v2(output_name)
    else:
        tfprocess.save_leelaz_weights_v2(output_name)

    train_parser.shutdown()
    test_parser.shutdown()

def get_latest_chunks(path):
    chunks = []
    maia_chess_backend.printWithDate(f"found {glob.glob(path)} chunk dirs")
    for d in glob.glob(path):
        maia_chess_backend.printWithDate(f"found {len(chunks)} chunks", end = '\r')
        chunks += glob.glob(os.path.join(d, '*.gz'))
    maia_chess_backend.printWithDate(f"found {len(chunks)} chunks total")
    if len(chunks) < 10:
        print("Not enough chunks {}".format(len(chunks)))
        sys.exit(1)
    if len(chunks) < 1000:
        print("There are not very many chunks so results may be unstable")

    print("sorting {} chunks...".format(len(chunks)), end='')
    chunks.sort(key=os.path.getmtime, reverse=True)
    print("[done]")
    print("{} - {}".format(os.path.basename(chunks[-1]), os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks


class FileDataSrc:
    """
        data source yielding chunkdata from chunk files.
    """
    def __init__(self, chunks):
        self.chunks = []
        self.done = chunks
    def next(self):
        if not self.chunks:
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)
        if not self.chunks:
            return None
        while len(self.chunks):
            filename = self.chunks.pop()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    self.done.append(filename)
                    return chunk_file.read()
            except:
                print("failed to parse {}".format(filename))

def extract_inputs_outputs(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 7432 are easy, policy extraction.
    policy = tf.io.decode_raw(tf.strings.substr(raw, 4, 7432), tf.float32)
    # Next are 104 bit packed chess boards, they have to be expanded.
    bit_planes = tf.expand_dims(tf.reshape(tf.io.decode_raw(tf.strings.substr(raw, 7436, 832), tf.uint8), [-1, 104, 8]), -1)
    bit_planes = tf.bitwise.bitwise_and(tf.tile(bit_planes, [1, 1, 1, 8]), [128, 64, 32, 16, 8, 4, 2, 1])
    bit_planes = tf.minimum(1., tf.cast(bit_planes, tf.float32))
    # Next 5 planes are 1 or 0 to indicate 8x8 of 1 or 0.
    unit_planes = tf.expand_dims(tf.expand_dims(tf.io.decode_raw(tf.strings.substr(raw, 8268, 5), tf.uint8), -1), -1)
    unit_planes = tf.cast(tf.tile(unit_planes, [1, 1, 8, 8]), tf.float32)
    # rule50 count plane.
    rule50_plane = tf.expand_dims(tf.expand_dims(tf.io.decode_raw(tf.strings.substr(raw, 8273, 1), tf.uint8), -1), -1)
    rule50_plane = tf.cast(tf.tile(rule50_plane, [1, 1, 8, 8]), tf.float32)
    rule50_plane = tf.divide(rule50_plane, 99.)
    # zero plane and one plane
    zero_plane = tf.zeros_like(rule50_plane)
    one_plane = tf.ones_like(rule50_plane)
    inputs = tf.reshape(tf.concat([bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1), [-1, 112, 64])

    # winner is stored in one signed byte and needs to be converted to one hot.
    winner = tf.cast(tf.io.decode_raw(tf.strings.substr(raw, 8275, 1), tf.int8), tf.float32)
    winner = tf.tile(winner, [1,3])
    z = tf.cast(tf.equal(winner, [1., 0., -1.]), tf.float32)

    # Outcome distribution needs to be calculated from q and d.
    best_q = tf.io.decode_raw(tf.strings.substr(raw, 8280, 4), tf.float32)
    best_d = tf.io.decode_raw(tf.strings.substr(raw, 8288, 4), tf.float32)
    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)

    q = tf.concat([best_q_w, best_d, best_q_l], 1)

    return (inputs, policy, z, q)

def sample(x):
    return tf.math.equal(tf.random.uniform([], 0, SKIP-1, dtype=tf.int32), 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow pipeline for training Leela Chess.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config', help='config file for model / training')
    args = parser.parse_args()

    collection_name = os.path.basename(os.path.dirname(args.config)).replace('configs_', '')
    name = os.path.basename(args.config).split('.')[0]

    multiprocessing.set_start_method('spawn')
    main(args.config, name, collection_name)
    multiprocessing.freeze_support()
