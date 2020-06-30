#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2018 Folkert Huizinga
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

import itertools
import multiprocessing as mp
import numpy as np
import random
import struct
import tensorflow as tf
import unittest

from .shufflebuffer import ShuffleBuffer

VERSION = struct.pack('i', 3)
STRUCT_STRING = '4s7432s832sBBBBBBBb'

# Interface for a chunk data source.
class ChunkDataSrc:
    def __init__(self, items):
        self.items = items
    def next(self):
        if not self.items:
            return None
        return self.items.pop()



class ChunkParser:
    # static batch size
    BATCH_SIZE = 8
    def __init__(self, chunkdatasrc, shuffle_size=1, sample=1, buffer_size=1, batch_size=256, workers=None):
        """
        Read data and yield batches of raw tensors.

        'chunkdatasrc' is an object yeilding chunkdata
        'shuffle_size' is the size of the shuffle buffer.
        'sample' is the rate to down-sample.
        'workers' is the number of child workers to use.

        The data is represented in a number of formats through this dataflow
        pipeline. In order, they are:

        chunk: The name of a file containing chunkdata

        chunkdata: type Bytes. Multiple records of v3 format where each record
        consists of (state, policy, result)

        raw: A byte string holding raw tensors contenated together. This is
        used to pass data from the workers to the parent. Exists because
        TensorFlow doesn't have a fast way to unpack bit vectors. 7950 bytes
        long.
        """

        # Build 2 flat float32 planes with values 0,1
        self.flat_planes = []
        for i in range(2):
            self.flat_planes.append(np.zeros(64, dtype=np.float32) + i)

        # set the down-sampling rate
        self.sample = sample
        # set the mini-batch size
        self.batch_size = batch_size
        # set number of elements in the shuffle buffer.
        self.shuffle_size = shuffle_size
        # Start worker processes, leave 2 for TensorFlow
        if workers is None:
            #At high values they end up doing nothing
            workers = 5#max(1, mp.cpu_count() - 2)

        print("Using {} worker processes.".format(workers))

        # Start the child workers running
        self.readers = []
        self.writers = []
        self.processes = []
        for _ in range(workers):
            read, write = mp.Pipe(duplex=False)
            p = mp.Process(target=self.task, args=(chunkdatasrc, write))
            self.processes.append(p)
            p.start()
            self.readers.append(read)
            self.writers.append(write)
        self.init_structs()

    def __del__(self):
        """Hoping this works"""
        self.shutdown()

    def shutdown(self):
        """
        Terminates all the workers
        """
        for i in range(len(self.readers)):
            self.processes[i].terminate()
            self.processes[i].join()
            self.readers[i].close()
            self.writers[i].close()


    def init_structs(self):
        """
        struct.Struct doesn't pickle, so it needs to be separately
        constructed in workers.

        V3 Format (8276 bytes total)
            int32 version (4 bytes)
            1858 float32 probabilities (7432 bytes)  (removed 66*4 = 264 bytes unused under-promotions)
            104 (13*8) packed bit planes of 8 bytes each (832 bytes)  (no rep2 plane)
            uint8 castling us_ooo (1 byte)
            uint8 castling us_oo (1 byte)
            uint8 castling them_ooo (1 byte)
            uint8 castling them_oo (1 byte)
            uint8 side_to_move (1 byte) aka us_black
            uint8 rule50_count (1 byte)
            uint8 move_count (1 byte)
            int8 result (1 byte)
        """
        self.v3_struct = struct.Struct(STRUCT_STRING)

    @staticmethod
    def parse_function(planes, probs, winner):
        """
        Convert unpacked record batches to tensors for tensorflow training
        """
        planes = tf.decode_raw(planes, tf.float32)
        probs = tf.decode_raw(probs, tf.float32)
        winner = tf.decode_raw(winner, tf.float32)

        planes = tf.reshape(planes, (ChunkParser.BATCH_SIZE, 112, 8*8))
        probs = tf.reshape(probs, (ChunkParser.BATCH_SIZE, 1858))
        winner = tf.reshape(winner, (ChunkParser.BATCH_SIZE, 1))

        return (planes, probs, winner)


    def convert_v3_to_tuple(self, content):
        """
        Unpack a v3 binary record to 3-tuple (state, policy pi, result)

        v3 struct format is (8276 bytes total)
            int32 version (4 bytes)
            1858 float32 probabilities (7432 bytes)
            104 (13*8) packed bit planes of 8 bytes each (832 bytes)
            uint8 castling us_ooo (1 byte)
            uint8 castling us_oo (1 byte)
            uint8 castling them_ooo (1 byte)
            uint8 castling them_oo (1 byte)
            uint8 side_to_move (1 byte)
            uint8 rule50_count (1 byte)
            uint8 move_count (1 byte)
            int8 result (1 byte)
        """
        (ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner) = self.v3_struct.unpack(content)
        # Enforce move_count to 0
        move_count = 0

        # Unpack bit planes and cast to 32 bit float
        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)
        rule50_plane = (np.zeros(8*8, dtype=np.float32) + rule50_count) / 99

        # Concatenate all byteplanes. Make the last plane all 1's so the NN can
        # detect edges of the board more easily
        planes = planes.tobytes() + \
                 self.flat_planes[us_ooo].tobytes() + \
                 self.flat_planes[us_oo].tobytes() + \
                 self.flat_planes[them_ooo].tobytes() + \
                 self.flat_planes[them_oo].tobytes() + \
                 self.flat_planes[stm].tobytes() + \
                 rule50_plane.tobytes() + \
                 self.flat_planes[move_count].tobytes() + \
                 self.flat_planes[1].tobytes()

        assert len(planes) == ((8*13*1 + 8*1*1) * 8 * 8 * 4)
        winner = float(winner)
        assert winner == 1.0 or winner == -1.0 or winner == 0.0
        winner = struct.pack('f', winner)

        return (planes, probs, winner)

    def sample_record(self, chunkdata):
        """
        Randomly sample through the v3 chunk data and select records
        """
        if chunkdata[0:4] == VERSION:
            for i in range(0, len(chunkdata), self.v3_struct.size):
                if self.sample > 1:
                    # Downsample, using only 1/Nth of the items.
                    if random.randint(0, self.sample-1) != 0:
                        continue  # Skip this record.
                yield chunkdata[i:i+self.v3_struct.size]

    def task(self, chunkdatasrc, writer):
        """
        Run in fork'ed process, read data from chunkdatasrc, parsing, shuffling and
        sending v3 data through pipe back to main process.
        """
        self.init_structs()
        while True:
            chunkdata = chunkdatasrc.next()
            if chunkdata is None:
                break
            for item in self.sample_record(chunkdata):
                # NOTE: This requires some more thinking, we can't just apply a
                # reflection along the horizontal or vertical axes as we would
                # also have to apply the reflection to the move probabilities
                # which is non trivial for chess.
                writer.send_bytes(item)

    def v3_gen(self):
        """
        Read v3 records from child workers, shuffle, and yield
        records.
        """
        sbuff = ShuffleBuffer(self.v3_struct.size, self.shuffle_size)
        while len(self.readers):
            #for r in mp.connection.wait(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue  # shuffle buffer not yet full
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        # drain the shuffle buffer.
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s


    def tuple_gen(self, gen):
        """
        Take a generator producing v3 records and convert them to tuples.
        applying a random symmetry on the way.
        """
        for r in gen:
            yield self.convert_v3_to_tuple(r)


    def batch_gen(self, gen):
        """
        Pack multiple records into a single batch
        """
        # Get N records. We flatten the returned generator to
        # a list because we need to reuse it.
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s):
                return
            yield ( b''.join([x[0] for x in s]),
                    b''.join([x[1] for x in s]),
                    b''.join([x[2] for x in s]) )


    def parse(self):
        """
        Read data from child workers and yield batches of unpacked records
        """
        gen = self.v3_gen()        # read from workers
        gen = self.tuple_gen(gen)  # convert v3->tuple
        gen = self.batch_gen(gen)  # assemble into batches
        for b in gen:
            yield b



# Tests to check that records parse correctly
class ChunkParserTest(unittest.TestCase):
    def setUp(self):
        self.v3_struct = struct.Struct(STRUCT_STRING)

    def generate_fake_pos(self):
        """
        Generate a random game position.
        Result is ([[64] * 104], [1]*5, [1858], [1])
        """
        # 0. 104 binary planes of length 64
        planes = [np.random.randint(2, size=64).tolist() for plane in range(104)]

        # 1. generate the other integer data
        integer = np.zeros(7, dtype=np.int32)
        for i in range(5):
            integer[i] = np.random.randint(2)
        integer[5] = np.random.randint(100)

        # 2. 1858 probs
        probs = np.random.randint(9, size=1858, dtype=np.int32)

        # 3. And a winner: 1, 0, -1
        winner = np.random.randint(3) - 1
        return (planes, integer, probs, winner)


    def v3_record(self, planes, i, probs, winner):
        pl = []
        for plane in planes:
            pl.append(np.packbits(plane))
        pl = np.array(pl).flatten().tobytes()
        pi = probs.tobytes()
        return self.v3_struct.pack(VERSION, pi, pl, i[0], i[1], i[2], i[3], i[4], i[5], i[6], winner)


    def test_structsize(self):
        """
        Test struct size
        """
        self.assertEqual(self.v3_struct.size, 8276)


    def test_parsing(self):
        """
        Test game position decoding pipeline.
        """
        truth = self.generate_fake_pos()
        batch_size = 4
        records = []
        for i in range(batch_size):
            record = b''
            for j in range(2):
                record += self.v3_record(*truth)
            records.append(record)

        parser = ChunkParser(ChunkDataSrc(records), shuffle_size=1, workers=1, batch_size=batch_size)
        batchgen = parser.parse()
        data = next(batchgen)

        batch = ( np.reshape(np.frombuffer(data[0], dtype=np.float32), (batch_size, 112, 64)),
                  np.reshape(np.frombuffer(data[1], dtype=np.int32), (batch_size, 1858)),
                  np.reshape(np.frombuffer(data[2], dtype=np.float32), (batch_size, 1)) )

        fltplanes = truth[1].astype(np.float32)
        fltplanes[5] /= 99
        for i in range(batch_size):
            data = (batch[0][i][:104], np.array([batch[0][i][j][0] for j in range(104,111)]), batch[1][i], batch[2][i])
            self.assertTrue((data[0] == truth[0]).all())
            self.assertTrue((data[1] == fltplanes).all())
            self.assertTrue((data[2] == truth[2]).all())
            self.assertEqual(data[3][0], truth[3])

        parser.shutdown()


    def test_tensorflow_parsing(self):
        """
        Test game position decoding pipeline including tensorflow.
        """
        truth = self.generate_fake_pos()
        batch_size = 4
        ChunkParser.BATCH_SIZE = batch_size
        records = []
        for i in range(batch_size):
            record = b''
            for j in range(2):
                record += self.v3_record(*truth)
            records.append(record)

        parser = ChunkParser(ChunkDataSrc(records), shuffle_size=1, workers=1, batch_size=batch_size)
        batchgen = parser.parse()
        data = next(batchgen)

        planes = np.frombuffer(data[0], dtype=np.float32, count=112*8*8*batch_size)
        planes = planes.reshape(batch_size, 112, 8*8)
        probs = np.frombuffer(data[1], dtype=np.float32, count=1858*batch_size)
        probs = probs.reshape(batch_size, 1858)
        winner = np.frombuffer(data[2], dtype=np.float32, count=1*batch_size)

        # Pass it through tensorflow
        with tf.Session() as sess:
            graph = ChunkParser.parse_function(data[0], data[1], data[2])
            tf_planes, tf_probs, tf_winner = sess.run(graph)

            for i in range(batch_size):
                self.assertTrue((probs[i] == tf_probs[i]).all())
                self.assertTrue((planes[i] == tf_planes[i]).all())
                self.assertTrue((winner[i] == tf_winner[i]).all())

        parser.shutdown()


if __name__ == '__main__':
    unittest.main()
