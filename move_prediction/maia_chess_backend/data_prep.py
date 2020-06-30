import os
import os.path
import zipfile
import gzip
import io
import shutil
import tempfile
import multiprocessing
import subprocess
import random
import tensorflow as tf

from .chunkparser import ChunkParser

#Parameters from Leela that probably don't need to be changed
shuffle_size = 524288

#Some files needed for preprocessing
weightsPath = os.path.abspath('weights_125.txt.gz')
lczeroPath = 'lczero'
pgnExtractPath = 'pgn-extract'


def cleanFile(filePath):
    try:
        subprocess.run(args = [pgnExtractPath, '-7', '-C', '-N',  '-#400', filePath], check = True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Something went wrong cleaning: {filePath}:\n{e}")

    outputs = []
    for e in os.scandir('.'):
        if e.name.split('.')[0].isdigit() and e.name.endswith('.pgn'):
            newName = f"{e.name.split('.')[0]}-{os.path.basename(filePath)}"
            shutil.move(e.path, newName)
            outputs.append(os.path.abspath(newName))
    if len(outputs) < 1:
        raise RuntimeError(f"To few files found in {os.getcwd()}: {os.listdir('.')}")
    return outputs

def lczeroFile(filePath):
    try:
        subprocess.run(args = [lczeroPath, '-w', weightsPath, '--supervise', filePath], check = True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        #lczero throws an error at the end of the file
        pass
    pgns = []
    for e in os.scandir('.'):
        if e.is_dir():
            pgns += [os.path.abspath(g.path) for g in os.scandir(e.path) if g.name.endswith('.gz')]
    if len(pgns) < 1:
        raise RuntimeError(f"To few files found in {os.getcwd()}: {os.listdir('.')}")
    return pgns

def binarizeFiles(targetFiles):
    with multiprocessing.Pool(processes=5) as pool:
        binaryFiles = pool.map(lczeroFile, targetFiles)
    return [p for i in binaryFiles for p in i]

def loadFiles(outputFile, filePaths, train_ratio, workingDir = '.'):
    if isinstance(filePaths, str):
        filePaths = [filePaths]
    filePaths = [os.path.abspath(p) for p in filePaths]
    outputFile = os.path.abspath(outputFile)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(dir='.', prefix='temp_') as tempdir:
        try:
            tempdir = os.path.abspath(tempdir)
            os.chdir(tempdir)
            cleanedPaths = []
            for p in filePaths:
                print(f"Cleaning: {p}")
                cleanedPaths += cleanFile(p)
            print(f"Binarizing: {cleanedPaths}")
            binFiles = binarizeFiles(cleanedPaths)

            random.shuffle(binFiles)

            num_chunks = len(binFiles)
            num_train = int(num_chunks * train_ratio)
            num_test = num_chunks - num_train

            print(f"Saving {len(binFiles)} files to: {outputFile}")
            with zipfile.ZipFile(outputFile, 'w') as myzip:
                for i, gzipPath in enumerate(binFiles[:num_train]):
                    myzip.write(
                        filename = gzipPath,
                        arcname = f"chunks/training/{i}.v3.gz"
                        )
                for i, gzipPath in enumerate(binFiles[num_train:]):
                    myzip.write(
                        filename = gzipPath,
                        arcname = f"chunks/testing/{i}.v3.gz"
                        )
        finally:
            os.chdir(cwd)
    return num_train, num_test

class FileDataSrc_OnDisk:
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

    def __len__(self):
        return len(self.chunks) + len(self.done)

class FileDataSrc_sequential(object):
    """
        data source yielding chunkdata from chunk files.
    """
    def __init__(self, chunks, subchunk_size, subchunk_iterations):
        self.current_subchunk = 0
        self.iteration_count = 0
        self.max_iters = subchunk_iterations
        self.chunks = {}
        self.length = len(chunks)
        random.shuffle(chunks)
        for n, i in enumerate(range(0, len(chunks), subchunk_size)):
            self.chunks[n] = {'done' : [], 'working' : chunks[i: i + subchunk_size]}

    def next(self):

        wChunks = self.chunks[self.current_subchunk]['working']
        wDone = self.chunks[self.current_subchunk]['done']
        if len(wChunks) < 1:

            self.chunks[self.current_subchunk]['working'], self.chunks[self.current_subchunk]['done'] = self.chunks[self.current_subchunk]['done'], self.chunks[self.current_subchunk]['working']

            self.iteration_count += 1
            if self.iteration_count >= self.max_iters:
                self.current_subchunk = (self.current_subchunk + 1) % len(self.chunks)
            wChunks = self.chunks[self.current_subchunk]['working']
            wDone = self.chunks[self.current_subchunk]['done']
            random.shuffle(wChunks)

        if len(wChunks) < 1:
            return None
        while len(wChunks):
            filename = wChunks.pop()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    wDone.append(filename)
                    return chunk_file.read()
            except:
                print("failed to parse {}".format(filename))

    def __len__(self):
        return self.length


class FileDataSrc_InMemory(object):
    """
        Modification of the Leela engine to allow reading from a single zip file into memory
    """
    def __init__(self, zipPath, prefix):
        print(f"Starting parser for: {zipPath} {prefix}")
        self.path = zipPath
        self.prefix = prefix
        with zipfile.ZipFile(self.path, 'r') as myzip:
            self.chunks = []
            self.done = []
            for c in myzip.filelist:
                if c.filename.startswith(prefix) and c.filename.endswith('.v3.gz'):
                    with myzip.open(c.filename) as f:
                        try:
                            with gzip.open(io.BytesIO(f.read()), 'rb') as chunk_file:
                                self.done.append(chunk_file.read())
                        except:
                            print("failed to parse {}".format(c))
                            raise

    def next(self):
        if not self.chunks:
            #If len == 0
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)
        if not self.chunks:
            #Don't like this
            return None
        while len(self.chunks):
            dat = self.chunks.pop()
            self.done.append(dat)
            return dat

    def __len__(self):
        return len(self.chunks) + len(self.done)

def genDataSets(trainSRC, testSRC, batch_size, sample_rate, num_workers):

    ChunkParser.BATCH_SIZE = batch_size

    #trainSRC = FileDataSrc(dataFile, 'chunks/training')
    train_parser = ChunkParser(
                trainSRC,
                shuffle_size = shuffle_size,
                sample = sample_rate,
                batch_size = batch_size,
                workers = num_workers,
                )
    train_dataset = tf.data.Dataset.from_generator(
                train_parser.parse,
                output_types=(tf.string, tf.string, tf.string),
                )
    train_dataset = train_dataset.map(ChunkParser.parse_function)
    train_dataset = train_dataset.prefetch(4)
    train_iterator = train_dataset.make_one_shot_iterator()

    #testSRC = FileDataSrc(dataFile, 'chunks/testing')

    train_ratio = len(testSRC)  / (len(trainSRC) + len(testSRC))
    test_parser = ChunkParser(
                testSRC,
                shuffle_size = int(shuffle_size * (1.0-train_ratio)),
                sample = sample_rate,
                batch_size = batch_size,
                workers = num_workers,
                )
    test_dataset = tf.data.Dataset.from_generator(
                test_parser.parse,
                output_types=(tf.string, tf.string, tf.string),
                )
    test_dataset = test_dataset.map(ChunkParser.parse_function)
    test_dataset = test_dataset.prefetch(4)
    test_iterator = test_dataset.make_one_shot_iterator()

    return test_dataset, train_iterator, test_iterator
