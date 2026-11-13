import torch
import shutil
import gc
import os
import time
import struct
import logging
import numpy
from abc import ABC, abstractmethod
from collections.abc import Sequence 
from types import TracebackType
from enum import Enum
from typing import Type, Optional, Union, Tuple, List



try:
    import awkward as ak

    HAVE_AWKWARD = True
except ModuleNotFoundError:
    HAVE_AWKWARD = False



logger = logging.getLogger(__name__)

INDEX_HEADER = b"LLMIDX\x00\x00\x00"  # 9 bytes
# layout: header(9) + dtype code(1) + seq_count(8) + doc_count(8) = 26 bytes
HEADER_SIZE = len(INDEX_HEADER) + 1 + 8 + 8


class DType(Enum):
    """dtype management class"""
    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, dtype: Type[numpy.number]):
        return cls[numpy.dtype(dtype).name].value

    @classmethod
    def dtype_from_code(cls, code):
        return getattr(numpy, cls(code).name)

    @staticmethod
    def get_size(key: Union[int, Type[numpy.number]]):
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        dt = numpy.dtype(key)  
        if not numpy.issubdtype(dt, numpy.number):
            raise ValueError(f"cannot get size of {key}")
        return dt.itemsize


class _IndexWriter:
    """to write the metadata into .idx file , uses context management manually"""
    def __init__(self, idx_path: str, dtype: Type[numpy.number]):
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self):
        self.idx_writer = open(self.idx_path, "wb")
        self.idx_writer.write(INDEX_HEADER)  # 9 bytes
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype))) # 1 byte 
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        self.idx_writer.close()
        return None

    def write(
        self,
        sequence_lengths: Sequence[Union[int, numpy.integer]],
        document_indices: Sequence[Union[int, numpy.integer]],
    ):
        # added: catch a builder that forgot to close the last document
        if len(document_indices) == 0 or document_indices[-1] != len(sequence_lengths):
            raise ValueError("document_indices must be non-empty and end at the sequence count")

        sequence_pointers = self._sequence_pointers(sequence_lengths)

        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # tokens in each sequence
        self.idx_writer.write(numpy.array(sequence_lengths, dtype=numpy.int32).tobytes(order="C"))
        # byte offset of each sequence
        self.idx_writer.write(numpy.array(sequence_pointers, dtype=numpy.int64).tobytes(order="C"))
        # end of each document
        self.idx_writer.write(numpy.array(document_indices, dtype=numpy.int64).tobytes(order="C"))

    def _sequence_pointers(self, sequence_lengths):
        # fix: vectorised, the python loop is very slow for millions of sequences
        itemsize = DType.get_size(self.dtype)
        lengths = numpy.asarray(sequence_lengths, dtype=numpy.int64)
        pointers = numpy.zeros(len(lengths), dtype=numpy.int64)
        numpy.cumsum(lengths[:-1] * itemsize, out=pointers[1:])
        return pointers


class _IndexReader:
    """to read the metadata file(.idx) with a fast numpy's memap load"""
    def __init__(
        self,
        idx_path,
        seqndocs_per_dataset: Optional[Tuple[int, int]] = None,
        dtype_code: int = None,
    ):
        self.idx_path = idx_path
        if seqndocs_per_dataset:
            if dtype_code is None: 
                raise ValueError("dtype_code is required when seqndocs_per_dataset is given")
            self.dtype = DType.dtype_from_code(dtype_code)
            self.dtype_size = DType.get_size(self.dtype)
            self.sequence_count = seqndocs_per_dataset[0]
            self.document_count = seqndocs_per_dataset[1]
            offset = HEADER_SIZE
        else:
            with open(self.idx_path, "rb") as stream:
                header = stream.read(len(INDEX_HEADER))
                if header != INDEX_HEADER: 
                    raise ValueError(f"bad header, cannot read: {idx_path}")

                code = struct.unpack("<B", stream.read(1))[0]
                self.dtype = DType.dtype_from_code(code)
                self.dtype_size = DType.get_size(self.dtype)

                self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
                self.document_count = struct.unpack("<Q", stream.read(8))[0]

                offset = stream.tell()

        self.idx_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.idx_buffer = memoryview(self.idx_buffer_mmap)

        self.sequence_lengths = numpy.frombuffer(
            self.idx_buffer,
            dtype=numpy.int32,
            count=self.sequence_count,
            offset=offset,
        )
        self.sequence_pointers = numpy.frombuffer(
            self.idx_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        self.document_indices = numpy.frombuffer(
            self.idx_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )



    def __len__(self):
        return self.sequence_count

    def __getitem__(self, idx):
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
        )


class _BinReader(ABC):
    """abstract class for reading .bin"""

    @abstractmethod
    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        pass


class MMapBinReader(_BinReader):
    """works when mmap bin read is desired, faster than others"""
    def __init__(self, file_path):
        self._bin_buffer_mmap = numpy.memmap(file_path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap.data)

    def read(self, dtype: Type[numpy.number], count: int, offset: int):
        return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

   


class FileBinReader(_BinReader): 
    """works if read other without mmap """
    def __init__(self, file_path, num_max_retries=3, sleep_duration_start=0.1):
        self.file_path = file_path
        self.num_max_retries = num_max_retries
        self.sleep_duration_start = sleep_duration_start

    def read(self, dtype: Type[numpy.number], count: int, offset: int):
        def _read():
            sequence = numpy.empty(shape=count, dtype=dtype)
            with open(self.file_path, "rb") as cur:
                cur.seek(offset)
                n = cur.readinto(sequence)
                if n != sequence.nbytes:
                    raise OSError(f"short read: expected {sequence.nbytes} bytes, got {n}")
            return sequence

        sleep_duration = self.sleep_duration_start
        for i in range(self.num_max_retries + 1):
            try:
                return _read()
            except OSError: 
                if i == self.num_max_retries:
                    raise
                time.sleep(sleep_duration)
                sleep_duration *= 2


class BinIdxDataset:
    """this gets the token sequnces and other metadata for training purpsoses. serves as the final class as gives the token sequences """
    def __init__(
        self,
        path_prefix: str,
        mmap: bool = True,  
        sequences_per_dataset: Optional[Tuple[int, int]] = None,
        dtype_code: int = None,
    ):
        self.path_prefix = path_prefix
        self.sequences_per_dataset = sequences_per_dataset
        self.mmap = mmap
        self.dtype_code = dtype_code

        
        if sequences_per_dataset is None and not self.exists(path_prefix):
            raise FileNotFoundError(f"missing .idx or .bin file for prefix {path_prefix}")

        if self.mmap:
            self.bin_reader = MMapBinReader(get_bin_path(self.path_prefix))
        else:
            self.bin_reader = FileBinReader(get_bin_path(self.path_prefix))  # fix: bin_reader was never set when mmap=False
        self.idx_reader = _IndexReader(
            get_idx_path(self.path_prefix), sequences_per_dataset, dtype_code=self.dtype_code
        )

   
        if sequences_per_dataset is None:
            doc_idx = self.idx_reader.document_indices
            if len(doc_idx) == 0 or doc_idx[-1] != len(self.idx_reader):
                raise ValueError("corrupt .idx: last document index must equal the sequence count")


    def __len__(self) -> int:
        """Number of sequences in the index."""
        return len(self.idx_reader)

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, List[numpy.ndarray]]:
        """Whole sequence for an int, list of sequences for a contiguous slice."""
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length = self.idx_reader[idx]
            return self.bin_reader.read(
                dtype=self.idx_reader.dtype,  
                count=int(sequence_length),
                offset=int(sequence_pointer),
            )
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into the dataset must be contiguous")
            if start >= stop: 
                return []
            sequence_lengths = self.idx_reader.sequence_lengths[start:stop]
           
            total_tokens = int(sequence_lengths.sum(dtype=numpy.int64))
            boundaries = numpy.cumsum(sequence_lengths, dtype=numpy.int64)[:-1]
            flat = self.bin_reader.read(
                dtype=self.idx_reader.dtype,
                count=total_tokens,
                offset=int(self.idx_reader.sequence_pointers[start]),
            )
            return numpy.split(flat, boundaries)
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """Tokens [offset, offset + length) of sequence idx. offset/length are in tokens."""
        sequence_pointer, sequence_length = self.idx_reader[idx]
        sequence_length = int(sequence_length)
        if length is None:
            length = sequence_length - offset
       
        if offset < 0 or length < 0 or offset + length > sequence_length:
            raise ValueError(
                f"offset={offset}, length={length} out of range for sequence of {sequence_length} tokens"
            )
       
        byte_offset = int(sequence_pointer) + offset * self.idx_reader.dtype_size
        return self.bin_reader.read(
            dtype=self.idx_reader.dtype, count=length, offset=byte_offset
        )

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        """Get the sequence lengths."""
        return self.idx_reader.sequence_lengths

    @property
    def document_indices(self) -> numpy.ndarray:
        """Get the document indices."""
        return self.idx_reader.document_indices

    @staticmethod
    def exists(path_prefix: str) -> bool:
        return os.path.exists(get_idx_path(path_prefix)) and os.path.exists(
            get_bin_path(path_prefix)
        )


    
class IndexedDatasetBuilder:
    """builds the dataset in the provided file path, have option to merge with other existing dataset too,
    handles the bin file as well as the .idx file for meta data 
    """
    def __init__(self, bin_path: str, dtype: Type[numpy.number] = numpy.int32):
        self.bin_path = bin_path
        self.dtype = dtype
        self.data_file = open(bin_path, "wb")  
        self.sequence_lengths = []             
        self.document_indices = [0]

    def add_item(self, tensor: torch.Tensor):
        """to add some sequence of token not specified to be a doc"""
        np_array = numpy.array(tensor.cpu().numpy(), dtype=self.dtype)  
        self.data_file.write(np_array.tobytes(order="C"))               
        self.sequence_lengths.append(np_array.size)

    def add_document(self, tensor: torch.Tensor, lengths: List[int]):
        """to add a document with given sequence_lenghts : lenghts """

        np_array = numpy.array(tensor.cpu().numpy(), dtype=self.dtype)
        if int(np_array.size) != int(sum(lengths)):                 
            raise ValueError("sum(lengths) must equal the number of tokens")
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))

    def add_documents(self, documents: "ak.Array", eod_token: Optional[int] = None,
                      chunk_size: int = 1_000_000):
        """this lets add the whole set of document does in chunk as the size can grow large , 
        considers one sequence to be whole document, and writes the eod token if specified
        """
        if not HAVE_AWKWARD:
            raise ModuleNotFoundError("Please install it with `pip install awkward`.")
        n_docs = len(documents)
        for i in range(0, n_docs, chunk_size):
            chunk = documents[i : i + chunk_size]
            np_array = numpy.asarray(ak.flatten(chunk), dtype=self.dtype)
            doc_lengths = numpy.asarray(ak.num(chunk, axis=1), dtype=numpy.int64)
            if eod_token is not None:
                insert_at = numpy.cumsum(doc_lengths)
                np_array = numpy.insert(np_array, insert_at, eod_token).astype(self.dtype, copy=False)  
                doc_lengths = doc_lengths + 1                                                           
            self.data_file.write(np_array.tobytes(order="C"))                                          
            del np_array
            offset = len(self.sequence_lengths)
            self.sequence_lengths.extend(doc_lengths.tolist())
            self.document_indices.extend((offset + numpy.arange(1, len(doc_lengths) + 1)).tolist())

    def end_document(self) -> None:
        self.document_indices.append(len(self.sequence_lengths))

    def add_index(self, path_prefix):
        """ helps in merging the two datasets , merge the bin files and provide content for writing the final .idx file as well"""
        idx_reader = _IndexReader(get_idx_path(path_prefix))
        if numpy.dtype(idx_reader.dtype) != numpy.dtype(self.dtype):  
            raise ValueError("dtype mismatch, cannot merge")
        offset = len(self.sequence_lengths)
        self.sequence_lengths.extend(idx_reader.sequence_lengths.tolist())             
        self.document_indices.extend((offset + idx_reader.document_indices[1:]).tolist())  
        del idx_reader
        gc.collect()
        with open(get_bin_path(path_prefix), "rb") as f:               
            shutil.copyfileobj(f, self.data_file)

    def finalize(self, idx_path: str) -> None:
        """finalise the datatset writes the final .idx file"""
        self.data_file.close()                                          
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.document_indices)

#helper funcitons 

def get_idx_path(path_prefix: str) -> str:
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    return path_prefix + ".bin"