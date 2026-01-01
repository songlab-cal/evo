from typing import List, Tuple, Union, Iterator, Sequence, TextIO, Dict, Optional
from copy import copy
import contextlib
import math
import tempfile
import re
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from Bio import SeqIO
from Bio.Seq import Seq
from .typed import PathLike
from .tensor import apc


class MSA:
    """Class that represents a multiple sequence alignment.
    
    Supports both FASTA (.fasta) and TXT (.txt) file formats.
    Can optionally store alignment masks for full-length unaligned sequences.
    
    Source: Update existing class in peint-toolbox/src/evo/evo/align.py
    """
    
    # Standard amino acids (20 standard AAs)
    standard_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    # Standard nucleotides
    standard_nucleotides = "ACGTU"
    # Gap character
    gap_character = "-"

    @staticmethod
    def _process_sequence_for_aligned(sequence: str, standard_letters: str) -> str:
        """Helper: Process single sequence for aligned format (remove lowercase, keep gaps, alphabetize)."""
        # Remove lowercase letters (insertions)
        processed = "".join(char for char in sequence if not char.islower())
        
        # Alphabetize: convert non-standard letters to gaps
        allowed_chars = standard_letters + MSA.gap_character
        allowed_set = set(allowed_chars)
        processed = "".join(
            char if char in allowed_set else MSA.gap_character
            for char in processed
        )
        
        return processed
    
    @staticmethod
    def _process_sequence_for_unaligned(
        sequence: str, standard_letters: str
    ) -> Tuple[str, str]:
        """Helper: Process single sequence for unaligned format and generate mask.
        
        Returns:
            Tuple of (processed_sequence, alignment_mask)
        """
        processed_chars = []
        mask_chars = []
        
        for char in sequence:
            if char == MSA.gap_character:
                # Skip gaps entirely
                continue
            elif char.islower():
                # Lowercase = insertion, convert to uppercase
                processed_chars.append(char.upper())
                mask_chars.append("0")
            else:
                # Uppercase = aligned position
                processed_chars.append(char)
                mask_chars.append("1")
        
        processed = "".join(processed_chars)
        mask = "".join(mask_chars)
        
        # Alphabetize: convert non-standard letters to X
        allowed_set = set(standard_letters)
        new_processed = []
        new_mask = []
        for char, mask_char in zip(processed, mask):
            if char in allowed_set:
                new_processed.append(char)
                new_mask.append(mask_char)
            else:
                # Non-standard character: convert to X and mark as aligned
                new_processed.append("X")
                new_mask.append("1")
        
        return "".join(new_processed), "".join(new_mask)
    
    @staticmethod
    def _alphabetize_sequences(sequences: List[str], standard_letters: str) -> List[str]:
        """Helper: Alphabetize sequences (convert non-standard letters to gaps)."""
        allowed_chars = standard_letters + MSA.gap_character
        allowed_set = set(allowed_chars)
        return [
            "".join(
                char if char in allowed_set else MSA.gap_character
                for char in seq
            )
            for seq in sequences
        ]

    def __init__(
        self,
        sequences: List[Tuple[str, str]],
        includes_insertion_relative_to_query: bool = False,
        seqid_cutoff: float = 0.2,
        already_processed: bool = False,
        is_protein: bool = True,
    ):
        """
        Args:
            sequences: List of (header, sequence) tuples
            includes_insertion_relative_to_query: If True, lowercase letters are treated as 
                insertions relative to query. Sequences will be processed: aligned sequences 
                (lowercase stripped, gaps preserved) stored in self.sequences, and full-length 
                sequences stored in self.full_length_sequences. If False, all sequences 
                must have the same length and are stored as-is in self.sequences.
            seqid_cutoff: Sequence identity cutoff for filtering
            already_processed: If True, sequences are already processed and should be used as-is
                (skip processing logic). Used internally when creating MSAs from already-processed data.
        """
        if not sequences:
            raise ValueError("MSA must contain at least one sequence")
        
        self.headers = [header for header, _ in sequences]
        raw_sequences = [seq for _, seq in sequences]
        self.includes_insertion_relative_to_query = includes_insertion_relative_to_query
        self.seqid_cutoff = seqid_cutoff
        
        if already_processed:
            # Sequences are already processed, use as-is
            self.sequences = raw_sequences
            self._seqlen = len(raw_sequences[0]) if raw_sequences else 0
            self._depth = len(self.sequences)
            # These will be set by the caller if needed
            self.full_length_sequences = None
            self.alignment_masks = {}
            # Check if protein sequences
            self.is_protein = is_protein
            return

        # Get standard letters based on sequence type
        self.is_protein = is_protein
        standard_letters = (
            MSA.standard_amino_acids if self.is_protein else MSA.standard_nucleotides
        )
        
        # Query sequence (first sequence) should not have gap or lowercase
        if any(char.islower() or char == MSA.gap_character for char in raw_sequences[0]):
            raise ValueError("Query sequence should not have gap or lower case")
        
        # Process sequences based on includes_insertion_relative_to_query
        if includes_insertion_relative_to_query:
            # Process to aligned format (strip lowercase, keep gaps)
            aligned_sequences = []
            full_length_sequences = []
            alignment_masks = {}
            
            for header, raw_seq in zip(self.headers, raw_sequences):
                # Process for aligned format
                aligned_seq = self._process_sequence_for_aligned(raw_seq, standard_letters)
                aligned_sequences.append(aligned_seq)
                
                # Process for unaligned format (full-length sequences)
                full_seq, mask = self._process_sequence_for_unaligned(raw_seq, standard_letters)
                full_length_sequences.append(full_seq)
                alignment_masks[header] = mask
            
            # Validate that aligned sequences have same length
            aligned_lengths = [len(seq) for seq in aligned_sequences]
            if len(set(aligned_lengths)) > 1:
                raise ValueError(
                    f"Sequences have different lengths after stripping lowercase: {set(aligned_lengths)}"
                )
            
            self.sequences = aligned_sequences
            self.full_length_sequences = full_length_sequences
            self.alignment_masks = alignment_masks
            self._seqlen = aligned_lengths[0] if aligned_lengths else 0
        else:
            # All sequences must have same length
            lengths = [len(seq) for seq in raw_sequences]
            if len(set(lengths)) > 1:
                raise ValueError(
                    f"Seqlen Mismatch! Expected length {lengths[0]}, "
                    f"got lengths: {lengths}"
                )
            
            # Alphabetize sequences
            self.sequences = self._alphabetize_sequences(raw_sequences, standard_letters)
            self._seqlen = lengths[0]
            self.full_length_sequences = None
            self.alignment_masks = {}
        
        self._depth = len(self.sequences)

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return zip(self.headers, self.sequences)

    def select(self, indices: Sequence[int], axis: str = "seqs") -> "MSA":
        assert axis in ("seqs", "positions")
        if axis == "seqs":
            data = [(self.headers[idx], self.sequences[idx]) for idx in indices]
            result = self.__class__(
                data,
                includes_insertion_relative_to_query=self.includes_insertion_relative_to_query,
                seqid_cutoff=self.seqid_cutoff,
                already_processed=True,  # Already processed
            )
            # Preserve full_length_sequences and alignment_masks for selected sequences
            if self.full_length_sequences is not None:
                result.full_length_sequences = [
                    self.full_length_sequences[idx] for idx in indices
                ]
            if self.alignment_masks:
                result.alignment_masks = {
                    self.headers[idx]: self.alignment_masks[self.headers[idx]]
                    for idx in indices
                    if self.headers[idx] in self.alignment_masks
                }
            return result
        else:
            data = [
                (header, "".join(seq[idx] for idx in indices)) for header, seq in self
            ]
            result = self.__class__(
                data,
                includes_insertion_relative_to_query=self.includes_insertion_relative_to_query,
                seqid_cutoff=self.seqid_cutoff,
                already_processed=True,  # Already processed
            )
            # For position selection, full_length_sequences and masks need to be updated
            # For now, clear them as they may not be valid after position selection
            result.full_length_sequences = None
            result.alignment_masks = {}
            return result

    def swap(self, index1: int, index2: int) -> "MSA":
        headers = copy(self.headers)
        sequences = copy(self.sequences)
        headers[index1], headers[index2] = headers[index2], headers[index1]
        sequences[index1], sequences[index2] = sequences[index2], sequences[index1]
        data = list(zip(headers, sequences))
        result = self.__class__(
            data,
            includes_insertion_relative_to_query=self.includes_insertion_relative_to_query,
            seqid_cutoff=self.seqid_cutoff,
            already_processed=True,  # Already processed
        )
        # Preserve full_length_sequences and alignment_masks
        if self.full_length_sequences is not None:
            full_length = copy(self.full_length_sequences)
            full_length[index1], full_length[index2] = full_length[index2], full_length[index1]
            result.full_length_sequences = full_length
        result.alignment_masks = self.alignment_masks.copy()
        return result

    def filter_coverage(self, threshold: float, axis: str = "seqs") -> "MSA":
        assert 0 <= threshold <= 1
        assert axis in ("seqs", "positions")
        notgap = self.array != self.gap
        match = notgap.mean(1 if axis == "seqs" else 0)
        indices = np.where(match >= threshold)[0]
        return self.select(indices, axis=axis)

    def hhfilter(
        self,
        seqid: int = 90,
        diff: int = 0,
        cov: int = 0,
        qid: int = 0,
        qsc: float = -20.0,
        binary: str = "hhfilter",
    ) -> "MSA":

        with tempfile.TemporaryDirectory(dir="/dev/shm") as tempdirname:
            tempdir = Path(tempdirname)
            fasta_file = tempdir / "input.fasta"
            fasta_file.write_text(
                "\n".join(f">{i}\n{seq}" for i, seq in enumerate(self.sequences))
            )
            output_file = tempdir / "output.fasta"
            command = " ".join(
                [
                    f"{binary}",
                    f"-i {fasta_file}",
                    "-M a3m",
                    f"-o {output_file}",
                    f"-id {seqid}",
                    f"-diff {diff}",
                    f"-cov {cov}",
                    f"-qid {qid}",
                    f"-qsc {qsc}",
                ]
            ).split(" ")
            result = subprocess.run(command, capture_output=True)
            result.check_returncode()
            with output_file.open() as f:
                indices = [int(line[1:].strip()) for line in f if line.startswith(">")]
            return self.select(indices, axis="seqs")

    def replace_(self, inp: str, rep: str) -> "MSA":
        dtype = self.dtype
        self.dtype = np.dtype("S1")  # type: ignore
        self.array[self.array == inp.encode()] = rep.encode()
        self.dtype = dtype
        return self

    @property
    def gap(self) -> Union[bytes, int]:
        return b"-" if self.dtype == np.dtype("S1") else ord("-")

    def __repr__(self) -> str:
        return f"MSA, L: {self.seqlen}, N: {self.depth}\n" f"{self.array}"

    def __getitem__(self, idx):
        return self.array[idx]

    def pdist(self) -> np.ndarray:
        dtype = self.dtype
        self.dtype = np.uint8
        dist = squareform(pdist(self.array, "hamming"))
        self.dtype = dtype
        return dist

    def greedy_select(self, num_seqs: int, mode: str = "max") -> "MSA":
        assert mode in ("max", "min")
        if self.depth <= num_seqs:
            return self
        dtype = self.dtype
        self.dtype = np.uint8

        optfunc = np.argmax if mode == "max" else np.argmin
        all_indices = np.arange(self.depth)
        indices = [0]
        pairwise_distances = np.zeros((0, self.depth))
        for _ in range(num_seqs - 1):
            dist = cdist(self.array[indices[-1:]], self.array, "hamming")
            pairwise_distances = np.concatenate([pairwise_distances, dist])
            shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
            shifted_index = optfunc(shifted_distance)
            index = np.delete(all_indices, indices)[shifted_index]
            indices.append(index)
        indices = sorted(indices)
        self.dtype = dtype
        result = self.select(indices, axis="seqs")
        # alignment_masks already preserved by select()
        return result

    def sample_weights(self, num_seqs: int) -> "MSA":
        if self.depth <= num_seqs:
            return self
        weights = self.weights[1:]
        weights = weights / weights.sum()
        indices = (
            np.random.choice(
                self.depth - 1, size=num_seqs - 1, replace=False, p=weights
            )
            + 1
        )
        indices = np.sort(indices)
        indices = np.append(0, indices)
        return self.select(indices, axis="seqs")

    def select_diverse(self, num_seqs: int, method: str = "hhfilter") -> "MSA":
        assert method in ("hhfilter", "sample-weights")
        if num_seqs >= self.depth:
            return self

        if method == "hhfilter":
            msa = self.hhfilter(diff=num_seqs)
            if num_seqs < msa.depth:
                msa = msa.select(np.arange(num_seqs))
        else:
            msa = self.sample_weights(num_seqs)
        return msa

    def invcov(self) -> np.ndarray:
        """given one-hot encoded MSA, return contacts"""
        from sklearn.preprocessing import OneHotEncoder
        dtype = self.dtype
        self.dtype = np.uint8
        Y = OneHotEncoder(drop=[self.gap]).fit_transform(self.array.reshape(-1, 1)).toarray().reshape(self.depth, self.seqlen, -1)
        K = Y.shape[-1]
        Y_flat = Y.reshape(self.depth, -1)
        c = np.cov(Y_flat.T)
        self.dtype = dtype
        return np.linalg.norm(c.reshape(self.seqlen, K, self.seqlen, K), ord=2, axis=(1, 3))
        # shrink = 4.5 / math.sqrt(self.depth) * np.eye(c.shape[0])
        # ic = np.linalg.inv(c + shrink)
        # ic = ic.reshape(self.seqlen, K, self.seqlen, K)
        # return apc(np.sqrt(np.square(ic).sum((1, 3))))

    @property
    def array(self) -> np.ndarray:
        if not hasattr(self, "_array"):
            self._array = np.array([list(seq) for seq in self.sequences], dtype="|S1")
        return self._array

    @property
    def dtype(self) -> type:
        return self.array.dtype

    @dtype.setter
    def dtype(self, value: type) -> None:
        assert value in (np.uint8, np.dtype("S1"))
        self._array = self.array.view(value)

    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def seqid_cutoff(self) -> float:
        return self._seqid_cutoff

    @seqid_cutoff.setter
    def seqid_cutoff(self, value: float) -> None:
        assert 0 <= value <= 1
        if getattr(self, "_seqid_cutoff", None) != value:
            with contextlib.suppress(AttributeError):
                delattr(self, "_weights")
            with contextlib.suppress(AttributeError):
                delattr(self, "_neff")
        self._seqid_cutoff = value

    @property
    def is_covered(self) -> np.ndarray:
        if not hasattr(self, "_is_covered"):
            self._is_covered = (self[1:] != self.gap).any(0)
        return self._is_covered

    @property
    def coverage(self) -> float:
        if not hasattr(self, "_coverage"):
            notgap = self.array != self.gap
            self._coverage = notgap.mean(0)
        return self._coverage

    @property
    def weights(self) -> np.ndarray:
        if not hasattr(self, "_weights"):
            self._weights = 1 / (self.pdist() < self.seqid_cutoff).sum(1)
        return self._weights
    
    @property
    def has_full_length_sequences(self) -> bool:
        """Check if full-length sequences are available."""
        return self.full_length_sequences is not None
    
    @property
    def full_length_msa(self) -> Iterator[Tuple[str, str]]:
        """Return iterator over (header, full_length_sequence) tuples.
        
        Returns:
            Iterator of (header, sequence) tuples for full-length unaligned sequences
            
        Raises:
            ValueError: If full-length sequences are not available
        """
        if self.full_length_sequences is None:
            raise ValueError("Full-length sequences not available")
        return zip(self.headers, self.full_length_sequences)

    def neff(self, normalization: Union[float, str] = "none") -> float:
        if isinstance(normalization, str):
            assert normalization in ("none", "sqrt", "seqlen")
            normalization = {
                "none": 1,
                "sqrt": math.sqrt(self.seqlen),
                "seqlen": self.seqlen,
            }[normalization]
        if not hasattr(self, "_neff"):
            self._neff = self.weights.sum()
        return self._neff / normalization

    @classmethod
    def from_stockholm(
        cls,
        stofile: Union[PathLike, TextIO],
        keep_insertions: bool = False,
        includes_insertion_relative_to_query: bool = False,
        **kwargs,
    ) -> "MSA":

        output = []
        valid_indices = None
        for record in SeqIO.parse(stofile, "stockholm"):
            description = record.description
            sequence = str(record.seq)
            if not keep_insertions:
                if valid_indices is None:
                    valid_indices = [i for i, aa in enumerate(sequence) if aa != "-"]
                sequence = "".join(sequence[idx] for idx in valid_indices)
            output.append((description, sequence))
        return cls(
            output,
            includes_insertion_relative_to_query=includes_insertion_relative_to_query,
            **kwargs,
        )

    @classmethod
    def from_fasta(
        cls,
        fasfile: Union[PathLike, TextIO],
        keep_insertions: bool = False,
        uppercase: bool = False,
        remove_lowercase_cols: bool = False,
        includes_insertion_relative_to_query: bool = False,
        **kwargs,
    ) -> "MSA":

        output = []
        valid_indices = None
        for record in SeqIO.parse(fasfile, "fasta"):
            description = record.description
            sequence = str(record.seq)
            if remove_lowercase_cols:
                if valid_indices is None:
                    valid_indices = [i for i, aa in enumerate(sequence) if aa.isupper()]
                sequence = "".join(sequence[i] for i in valid_indices)
            if not keep_insertions:
                sequence = re.sub(r"([a-z]|\.|\*)", "", sequence)
            if uppercase:
                sequence = sequence.upper()
            output.append((description, sequence))
        return cls(
            output,
            includes_insertion_relative_to_query=includes_insertion_relative_to_query,
            **kwargs,
        )

    @classmethod
    def from_file(
        cls,
        alnfile: PathLike,
        format: Optional[str] = None,
        keep_insertions: bool = False,
        includes_insertion_relative_to_query: bool = False,
        **kwargs,
    ) -> "MSA":
        """Load MSA from file (supports both .fasta and .txt formats).
        
        Args:
            alnfile: Path to MSA file
            format: Explicit format ("fasta" or "txt"), auto-detect if None
            keep_insertions: Whether to keep insertion characters. If includes_insertion_relative_to_query=True,
                this will be automatically set to True to preserve original sequences.
            includes_insertion_relative_to_query: If True, lowercase letters are treated as 
                insertions relative to query. When True, automatically sets keep_insertions=True,
                remove_lowercase_cols=False, and uppercase=False to preserve original sequences.
        
        Returns:
            MSA object
        """
        filename = Path(alnfile)
        
        # Auto-detect format if not specified
        if format is None:
            if filename.suffix == ".sto":
                format = "stockholm"
            elif filename.suffix in (".fas", ".fasta", ".a3m", ".a2m"):
                format = "fasta"
            elif filename.suffix == ".txt":
                format = "txt"
            else:
                raise ValueError(f"Unknown file format {filename.suffix}")
        
        # When includes_insertion_relative_to_query=True, preserve original sequences
        if includes_insertion_relative_to_query:
            keep_insertions = True
            remove_lowercase_cols = False
            uppercase = False
        else:
            # Allow these to be passed via kwargs if needed
            remove_lowercase_cols = kwargs.get("remove_lowercase_cols", False)
            uppercase = kwargs.get("uppercase", False)
        
        if format == "stockholm":
            return cls.from_stockholm(
                filename,
                keep_insertions,
                includes_insertion_relative_to_query=includes_insertion_relative_to_query,
                **kwargs,
            )
        elif format == "fasta":
            return cls.from_fasta(
                filename,
                keep_insertions=keep_insertions,
                uppercase=uppercase,
                remove_lowercase_cols=remove_lowercase_cols,
                includes_insertion_relative_to_query=includes_insertion_relative_to_query,
                **kwargs,
            )
        elif format == "txt":
            return cls.from_fasta(
                filename,
                keep_insertions=keep_insertions,
                uppercase=uppercase,
                remove_lowercase_cols=remove_lowercase_cols,
                includes_insertion_relative_to_query=includes_insertion_relative_to_query,
                **kwargs,
            )  # TXT uses same format as FASTA
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def from_sequences(
        cls,
        sequences: Sequence[str],
        includes_insertion_relative_to_query: bool = False,
        **kwargs,
    ) -> "MSA":
        return cls(
            [("", seq) for seq in sequences],
            includes_insertion_relative_to_query=includes_insertion_relative_to_query,
            **kwargs,
        )

    def write(self, outfile: PathLike, format: Optional[str] = None) -> None:
        """Write MSA to file (supports both formats).
        
        Args:
            outfile: Path to output file
            format: Explicit format, auto-detect from extension if None
        """
        filename = Path(outfile)
        
        # Auto-detect format if not specified
        if format is None:
            if filename.suffix == ".sto":
                format = "stockholm"
            elif filename.suffix in (".fas", ".fasta", ".a3m", ".a2m"):
                format = "fasta"
            elif filename.suffix == ".txt":
                format = "fasta"  # TXT uses FASTA format structure
            else:
                format = "fasta"  # Default to FASTA
        
        SeqIO.write(
            (SeqIO.SeqRecord(Seq(seq), id=header, description="") for header, seq in self),
            outfile,
            format,
        )
    
    def write_full_length_sequences(
        self,
        outfile: PathLike,
        format: Optional[str] = None,
    ) -> None:
        """Write full-length unaligned sequences to file.
        
        Args:
            outfile: Path to output file
            format: Explicit format, auto-detect from extension if None
        
        Raises:
            ValueError: If full-length sequences not available
        """
        if self.full_length_sequences is None:
            raise ValueError("Full-length sequences not available")
        
        filename = Path(outfile)
        
        # Auto-detect format if not specified
        if format is None:
            if filename.suffix == ".sto":
                format = "stockholm"
            elif filename.suffix in (".fas", ".fasta", ".a3m", ".a2m"):
                format = "fasta"
            elif filename.suffix == ".txt":
                format = "fasta"  # TXT uses FASTA format structure
            else:
                format = "fasta"  # Default to FASTA
        
        SeqIO.write(
            (SeqIO.SeqRecord(Seq(seq), id=header, description="") for header, seq in self.full_length_msa),
            outfile,
            format,
        )
    
    def write_alignment_masks(
        self,
        filepath: PathLike,
        format: Optional[str] = None,
    ) -> None:
        """Write alignment masks to file (same format as MSA).
        
        Args:
            filepath: Path to output file
            format: Explicit format, auto-detect from extension if None
        
        Raises:
            ValueError: If alignment_masks not available
        """
        if not self.alignment_masks:
            raise ValueError("Alignment masks not available")
        
        filename = Path(filepath)
        
        # Auto-detect format if not specified
        if format is None:
            if filename.suffix == ".sto":
                format = "stockholm"
            elif filename.suffix in (".fas", ".fasta", ".a3m", ".a2m"):
                format = "fasta"
            elif filename.suffix == ".txt":
                format = "fasta"  # TXT uses FASTA format structure
            else:
                format = "fasta"  # Default to FASTA
        
        # Write masks in same format as MSA
        mask_sequences = [
            (header, self.alignment_masks.get(header, ""))
            for header in self.headers
        ]
        mask_msa = MSA(
            mask_sequences,
            includes_insertion_relative_to_query=False,  # Masks are always aligned
            seqid_cutoff=self.seqid_cutoff,
            already_processed=True,  # Masks are already strings, no processing needed
        )
        mask_msa.write(filepath, format=format)
