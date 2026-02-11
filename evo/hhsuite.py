"""Wrappers for HH-suite tools (hhblits, hhfilter).

Provides low-level binary wrappers and high-level search strategies for building
MSAs from query sequences using HH-suite databases.

Two search strategies are available (from the humanPPI paper):
- "uniref100": Single aggressive search (default). Fast, produces large MSAs.
- "iterative": Iterative search with increasing E-values and early stopping
  when enough sequences pass hhfilter thresholds. Slower but more controlled.

Ported from protevo-complex/ppievo/datasets/alignment.py
Cleaned up: separated binary wrappers from search strategies,
added skip_existing/protein_ids for resumable targeted runs.

Usage:
    from evo.hhsuite import search_hhblits, search_hhblits_batch

    # Single query
    output = search_hhblits(
        query_fasta=Path("query.fasta"),
        database=Path("/scr/hhsuite-databases/UniRef30_2023_02"),
        output_dir=Path("output/"),
    )

    # Batch (e.g., only missing proteins)
    results = search_hhblits_batch(
        query_fasta=Path("data/metadata/query_seqs_all.fasta"),
        database=Path("/scr/hhsuite-databases/UniRef30_2023_02"),
        output_dir=Path("data/hhblits/msa_unfiltered/"),
        protein_ids=["Q9BTK6", "Q9BTN0"],  # or None for all
        skip_existing=True,
    )
"""

import logging
import subprocess
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


# =============================================================================
# FASTA I/O helpers
# =============================================================================


def read_fasta(filename: str | Path) -> Generator[tuple[str, str], None, None]:
    """Read sequences from a FASTA file.

    Yields:
        (header, sequence) tuples. Headers do not include the '>' prefix.
    """
    with open(filename) as f:
        header, sequence = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and sequence:
                    yield (header, sequence)
                header, sequence = line[1:], ""
            else:
                sequence += line
        if header and sequence:
            yield (header, sequence)


def count_sequences(fasta_file: str | Path) -> int:
    """Count the number of sequences in a FASTA file."""
    count = 0
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def remove_descriptions(fasta_file: str | Path) -> None:
    """Strip description text from FASTA headers, keeping only the ID.

    Modifies the file in-place. For a header like ">ID description text",
    keeps only ">ID".
    """
    path = Path(fasta_file)
    lines = path.read_text().splitlines()
    cleaned = []
    for line in lines:
        if line.startswith(">"):
            cleaned.append(line.split()[0])
        else:
            cleaned.append(line)
    path.write_text("\n".join(cleaned) + "\n")


# =============================================================================
# Low-level binary wrappers
# =============================================================================


class HHBlits:
    """Wrapper around the hhblits binary.

    Args:
        database: Path to HH-suite database (e.g., UniRef30_2023_02).
        mact: Posterior prob threshold for MAC realignment (0=global, >0.1=local).
        maxfilt: Max hits allowed to pass 2nd prefilter.
        neffmax: Skip further iterations when diversity exceeds this.
        cpu: Number of CPUs to use.
        all_seqs: Show all sequences in result MSA (don't filter).
        realign_max: Max hits to realign.
        maxmem: Memory limit for realignment (GB).
        n: Number of search iterations (1-8).
        diff: Filter MSA by selecting most diverse set of sequences.
        evalue: E-value cutoff for inclusion in result alignment.
        binary: Path to hhblits binary.
    """

    def __init__(
        self,
        database: str | Path,
        *,
        mact: float = 0.35,
        maxfilt: int = 20000,
        neffmax: float = 20.0,
        cpu: int = 2,
        all_seqs: bool = False,
        realign_max: int = 500,
        maxmem: float = 3.0,
        n: int = 2,
        diff: int = 1000,
        evalue: float = 0.001,
        binary: str = "hhblits",
    ):
        self.database = str(database)
        self.mact = mact
        self.maxfilt = maxfilt
        self.neffmax = neffmax
        self.cpu = cpu
        self.all_seqs = all_seqs
        self.realign_max = realign_max
        self.maxmem = maxmem
        self.n = n
        self.diff = diff
        self.evalue = evalue
        self.binary = binary

    def _build_command(
        self,
        input_file: Path,
        output_a3m: Path,
        output_tab: Path | None = None,
        evalue: float | None = None,
    ) -> list[str]:
        """Build the hhblits command as a list of arguments."""
        ev = evalue if evalue is not None else self.evalue
        cmd = [
            self.binary,
            "-d", self.database,
            "-i", str(input_file),
            "-oa3m", str(output_a3m),
            "-e", str(ev),
            "-mact", str(self.mact),
            "-maxfilt", str(int(self.maxfilt)),
            "-neffmax", str(self.neffmax),
            "-cpu", str(self.cpu),
            "-realign_max", str(int(self.realign_max)),
            "-maxmem", str(self.maxmem),
            "-n", str(self.n),
            "-diff", str(self.diff),
            "-o", "/dev/null",
            "-v", "0",
        ]
        if output_tab is not None:
            cmd.extend(["-blasttab", str(output_tab)])
        if self.all_seqs:
            cmd.append("-all")
        return cmd

    def run(
        self,
        input_file: str | Path,
        output_prefix: str | Path | None = None,
        evalue: float | None = None,
    ) -> Path:
        """Run hhblits on a single query FASTA file.

        Args:
            input_file: Path to input FASTA (single or multi-sequence).
            output_prefix: Prefix for output files. If None, uses input_file stem.
                Output files will be {prefix}.a3m and {prefix}.tab.
            evalue: Override E-value for this run.

        Returns:
            Path to output .a3m file.

        Raises:
            subprocess.CalledProcessError: If hhblits fails.
        """
        input_file = Path(input_file)
        if output_prefix is None:
            output_prefix = input_file.parent / input_file.stem
        else:
            output_prefix = Path(output_prefix)

        # Use string concatenation — Path.with_suffix() mangles numeric dots
        # (e.g., Path(".q.0.001").with_suffix(".a3m") -> ".q.0.a3m")
        output_a3m = Path(str(output_prefix) + ".a3m")
        output_tab = Path(str(output_prefix) + ".tab")

        cmd = self._build_command(input_file, output_a3m, output_tab, evalue)
        logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        result.check_returncode()

        return output_a3m


class HHFilter:
    """Wrapper around the hhfilter binary.

    Args:
        seqid: Maximum pairwise sequence identity (%).
        diff: Min diverse sequences per MSA block of length 50.
        cov: Minimum coverage with query (%).
        qid: Minimum sequence identity with query (%).
        qsc: Minimum score per column with query.
        M: Match state definition ("a2m", "first", or percentage).
        maxseq: Max number of input rows.
        maxres: Max number of HMM columns.
        binary: Path to hhfilter binary.
    """

    def __init__(
        self,
        *,
        seqid: int = 90,
        diff: int = 0,
        cov: int = 0,
        qid: int = 0,
        qsc: float = -20.0,
        M: str = "a2m",
        maxseq: int = 65535,
        maxres: int = 20001,
        binary: str = "hhfilter",
    ):
        self.seqid = seqid
        self.diff = diff
        self.cov = cov
        self.qid = qid
        self.qsc = qsc
        self.M = M
        self.maxseq = maxseq
        self.maxres = maxres
        self.binary = binary

    def _build_command(
        self, input_file: Path, output_file: Path
    ) -> list[str]:
        """Build the hhfilter command as a list of arguments."""
        return [
            self.binary,
            "-i", str(input_file),
            "-o", str(output_file),
            "-id", str(self.seqid),
            "-diff", str(self.diff),
            "-cov", str(self.cov),
            "-qid", str(self.qid),
            "-qsc", str(self.qsc),
            "-M", self.M,
            "-maxseq", str(self.maxseq),
            "-maxres", str(self.maxres),
            "-v", "0",
        ]

    def run(self, input_file: str | Path, output_file: str | Path) -> Path:
        """Run hhfilter on an MSA file.

        Args:
            input_file: Path to input A3M/FASTA file.
            output_file: Path to write filtered output.

        Returns:
            Path to output file.

        Raises:
            subprocess.CalledProcessError: If hhfilter fails.
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(input_file, output_file)
        logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        result.check_returncode()

        return output_file


# =============================================================================
# Search strategies
# =============================================================================


def _search_uniref100(
    query_fasta: Path,
    output_dir: Path,
    database: str,
    *,
    evalue: float = 0.001,
    n_cpus: int = 32,
    keep_intermediates: bool = False,
    hhblits_binary: str = "hhblits",
) -> Path:
    """UniRef100 strategy: single aggressive hhblits search.

    From the humanPPI paper (Supplementary M4.6). Uses aggressive parameters
    (maxfilt=1e5, realign_max=1e4, n=4, all_seqs=True) for comprehensive results.

    Returns:
        Path to output .a3m file.
    """
    stem = query_fasta.stem
    output_a3m = output_dir / f"{stem}.a3m"

    if output_a3m.exists():
        logger.info(f"Output already exists: {output_a3m}")
        return output_a3m

    hhblits = HHBlits(
        database,
        mact=0.35,
        maxfilt=int(1e5),
        neffmax=20.0,
        cpu=n_cpus,
        all_seqs=True,
        realign_max=int(1e4),
        maxmem=64.0,
        n=4,
        diff=1000,
        evalue=evalue,
        binary=hhblits_binary,
    )

    # Run hhblits to intermediate path, then rename
    # Use string concatenation — Path.with_suffix() mangles numeric dots
    intermediate_prefix = output_dir / f".{stem}.{evalue}"
    intermediate_a3m = Path(str(intermediate_prefix) + ".a3m")
    intermediate_tab = Path(str(intermediate_prefix) + ".tab")

    try:
        hhblits.run(query_fasta, intermediate_prefix, evalue=evalue)

        # Move to final location
        intermediate_a3m.rename(output_a3m)
        output_tab = output_dir / f"{stem}.tab"
        if intermediate_tab.exists():
            intermediate_tab.rename(output_tab)

        # Clean up header descriptions
        remove_descriptions(output_a3m)

    finally:
        if not keep_intermediates:
            for f in [intermediate_a3m, intermediate_tab]:
                if f.exists():
                    f.unlink()

    return output_a3m


def _search_iterative(
    query_fasta: Path,
    output_dir: Path,
    database: str,
    *,
    metagenomic_database: str | None = None,
    evalues: list[float] | None = None,
    min_seqs_cov75: int = 2000,
    min_seqs_cov50: int = 5000,
    n_cpus: int = 20,
    keep_intermediates: bool = False,
    hhblits_binary: str = "hhblits",
    hhfilter_binary: str = "hhfilter",
) -> Path:
    """Iterative E-value strategy: search with increasing E-values, stop early.

    Iterates over E-values from strict to permissive. At each step:
    1. Run hhblits at current E-value
    2. Filter with hhfilter (id90, cov75) - stop if > min_seqs_cov75 sequences
    3. Filter with hhfilter (id90, cov50) - stop if > min_seqs_cov50 sequences
    4. Use filtered result as input for next iteration

    If primary database is exhausted, optionally searches metagenomic database.

    Returns:
        Path to output .a3m file.
    """
    stem = query_fasta.stem
    output_a3m = output_dir / f"{stem}.a3m"

    if output_a3m.exists():
        raise FileExistsError(f"{output_a3m} already exists!")

    if evalues is None:
        evalues = [1e-80, 1e-60, 1e-40, 1e-20, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-1]

    hhblits = HHBlits(
        database,
        mact=0.35,
        maxfilt=int(1e8),
        neffmax=20.0,
        cpu=n_cpus,
        all_seqs=True,
        realign_max=int(1e7),
        maxmem=64.0,
        n=4,
        binary=hhblits_binary,
    )
    hhfilter_cov75 = HHFilter(seqid=90, cov=75, binary=hhfilter_binary)
    hhfilter_cov50 = HHFilter(seqid=90, cov=50, binary=hhfilter_binary)

    intermediates: list[Path] = []

    def _run_evalue_sweep(
        db_hhblits: HHBlits, evalues: list[float], prev_a3m: Path, tag: str = ""
    ) -> bool:
        """Run E-value sweep. Returns True if early-stop threshold was met."""
        nonlocal prev_a3m_ref
        for ev in evalues:
            # Run hhblits — pass prefix (without .a3m), run() appends suffixes
            out_prefix = output_dir / f".{stem}.{ev}{tag}"
            out_a3m = Path(str(out_prefix) + ".a3m")
            if not out_a3m.exists():
                db_hhblits.run(prev_a3m_ref, out_prefix, evalue=ev)
            intermediates.append(out_a3m)

            # hhfilter id90 cov75
            cov75_path = Path(str(out_prefix) + ".id90cov75.a3m")
            intermediates.append(cov75_path)
            if not cov75_path.exists():
                hhfilter_cov75.run(out_a3m, cov75_path)
            if count_sequences(cov75_path) > min_seqs_cov75:
                cov75_path.rename(output_a3m)
                return True

            # hhfilter id90 cov50
            cov50_path = Path(str(out_prefix) + ".id90cov50.a3m")
            intermediates.append(cov50_path)
            if not cov50_path.exists():
                hhfilter_cov50.run(out_a3m, cov50_path)
            if count_sequences(cov50_path) > min_seqs_cov50:
                cov50_path.rename(output_a3m)
                return True

            prev_a3m_ref = cov50_path

        return False

    prev_a3m_ref = query_fasta

    try:
        found = _run_evalue_sweep(hhblits, evalues, prev_a3m_ref)

        # Try metagenomic database if primary didn't yield enough
        if not found and metagenomic_database is not None:
            meta_hhblits = HHBlits(
                metagenomic_database,
                mact=0.35,
                maxfilt=int(1e8),
                neffmax=20.0,
                cpu=n_cpus,
                all_seqs=True,
                realign_max=int(1e7),
                maxmem=64.0,
                n=4,
                binary=hhblits_binary,
            )
            found = _run_evalue_sweep(
                meta_hhblits, evalues, prev_a3m_ref, tag=".metagenomic"
            )

        # If still not found, use last filtered result
        if not found and not output_a3m.exists():
            # prev_a3m_ref points to the last cov50 filtered file
            if prev_a3m_ref.exists() and prev_a3m_ref != query_fasta:
                prev_a3m_ref.rename(output_a3m)

        if output_a3m.exists():
            remove_descriptions(output_a3m)

    finally:
        if not keep_intermediates:
            for f in intermediates:
                if f.exists():
                    f.unlink()

    return output_a3m


# =============================================================================
# Public API
# =============================================================================


def search_hhblits(
    query_fasta: str | Path,
    database: str | Path,
    output_dir: str | Path,
    *,
    strategy: str = "uniref100",
    n_cpus: int = 32,
    keep_intermediates: bool = False,
    evalue: float = 0.001,
    evalues: list[float] | None = None,
    min_seqs_cov75: int = 2000,
    min_seqs_cov50: int = 5000,
    metagenomic_database: str | Path | None = None,
    hhblits_binary: str = "hhblits",
    hhfilter_binary: str = "hhfilter",
) -> Path:
    """Run hhblits search for a single query sequence.

    Args:
        query_fasta: Path to input FASTA file (single sequence).
        database: Path to HH-suite database (e.g., UniRef30_2023_02).
        output_dir: Directory for output files.
        strategy: Search strategy - "uniref100" (default) or "iterative".
        n_cpus: Number of CPUs for hhblits.
        keep_intermediates: Keep intermediate files.
        evalue: E-value cutoff (uniref100 strategy).
        evalues: List of E-values to try (iterative strategy).
        min_seqs_cov75: Early stop threshold at cov75 (iterative strategy).
        min_seqs_cov50: Early stop threshold at cov50 (iterative strategy).
        metagenomic_database: Optional metagenomic database (iterative strategy).
        hhblits_binary: Path to hhblits binary.
        hhfilter_binary: Path to hhfilter binary.

    Returns:
        Path to output .a3m file.
    """
    query_fasta = Path(query_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if strategy == "uniref100":
        return _search_uniref100(
            query_fasta,
            output_dir,
            str(database),
            evalue=evalue,
            n_cpus=n_cpus,
            keep_intermediates=keep_intermediates,
            hhblits_binary=hhblits_binary,
        )
    elif strategy == "iterative":
        return _search_iterative(
            query_fasta,
            output_dir,
            str(database),
            metagenomic_database=str(metagenomic_database)
            if metagenomic_database
            else None,
            evalues=evalues,
            min_seqs_cov75=min_seqs_cov75,
            min_seqs_cov50=min_seqs_cov50,
            n_cpus=n_cpus,
            keep_intermediates=keep_intermediates,
            hhblits_binary=hhblits_binary,
            hhfilter_binary=hhfilter_binary,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'uniref100' or 'iterative'.")


def search_hhblits_batch(
    query_fasta: str | Path,
    database: str | Path,
    output_dir: str | Path,
    *,
    protein_ids: list[str] | None = None,
    skip_existing: bool = True,
    strategy: str = "uniref100",
    n_cpus: int = 32,
    **kwargs,
) -> dict[str, Path]:
    """Run hhblits for multiple query sequences from a multi-sequence FASTA.

    Splits the input FASTA into individual query files and runs hhblits on each.

    Args:
        query_fasta: Path to multi-sequence FASTA file.
        database: Path to HH-suite database.
        output_dir: Directory for output .a3m files.
        protein_ids: If provided, only process these protein IDs.
            Useful for running only missing proteins.
        skip_existing: Skip proteins with existing .a3m output files.
        strategy: Search strategy ("uniref100" or "iterative").
        n_cpus: Number of CPUs for each hhblits run.
        **kwargs: Additional arguments passed to search_hhblits().

    Returns:
        Dict mapping protein_id -> output .a3m path (only successful runs).
    """
    query_fasta = Path(query_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all sequences from input FASTA
    sequences = dict(read_fasta(query_fasta))
    logger.info(f"Loaded {len(sequences)} sequences from {query_fasta}")

    # Filter to requested protein IDs
    if protein_ids is not None:
        protein_set = set(protein_ids)
        missing_from_fasta = protein_set - set(sequences.keys())
        if missing_from_fasta:
            logger.warning(
                f"{len(missing_from_fasta)} requested proteins not found in FASTA: "
                f"{sorted(missing_from_fasta)[:5]}..."
            )
        sequences = {k: v for k, v in sequences.items() if k in protein_set}
        logger.info(f"Filtered to {len(sequences)} requested proteins")

    # Skip existing
    if skip_existing:
        before = len(sequences)
        sequences = {
            k: v
            for k, v in sequences.items()
            if not (output_dir / f"{k}.a3m").exists()
        }
        skipped = before - len(sequences)
        if skipped > 0:
            logger.info(f"Skipping {skipped} proteins with existing .a3m files")

    logger.info(f"Running hhblits for {len(sequences)} proteins")

    results: dict[str, Path] = {}
    for i, (seq_id, sequence) in enumerate(sequences.items()):
        logger.info(f"[{i + 1}/{len(sequences)}] Processing {seq_id}")

        # Write individual query FASTA
        query_file = output_dir / f"{seq_id}.fasta"
        query_file.write_text(f">{seq_id}\n{sequence}\n")

        try:
            output_path = search_hhblits(
                query_file,
                database,
                output_dir,
                strategy=strategy,
                n_cpus=n_cpus,
                **kwargs,
            )
            results[seq_id] = output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"hhblits failed for {seq_id}: {e}")
        finally:
            # Clean up temporary query file
            if query_file.exists():
                query_file.unlink()

    logger.info(
        f"Completed: {len(results)}/{len(sequences)} successful "
        f"({len(sequences) - len(results)} failed)"
    )
    return results
