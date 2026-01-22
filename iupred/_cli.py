"""Command-line interface for IUPred and AIUPred predictions."""

import sys
import logging
from pathlib import Path
import argparse

from .iupred2a import iupred, anchor2


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def _read_fasta(path: Path) -> dict[str, str]:
    """Read sequences from a FASTA file.

    Args:
        path: Path to FASTA file.

    Returns:
        Dictionary mapping headers to sequences.
    """
    sequences = {}
    header = None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:].split()[0]  # Take first word as ID
                sequences[header] = ''
            elif header and line:
                sequences[header] += line.upper()
    return sequences


def _format_output(
    header: str,
    sequence: str,
    scores: list,
    score_name: str = 'score',
    binding_scores: list = None,
) -> str:
    """Format prediction results as tab-separated output.

    Args:
        header: Sequence identifier.
        sequence: Amino acid sequence.
        scores: Primary prediction scores.
        score_name: Name for the score column.
        binding_scores: Optional binding scores.

    Returns:
        Formatted string with results.
    """
    lines = [f'# {header}']

    if binding_scores is not None:
        lines.append(f'# pos\taa\t{score_name}\tbinding')
        for i, (aa, score, bind) in enumerate(
            zip(sequence, scores, binding_scores), 1
        ):
            lines.append(f'{i}\t{aa}\t{score:.4f}\t{bind:.4f}')
    else:
        lines.append(f'# pos\taa\t{score_name}')
        for i, (aa, score) in enumerate(zip(sequence, scores), 1):
            lines.append(f'{i}\t{aa}\t{score:.4f}')

    return '\n'.join(lines)


def run_iupred2(args: argparse.Namespace) -> int:
    """Run IUPred2/ANCHOR2 prediction.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    _setup_logging(args.verbose)

    sequences = _read_fasta(args.input_file)
    if not sequences:
        logging.error('No sequences found in input file')
        return 1

    logging.info(f'Read {len(sequences)} sequences')

    output_lines = []
    for header, sequence in sequences.items():
        logging.debug(f'Processing {header} ({len(sequence)} residues)')

        disorder_scores, glob_text = iupred(sequence, mode=args.mode)

        if args.anchor:
            anchor_scores = anchor2(sequence, disorder_scores)
            output = _format_output(
                header,
                sequence,
                disorder_scores,
                score_name='iupred',
                binding_scores=anchor_scores,
            )
        else:
            output = _format_output(
                header, sequence, disorder_scores, score_name='iupred'
            )

        output_lines.append(output)

        if args.mode == 'glob' and glob_text:
            output_lines.append(f'# Globular domains:\n{glob_text}')

    result = '\n\n'.join(output_lines)

    if args.output_file:
        Path(args.output_file).write_text(result + '\n')
        logging.info(f'Results written to {args.output_file}')
    else:
        print(result)

    return 0


def run_aiupred(args: argparse.Namespace) -> int:
    """Run AIUPred prediction.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    # Import AIUPred functions lazily (requires torch)
    from .aiupred import (
        init_models,
        predict_binding,
        predict_disorder,
        low_memory_predict_binding,
        low_memory_predict_disorder,
    )

    _setup_logging(args.verbose)

    sequences = _read_fasta(args.input_file)
    if not sequences:
        logging.error('No sequences found in input file')
        return 1

    logging.info(f'Read {len(sequences)} sequences')

    # Initialize models once for efficiency
    embedding_model, reg_model, device = init_models(
        'disorder', force_cpu=args.force_cpu, gpu_num=args.gpu
    )

    binding_embedding = None
    binding_reg = None
    if args.binding:
        binding_embedding, binding_reg, _ = init_models(
            'binding', force_cpu=args.force_cpu, gpu_num=args.gpu
        )

    output_lines = []
    for header, sequence in sequences.items():
        logging.debug(f'Processing {header} ({len(sequence)} residues)')

        # Choose prediction function based on low-memory flag
        if args.low_memory:
            disorder_scores = low_memory_predict_disorder(
                sequence,
                embedding_model,
                reg_model,
                device,
                smoothing=True,
                chunk_len=args.chunk_size,
            )
        else:
            disorder_scores = predict_disorder(
                sequence, embedding_model, reg_model, device, smoothing=True
            )

        binding_scores = None
        if args.binding:
            if args.low_memory:
                binding_scores = low_memory_predict_binding(
                    sequence,
                    binding_embedding,
                    binding_reg,
                    device,
                    smoothing=True,
                    chunk_len=args.chunk_size,
                )
            else:
                binding_scores = predict_binding(
                    sequence,
                    binding_embedding,
                    binding_reg,
                    device,
                    binding=True,
                    smoothing=True,
                )

        output = _format_output(
            header,
            sequence,
            disorder_scores,
            score_name='aiupred',
            binding_scores=binding_scores,
        )
        output_lines.append(output)

    result = '\n\n'.join(output_lines)

    if args.output_file:
        Path(args.output_file).write_text(result + '\n')
        logging.info(f'Results written to {args.output_file}')
    else:
        print(result)

    return 0


def main(argv: list[str] = None) -> int:
    """Main entry point for the iupred CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        prog='iupred',
        description=(
            'Predict intrinsically disordered protein regions using '
            'IUPred2/ANCHOR2 or AIUPred methods.\n\n'
            'Developed by Zsuzsanna Dosztányi and Gábor Erdős (ELTE, Budapest)'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title='methods',
        description='Available prediction methods',
        dest='method',
        required=True,
    )

    # IUPred2 subcommand
    iupred_parser = subparsers.add_parser(
        'iupred2',
        help='IUPred2/ANCHOR2 energy-based disorder prediction',
        description=(
            'Predict disordered regions using IUPred2 energy estimation. '
            'Optionally predict binding regions with ANCHOR2.'
        ),
    )
    iupred_parser.add_argument(
        '-i',
        '--input-file',
        type=Path,
        required=True,
        help='Input FASTA file with protein sequences',
    )
    iupred_parser.add_argument(
        '-o',
        '--output-file',
        type=Path,
        help='Output file (default: stdout)',
    )
    iupred_parser.add_argument(
        '-m',
        '--mode',
        choices=['short', 'long', 'glob'],
        default='long',
        help='Prediction mode: short, long, or glob (default: long)',
    )
    iupred_parser.add_argument(
        '-a',
        '--anchor',
        action='store_true',
        help='Also predict binding regions with ANCHOR2',
    )
    iupred_parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output',
    )
    iupred_parser.set_defaults(func=run_iupred2)

    # AIUPred subcommand
    aiupred_parser = subparsers.add_parser(
        'aiupred',
        help='AIUPred deep learning disorder prediction',
        description=(
            'Predict disordered regions using AIUPred transformer networks. '
            'Model weights are downloaded automatically on first use.'
        ),
    )
    aiupred_parser.add_argument(
        '-i',
        '--input-file',
        type=Path,
        required=True,
        help='Input FASTA file with protein sequences',
    )
    aiupred_parser.add_argument(
        '-o',
        '--output-file',
        type=Path,
        help='Output file (default: stdout)',
    )
    aiupred_parser.add_argument(
        '-b',
        '--binding',
        action='store_true',
        help='Also predict binding regions with AIUPred-binding',
    )
    aiupred_parser.add_argument(
        '-g',
        '--gpu',
        type=int,
        default=0,
        help='GPU index to use (default: 0)',
    )
    aiupred_parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU-only mode (slower but works without GPU)',
    )
    aiupred_parser.add_argument(
        '--low-memory',
        action='store_true',
        help='Use low-memory mode for long sequences',
    )
    aiupred_parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size for low-memory mode (default: 1000)',
    )
    aiupred_parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output',
    )
    aiupred_parser.set_defaults(func=run_aiupred)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
