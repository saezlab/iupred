"""AIUPred: Transformer-based prediction of intrinsically disordered regions."""

import logging
import math
import os
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import pad

warnings.filterwarnings("ignore")

# Package and cache directories
PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / 'data'
CACHE_DIR = Path.home() / '.cache' / 'iupred'

# Model file URLs (to be configured)
# These will be set to actual URLs when hosting is configured
MODEL_URLS = {
    'embedding_disorder.pt': None,  # URL placeholder
    'disorder_decoder.pt': None,  # URL placeholder
    'embedding_binding.pt': None,  # URL placeholder
    'binding_decoder.pt': None,  # URL placeholder
    'binding_transform': None,  # URL placeholder
}

AA_CODE = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X',
]
WINDOW = 100


def _ensure_data_dir():
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_model_path(filename):
    """Get the path to a model file, downloading if necessary.

    Args:
        filename: Name of the model file.

    Returns:
        Path to the model file.

    Raises:
        FileNotFoundError: If the model is not available and URL is not configured.
    """
    # First check package data directory
    package_path = DATA_DIR / filename
    if package_path.exists():
        return package_path

    # Then check cache directory
    cache_path = CACHE_DIR / filename
    if cache_path.exists():
        return cache_path

    # Try to download
    url = MODEL_URLS.get(filename)
    if url is None:
        raise FileNotFoundError(
            f"Model file '{filename}' not found. "
            f"AIUPred model weights must be downloaded separately. "
            f"Please visit https://iupred.elte.hu for more information."
        )

    _ensure_data_dir()
    logging.info(f"Downloading {filename} from {url}...")
    urllib.request.urlretrieve(url, cache_path)
    logging.info(f"Downloaded {filename} to {cache_path}")
    return cache_path


def configure_model_urls(
    embedding_disorder=None,
    disorder_decoder=None,
    embedding_binding=None,
    binding_decoder=None,
    binding_transform=None,
):
    """Configure URLs for downloading AIUPred model weights.

    Args:
        embedding_disorder: URL for embedding_disorder.pt
        disorder_decoder: URL for disorder_decoder.pt
        embedding_binding: URL for embedding_binding.pt
        binding_decoder: URL for binding_decoder.pt
        binding_transform: URL for binding_transform
    """
    if embedding_disorder:
        MODEL_URLS['embedding_disorder.pt'] = embedding_disorder
    if disorder_decoder:
        MODEL_URLS['disorder_decoder.pt'] = disorder_decoder
    if embedding_binding:
        MODEL_URLS['embedding_binding.pt'] = embedding_binding
    if binding_decoder:
        MODEL_URLS['binding_decoder.pt'] = binding_decoder
    if binding_transform:
        MODEL_URLS['binding_transform'] = binding_transform


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer network
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer model to estimate positional contact potential from an amino acid sequence
    """

    def __init__(self):
        super().__init__()
        self.d_model = 32
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, 2, 256, 0)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.encoder = nn.Embedding(21, self.d_model)
        self.decoder = nn.Linear((WINDOW + 1) * self.d_model, 1)

    def forward(self, src: Tensor, embed_only=False) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)  # (Batch x Window+1 x Embed_dim)
        embedding = self.transformer_encoder(src)
        if embed_only:
            return embedding
        output = torch.flatten(embedding, 1)
        output = self.decoder(output)
        return torch.squeeze(output)


class BindingTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = 32
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, 2, 64, 0, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.encoder = nn.Embedding(21, self.d_model)
        self.decoder = nn.Linear((WINDOW + 1) * self.d_model, 1)

    def forward(self, src: Tensor, embed_only=False) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)  # (Batch x Window+1 x Embed_dim
        embedding = self.transformer_encoder(src)
        if embed_only:
            return embedding
        output = torch.flatten(embedding, 1)
        output = self.decoder(output)
        return torch.squeeze(output)


class BindingDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = (WINDOW + 1) * (WINDOW + 1) * 32
        output_dim = 1
        current_dim = input_dim
        layer_architecture = [64, 64, 64, 16]
        self.layers = nn.ModuleList()
        for hdim in layer_architecture:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        output = torch.sigmoid(self.layers[-1](x))
        return torch.squeeze(output)


class DecoderModel(nn.Module):
    """
    Regression model to estimate disorder propensity from and energy tensor
    """

    def __init__(self):
        super().__init__()
        input_dim = WINDOW + 1
        output_dim = 1
        current_dim = input_dim
        layer_architecture = [16, 8, 4]
        self.layers = nn.ModuleList()
        for hdim in layer_architecture:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        output = torch.sigmoid(self.layers[-1](x))
        return torch.squeeze(output)


@torch.no_grad()
def tokenize(sequence, device):
    """
    Tokenize an amino acid sequence. Non-standard amino acids are treated as X
    :param sequence: Amino acid sequence in string
    :param device: Device to run on. CUDA{x} or CPU
    :return: Tokenized tensors
    """
    return torch.tensor([AA_CODE.index(aa) if aa in AA_CODE else 20 for aa in sequence], device=device)


def predict_disorder(sequence, energy_model, regression_model, device, smoothing=None):
    """
    Predict disorder propensity from a sequence using a transformer and a regression model
    :param sequence: Amino acid sequence in string
    :param energy_model: Transformer model
    :param regression_model: regression model
    :param device: Device to run on. CUDA{x} or CPU
    :param smoothing: Use the SavGol filter to smooth the output
    :return:
    """
    predicted_energies = calculate_energy(sequence, energy_model, device)
    padded_energies = pad(predicted_energies, (WINDOW // 2, WINDOW // 2), 'constant', 0)
    unfolded_energies = padded_energies.unfold(0, WINDOW + 1, 1)
    predicted_disorder = regression_model(unfolded_energies).detach().cpu().numpy()
    if smoothing and len(sequence) > 10:
        predicted_disorder = savgol_filter(predicted_disorder, 11, 5)
    return predicted_disorder


def calculate_energy(sequence, energy_model, device):
    """
    Calculates residue energy from a sequence using a transformer network
    :param sequence: Amino acid sequence in string
    :param energy_model: Transformer model
    :param device: Device to run on. CUDA{x} or CPU
    :return: Tensor of energy values
    """
    tokenized_sequence = tokenize(sequence, device)
    padded_token = pad(tokenized_sequence, (WINDOW // 2, WINDOW // 2), 'constant', 20)
    unfolded_tokens = padded_token.unfold(0, WINDOW + 1, 1)
    return energy_model(unfolded_tokens)


def predict_binding(sequence, embedding_model, decoder_model, device, smoothing=None, energy_only=False, binding=False):
    _tokens = tokenize(sequence, device)
    _padded_token = pad(_tokens, (WINDOW // 2, WINDOW // 2), 'constant', 0)
    _unfolded_tokes = _padded_token.unfold(0, WINDOW + 1, 1)
    if energy_only:
        return embedding_model(_unfolded_tokes).detach().cpu().numpy()
    _token_embedding = embedding_model(_unfolded_tokes, embed_only=True)
    _padded_embed = pad(_token_embedding, (0, 0, 0, 0, WINDOW // 2, WINDOW // 2), 'constant', 0)
    _unfolded_embedding = _padded_embed.unfold(0, WINDOW + 1, 1)
    _decoder_input = _unfolded_embedding.permute(0, 2, 1, 3)
    _prediction = decoder_model(_decoder_input).detach().cpu().numpy()
    if binding:
        return binding_transform(_prediction, smoothing=smoothing)
    if not smoothing or len(sequence) <= 10:
        return _prediction
    return savgol_filter(transformed_pred, 11, 5)


def low_memory_predict_disorder(sequence, embedding_model, decoder_model, device, smoothing=None, chunk_len=1000):
    overlap = 100
    if (len(sequence) - 1) % (chunk_len - overlap) == 0:
        logging.warning('Chunk length decreased by 1 to fit sequence length')
        chunk_len -= 1
    if chunk_len <= overlap:
        raise ValueError("Chunk len must be bigger than 200!")
    overlapping_predictions = []
    for chunk in range(0, len(sequence), chunk_len - overlap):
        overlapping_predictions.append(predict_disorder(
            sequence[chunk:chunk + chunk_len],
            embedding_model,
            decoder_model,
            device,
        ))
    prediction = np.concatenate((overlapping_predictions[0], *[x[overlap:] for x in overlapping_predictions[1:]]))
    if not smoothing or len(sequence) <= 10:
        return prediction
    return savgol_filter(transformed_pred, 11, 5)


def low_memory_predict_binding(sequence, embedding_model, decoder_model, device, smoothing=None, chunk_len=1000):
    overlap = 100
    if chunk_len <= overlap:
        raise ValueError("Chunk len must be bigger than 200!")
    overlapping_predictions = []
    for chunk in range(0, len(sequence), chunk_len - overlap):
        overlapping_predictions.append(predict_binding(
            sequence[chunk:chunk + chunk_len],
            embedding_model,
            decoder_model,
            device
        ))
    prediction = np.concatenate((overlapping_predictions[0], *[x[overlap:] for x in overlapping_predictions[1:]]))
    return binding_transform(prediction, smoothing=smoothing)
    # return transformed_pred


def binding_transform(prediction, smoothing=True):
    transform = {}
    binding_transform_path = _get_model_path('binding_transform')
    with open(binding_transform_path) as fn:
        for line in fn:
            key, value = line.strip().split()
            transform[int(float(key) * 1000)] = float(value)
    rounded_pred = np.rint(prediction * 1000)
    transformed_pred = np.vectorize(transform.get)(rounded_pred)
    if not smoothing:
        return transformed_pred
    pred = savgol_filter(transformed_pred, 11, 5)
    pred[pred > 1] = 1
    return pred


def multifasta_reader(file_handler):
    """
    (multi) FASTA reader function
    :return: Dictionary with header -> sequence mapping from the file
    """
    sequence_dct = {}
    header = None
    for line in file_handler:
        if line.startswith('>'):
            header = line.strip()
            sequence_dct[header] = ''
        elif line.strip():
            sequence_dct[header] += line.strip()
    file_handler.close()
    return sequence_dct


def init_models(prediction_type, force_cpu=False, gpu_num=0):
    """Initialize networks and device to run on.

    Args:
        prediction_type: Type of prediction ('disorder' or 'binding').
        force_cpu: Force the method to run on CPU only mode.
        gpu_num: Index of the GPU to use, default=0.

    Returns:
        Tuple of (embedding_model, regression_model, device).
    """
    device = torch.device(
        f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    )
    if force_cpu:
        device = 'cpu'
    logging.debug(f'Running on {device}')
    if device == 'cpu':
        logging.warning(
            'No GPU found, running on CPU. '
            'It is advised to run AIUPred on a GPU.'
        )

    if prediction_type == 'disorder':
        embedding_model = TransformerModel()
        embedding_path = _get_model_path('embedding_disorder.pt')
        embedding_model.load_state_dict(
            torch.load(embedding_path, map_location=device)
        )
        reg_model = DecoderModel()
        decoder_path = _get_model_path('disorder_decoder.pt')
        reg_model.load_state_dict(
            torch.load(decoder_path, map_location=device)
        )
    else:
        embedding_model = BindingTransformerModel()
        embedding_path = _get_model_path('embedding_binding.pt')
        embedding_model.load_state_dict(
            torch.load(embedding_path, map_location=device)
        )
        reg_model = BindingDecoderModel()
        decoder_path = _get_model_path('binding_decoder.pt')
        reg_model.load_state_dict(
            torch.load(decoder_path, map_location=device)
        )

    embedding_model.to(device)
    embedding_model.eval()

    reg_model.to(device)
    reg_model.eval()
    logging.debug("Networks initialized")
    return embedding_model, reg_model, device


def aiupred_disorder(sequence, force_cpu=False, gpu_num=0):
    """
    Library function to carry out single sequence analysis
    :param sequence: Amino acid sequence in a string
    """
    embedding_model, reg_model, device = init_models('disorder', force_cpu, gpu_num)
    return predict_disorder(sequence, embedding_model, reg_model, device, smoothing=True)


def aiupred_binding(sequence, force_cpu=False, gpu_num=0):
    """
    Library function to carry out single sequence analysis
    :param sequence: Amino acid sequence in a string
    """
    embedding_model, reg_model, device = init_models('binding', force_cpu, gpu_num)
    return predict_binding(sequence, embedding_model, reg_model, device, binding=True, smoothing=True)


def main(multifasta_file, force_cpu=False, gpu_num=0, binding=False):
    """
    Main function to be called from aiupred.py
    :param multifasta_file: Location of (multi) FASTA formatted sequences
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Dictionary with parsed sequences and predicted results
    """
    embedding_model, reg_model, device = init_models('disorder', force_cpu, gpu_num)
    sequences = multifasta_reader(multifasta_file)
    logging.debug("Sequences read")
    logging.info(f'{len(sequences)} sequences read')
    if not sequences:
        raise ValueError("FASTA file is empty")
    results = {}
    logging.StreamHandler.terminator = ""
    for num, (ident, sequence) in enumerate(sequences.items()):
        results[ident] = {}
        results[ident]['aiupred'] = predict_disorder(sequence, embedding_model, reg_model, device, smoothing=True)
        if binding:
            binding_embedding, binding_regression, _ = init_models('binding', force_cpu, gpu_num)
            results[ident]['aiupred-binding'] = predict_binding(sequence, binding_embedding, binding_regression, device,
                                                                binding=binding,
                                                                smoothing=True)
        results[ident]['sequence'] = sequence
        logging.debug(f'{num}/{len(sequences)} sequences done...\r')
    logging.StreamHandler.terminator = '\n'
    logging.debug(f'Analysis done, writing output')
    return results
