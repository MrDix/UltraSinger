# Suppress noisy third-party warnings at import time.
import warnings

warnings.filterwarnings("ignore", module="requests")
warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", message="In 2\\.9.*torchaudio")
