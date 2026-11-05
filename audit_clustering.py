"""Run the training-only K-Means count and scientific-integrity audit."""
from pathlib import Path

from hotel_booking.clustering_audit import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
