"""Run reproducible H2 preprocessing and ."""
from pathlib import Path
from hotel_booking.workflow import run

if __name__ == "__main__":
    run(Path(__file__).resolve().parent)
