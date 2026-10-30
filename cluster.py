"""Run the reproducible Milestone 1 clustering experiment."""
from pathlib import Path
from hotel_booking.clustering import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
