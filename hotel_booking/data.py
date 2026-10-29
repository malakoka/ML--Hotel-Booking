"""Source schema, lossless parsing, semantic missing values, and quality flags."""
import hashlib
import numpy as np
import pandas as pd

SOURCE_URL = "https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv"
ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/"


SCHEMA = {
    "IsCanceled": ("binary target", "Final cancellation indicator: 1 canceled, 0 otherwise"),
    "LeadTime": ("count", "Days between booking creation and planned arrival"),
    "ArrivalDateYear": ("calendar", "Planned arrival year"),
    "ArrivalDateMonth": ("category", "Planned arrival month name"),
    "ArrivalDateWeekNumber": ("calendar", "Planned arrival week number"),
    "ArrivalDateDayOfMonth": ("calendar", "Planned arrival day of month"),
    "StaysInWeekendNights": ("count", "Booked/stayed weekend nights"),
    "StaysInWeekNights": ("count", "Booked/stayed weekday nights"),
    "Adults": ("count", "Adult guest count"), "Children": ("count", "Child guest count"),
    "Babies": ("count", "Baby guest count"), "Meal": ("category", "Meal package code"),
    "Country": ("category", "Country of origin; may only be confirmed at check-in"),
    "MarketSegment": ("category", "Booking market segment"),
    "DistributionChannel": ("category", "Booking distribution channel"),
    "IsRepeatedGuest": ("binary", "Guest profile existed before this booking"),
    "PreviousCancellations": ("count", "Earlier canceled bookings linked to the guest"),
    "PreviousBookingsNotCanceled": ("count", "Earlier non-canceled bookings linked to the guest"),
    "ReservedRoomType": ("category", "Reserved room-type code"),
    "AssignedRoomType": ("category", "Assigned room-type code"),
    "BookingChanges": ("count", "Amendments accumulated before check-in/cancellation"),
    "DepositType": ("category", "Deposit category engineered from payment transactions"),
    "Agent": ("identifier", "Travel-agency code; NULL means no agency applies"),
    "Company": ("identifier", "Booking/paying company code; NULL means not applicable"),
    "DaysInWaitingList": ("count", "Waiting time until booking confirmation"),
    "CustomerType": ("category", "Contract, group, transient, or transient-party booking"),
    "ADR": ("numeric target", "Recorded average daily rate; not total booking revenue"),
    "RequiredCarParkingSpaces": ("count", "Requested parking spaces"),
    "TotalOfSpecialRequests": ("count", "Recorded number of special requests"),
    "ReservationStatus": ("outcome", "Final reservation status"),
    "ReservationStatusDate": ("date", "Date the final reservation status was set"),
}
TEXT = [c for c, (kind, _) in SCHEMA.items()
        if kind in {"category", "identifier", "outcome", "date"}]
NUMERIC = [c for c in SCHEMA if c not in TEXT]
MONTHS = {m: i + 1 for i, m in enumerate([
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"])}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonicalize(raw):
    """Normalize source formatting. Keep unknown Country/Children genuinely missing."""
    data = raw.copy()
    for col in data:
        data[col] = data[col].astype("string").str.strip()
    for col, label in [("Agent", "NoAgent"), ("Company", "NoCompany")]:
        # The research article explicitly defines these NULLs as not applicable.
        data[col] = data[col].replace({"NULL": label})
    for col in data:
        data[col] = data[col].replace({"": pd.NA, "NULL": pd.NA, "NA": pd.NA})
    for col in NUMERIC:
        original = data[col]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"Unexpected non-numerical tokens in {col}.")
        data[col] = numeric.astype(float)
    # sklearn imputers accept np.nan; object strings are stable across pandas releases.
    for col in TEXT:
        data[col] = data[col].astype(object).where(data[col].notna(), np.nan)
    return data


def arrival_dates(data):
    return pd.to_datetime(dict(year=data["ArrivalDateYear"],
                               month=data["ArrivalDateMonth"].map(MONTHS),
                               day=data["ArrivalDateDayOfMonth"]), errors="coerce")


def read_source(path):
    raw = pd.read_csv(path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    if list(raw.columns) != list(SCHEMA) or raw.shape != (79330, 31):
        raise ValueError("Expected the original H2 schema: 79,330 rows and 31 ordered columns.")
    return raw, canonicalize(raw)


def quality_flags(data):
    flags = pd.DataFrame(index=data.index)
    count_cols = [c for c, (kind, _) in SCHEMA.items() if kind == "count"]
    counts = data[count_cols]
    flags["invalid_negative_count"] = counts.lt(0).any(axis=1)
    flags["invalid_fractional_count"] = (counts.notna() & counts.mod(1).ne(0)).any(axis=1)
    flags["invalid_binary"] = ~data["IsCanceled"].isin([0, 1]) | ~data["IsRepeatedGuest"].isin([0, 1])
    flags["invalid_arrival_date"] = arrival_dates(data).isna()
    flags["invalid_required_numeric"] = data[[c for c in NUMERIC if c != "Children"]].isna().any(axis=1)
    flags["invalid_nonfinite"] = np.isinf(data[NUMERIC].to_numpy(dtype=float)).any(axis=1)
    flags["invalid_negative_adr"] = data["ADR"].lt(0)
    total_nights = data[["StaysInWeekendNights", "StaysInWeekNights"]].sum(axis=1, min_count=2)
    total_guests = data[["Adults", "Children", "Babies"]].sum(axis=1, min_count=3)
    flags["review_zero_nights"] = total_nights.eq(0)
    flags["review_zero_guests"] = total_guests.eq(0)
    flags["review_zero_adr"] = data["ADR"].eq(0)
    flags["review_adr_above_1000"] = data["ADR"].gt(1000)
    flags["review_babies_above_2"] = data["Babies"].gt(2)
    # Review thresholds flag records; none of these review flags removes a record.
    return flags
