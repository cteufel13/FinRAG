import pandas as pd
from typing import List, Dict, Any
import re


def parse_html_table(page_content: str) -> List[Dict[str, str]]:
    """
    Parses a single HTML <table> from the given page_content and returns
    a list of row-dictionaries.
    Column headers are taken from the table's first row.
    All cell values are converted to strings.
    """
    # Read the first table found in the HTML
    dfs = pd.read_html(page_content)
    if not dfs:
        return []
    df = dfs[0]

    # Ensure column names are strings
    df.columns = df.columns.map(str)
    # Replace NaN with empty string and cast all values to str
    df = df.fillna("").astype(str)

    # Convert each row to a dict mapping header -> cell text
    return df.to_dict(orient="records")


def convert_table_row_to_text(row: str) -> str:
    """
    Converts a table row (dict) to a string representation.
    """

    return (
        row.replace('","', "and for")
        .replace('"', "")
        .replace("Unnamed: 0", "")
        .replace("Unnamed", "")
        .replace(" : ", "")
        .replace("0:", "")
        .replace("1:", "")
        .replace("2:", "")
        .replace("3:", "")
        .replace("4:", "")
        .replace("5:", "")
    )


def convert_strings_to_floats(string_list):
    """
    Converts a list of strings with various formats into float numbers.
    Handles formats like "14.3 million", "37%", "40", "15300 cubic meters", etc.

    Args:
        string_list (list): List of strings to convert

    Returns:
        list: List of converted float values
    """
    result = []

    for string in string_list:
        # Remove any commas and convert to lowercase
        clean_string = string.replace(",", "").lower().strip()

        # Extract the numeric part first
        numeric_part = ""
        for char in clean_string:
            if char.isdigit() or char == "." or char == "-":
                numeric_part += char
            elif numeric_part:  # Stop when we hit non-numeric after getting some digits
                break

        # If no numeric part found, skip this string
        if not numeric_part:
            result.append(None)
            continue

        # Convert the numeric part to float
        try:
            value = float(numeric_part)
        except ValueError:
            result.append(None)
            continue

        # Check for multipliers and adjustments
        if "million" in clean_string or "m" == clean_string.split()[-1]:
            value *= 1_000_000
        elif "billion" in clean_string or "b" == clean_string.split()[-1]:
            value *= 1_000_000_000
        elif "thousand" in clean_string or "k" == clean_string.split()[-1]:
            value *= 1_000

        # Check for percentage
        if "%" in clean_string:
            value /= 100

        # Units like "cubic meters" don't affect the numeric value,
        # so we don't need special handling for them

        result.append(value)

    return result


import re
import pandas as pd
from camelot.io import read_pdf


def clean_tables(tables: List) -> List[pd.DataFrame]:

    cleaned = []

    for tbl in tables:
        try:
            df = clean_camelot_df(tbl.df)
            cleaned.append(df)
        except ValueError:
            # could not parse this chunk as a table → skip or log
            continue

    return cleaned


def clean_camelot_df(
    raw_df,
    min_numeric_cols: int = 2,
    footnote_pattern: str = r"^(Notes?:|•|\d+\s[–-]\s)",
    combine_header_rows: bool = True,
):
    """
    Take a raw Camelot table (tbl.df) and:
      - normalize newlines → spaces
      - find the “header” row by looking for the first row
        with >= min_numeric_cols truly-numeric cells (e.g. years)
      - optionally flatten two rows of headers into one
      - drop any rows that look like footnotes
      - strip strings, convert numeric columns

    Returns a cleaned DataFrame.
    """
    # 1) Normalize whitespace/newlines
    df = raw_df.replace(r"\n", " ", regex=True).map(
        lambda x: x.strip() if isinstance(x, str) else x
    )

    # 2) Detect header row: look for row with at least N numeric cells
    def is_numeric(s):
        try:
            float(s.replace(",", ""))
            return True
        except:
            return False

    header_idx = None
    for i, row in df.iterrows():
        if sum(is_numeric(cell) for cell in row) >= min_numeric_cols:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Couldn't detect header row automatically.")

    # 3) Build column names
    if combine_header_rows and header_idx > 0:
        # assume row above + this row form a multi-row header
        hdr0 = df.iloc[header_idx - 1].fillna("").astype(str)
        hdr1 = df.iloc[header_idx].fillna("").astype(str)
        cols = [(h0 + " " + h1).strip() for h0, h1 in zip(hdr0, hdr1)]
        start_data = header_idx + 1
    else:
        cols = df.iloc[header_idx].astype(str).tolist()
        start_data = header_idx + 1

    df = df.iloc[start_data:].copy()
    df.columns = pd.Index(cols).str.replace(r"\s+", " ", regex=True).str.strip()

    # 4) Drop footnote/Notes rows
    fn_re = re.compile(footnote_pattern, re.IGNORECASE)
    mask = df.iloc[:, 0].astype(str).apply(lambda x: not bool(fn_re.match(x)))
    df = df[mask]

    # 5) Attempt conversion of numeric columns
    for col in df.columns:
        # if a majority of values look numeric, convert
        vals = df[col].astype(str)
        if sum(is_numeric(v) for v in vals if v not in ["N/A", ""]) > len(vals) / 2:
            df[col] = (
                vals.str.replace(",", "", regex=False)
                .replace({"N/A": pd.NA})
                .astype("float", errors="ignore")
            )

    # Final clean: reset index
    return df.reset_index(drop=True)
