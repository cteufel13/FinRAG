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

    return row.replace('","', "and for").replace(":", " is").replace('"', "")


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
