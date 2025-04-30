import pandas as pd
from typing import List, Dict, Any


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
    Each key-value pair is formatted as 'key: value'.
    """

    return row.replace('","', "and for").replace(":", " is").replace('"', "")
