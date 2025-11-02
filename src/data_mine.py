from pathlib import Path
import pandas as pd
import sys

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "secom.data"
LABELS_FILE = DATA_DIR / "secom_labels.data"
NAMES_FILE = DATA_DIR / "secom.names"

def print_head(path, n=5):
    try:
        print(f"\n--- First {n} lines of {path} ---")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i in range(n):
                line = f.readline()
                if not line:
                    break
                print(repr(line.rstrip("\n\r")))
    except Exception as e:
        print(f"Could not print head of {path}: {e}")

def try_read_sensor(path):
    """Try several reading strategies and return (df, method_used) or (None, None)."""
    patterns = []

    # prefer raw regex sep
    patterns.append(("sep_regex_raw", {"sep": r"\s+", "header": None, "engine": "python"}))
    # escaped sep
    patterns.append(("sep_regex_escaped", {"sep": "\\s+", "header": None, "engine": "python"}))
    # delim_whitespace (fast)
    patterns.append(("delim_whitespace", {"delim_whitespace": True, "header": None}))
    # read_table fallback
    patterns.append(("read_table", {"sep": None, "header": None, "engine": "python"}))

    last_exc = None
    for name, kwargs in patterns:
        try:
            print(f"\nAttempting read with method: {name} and kwargs: {kwargs}")
            df = pd.read_csv(path, **kwargs)
            # drop all-empty trailing columns
            df = df.dropna(axis=1, how="all")
            print(f"Success using {name}: shape={df.shape}")
            return df, name
        except Exception as e:
            last_exc = e
            print(f"Failed {name}: {e}")

    print(f"\nAll read attempts failed. Last error: {last_exc}")
    return None, None

def try_read_labels(path):
    try:
        # common format: label timestamp (space separated)
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        if df.shape[1] >= 1:
            # ensure first column is labels
            df = df.iloc[:, :2] if df.shape[1] >= 2 else df.iloc[:, :1]
            return df
    except Exception as e:
        print(f"labels read attempt failed with regex sep: {e}")

    try:
        df = pd.read_csv(path, header=None)
        return df
    except Exception as e:
        print(f"labels read fallback failed: {e}")
    return None

def parse_names(path):
    """Return a cleaned list of lines from names file (skip comment lines)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Could not read names file: {e}")
        return []

    lines = []
    for ln in text.splitlines():
        ln_strip = ln.strip()
        if not ln_strip:
            continue
        # skip common comment/section markers
        if ln_strip.startswith("|") or ln_strip.startswith("#") or ln_strip.lower().startswith("attribute") or "class" in ln_strip.lower():
            continue
        # try splitting comma separated lists into items
        if "," in ln_strip:
            parts = [p.strip() for p in ln_strip.split(",") if p.strip()]
            lines.extend(parts)
            continue
        # otherwise accept short tokens (likely names) or lines like "sensor_1 : numeric"
        if ":" in ln_strip:
            left = ln_strip.split(":", 1)[0].strip()
            if left:
                lines.append(left)
                continue
        # accept token if plausible (no long descriptions)
        if len(ln_strip.split()) <= 3 and len(ln_strip) < 100:
            lines.append(ln_strip)
    # dedupe while preserving order
    seen = set()
    out = []
    for n in lines:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def main():
    # check files
    for p in (DATA_FILE, LABELS_FILE, NAMES_FILE):
        print(f"Checking: {p} -> exists: {p.exists()}")

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Put secom.data inside the 'data' folder.")
        sys.exit(1)
    if not LABELS_FILE.exists():
        print(f"ERROR: {LABELS_FILE} not found. Put secom_labels.data inside the 'data' folder.")
        sys.exit(1)

    # print first lines to help debug separators/encoding
    print_head(DATA_FILE, n=6)
    print_head(LABELS_FILE, n=6)
    if NAMES_FILE.exists():
        print_head(NAMES_FILE, n=12)

    # try to read sensor data
    df_sensor, method = try_read_sensor(DATA_FILE)
    if df_sensor is None:
        print("Failed to load sensor data with multiple strategies. Exiting.")
        sys.exit(1)

    # try to read labels
    df_labels = try_read_labels(LABELS_FILE)
    if df_labels is None:
        print("Failed to load labels file.")
        sys.exit(1)

    # clean labels: take first column as label
    label_series = df_labels.iloc[:, 0].rename("Label").reset_index(drop=True)

    # try parse names
    parsed = parse_names(NAMES_FILE) if NAMES_FILE.exists() else []
    print(f"\nParsed {len(parsed)} items from names file.")

    # assign columns if counts match (or sensible fallback)
    if len(parsed) == df_sensor.shape[1]:
        df_sensor.columns = parsed
        print("Assigned parsed names exactly to columns.")
    else:
        if len(parsed) > 0:
            print("Parsed names count differs from columns. Using parsed names for first columns and generic for remaining.")
            cols = [f"Sensor_{i+1}" for i in range(df_sensor.shape[1])]
            for i, nm in enumerate(parsed[: df_sensor.shape[1]]):
                cols[i] = nm
            df_sensor.columns = cols
        else:
            df_sensor.columns = [f"Sensor_{i+1}" for i in range(df_sensor.shape[1])]
            print("No parsed names available - assigned generic Sensor_i names.")

    # align lengths: if label length mismatch, warn
    if len(label_series) != len(df_sensor):
        print(f"WARNING: label length ({len(label_series)}) != sensor rows ({len(df_sensor)}).")
        # If labels are fewer/exact multiple - try to align by trimming/padding
        min_len = min(len(label_series), len(df_sensor))
        print(f"Truncating to minimum length: {min_len}")
        df_sensor = df_sensor.iloc[:min_len].reset_index(drop=True)
        label_series = label_series.iloc[:min_len].reset_index(drop=True)

    combined = pd.concat([df_sensor.reset_index(drop=True), label_series], axis=1)
    print("\nCombined shape:", combined.shape)
    print(combined.head())
    # Save for convenience
    out_path = DATA_DIR / "secom_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined CSV to: {out_path}")

if __name__ == "__main__":
    main()
