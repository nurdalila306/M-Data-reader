import io
import os
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

# --------------------------------------------------------------------------------------
# ------------------------------ Parsing Utilities -------------------------------------
# --------------------------------------------------------------------------------------

def parse_w_filename(name):
    """
    Parse filename of form: WYYMMDDHHMM.STATION  (e.g., W2303240010.TRE)
    Returns (year, month, day, hour, minute, station_id) or None if invalid.
    """
    base = os.path.basename(name)
    if not base.startswith("W") or "." not in base:
        return None
    stem, ext = base.split(".", 1)
    if len(stem) != 11:
        return None
    try:
        yy = int(stem[1:3])
        mm = int(stem[3:5])
        dd = int(stem[5:7])
        hh = int(stem[7:9])
        mi = int(stem[9:11])
    except ValueError:
        return None
    year = 2000 + yy if yy < 70 else 1900 + yy  # crude century rule; adjust if needed
    station_id = ext.upper()
    return year, mm, dd, hh, mi, station_id


def load_w_block(file_like):
    """
    Read binary 32-bit float triplets -> Nx3 array.
    """
    try:
        buf = np.frombuffer(file_like.read(), dtype=np.float32)
        if buf.size % 3 != 0:
            # pad if corrupted
            pad = 3 - (buf.size % 3)
            buf = np.pad(buf, (0, pad), mode="constant", constant_values=np.nan)
        return buf.reshape(-1, 3)
    except Exception:
        return np.empty((0, 3), dtype=np.float32)


def read_wdata_from_mapping(year, month, day, station_id, file_map):
    """
    Reimplementation of read_wdata_v1p1() but using an in-memory mapping:
        file_map[(year, month, day, hour, minute, station_id)] -> BytesIO
    Returns (ut1s, mag) same as original:
        ut1s: np.array of POSIX timestamps at 1-sec cadence (86400 length)
        mag:  (86400, 4) -> H,D,Z,F
    Missing files -> NaN.
    """
    start_time = datetime(year, month, day)
    end_time = start_time + timedelta(days=1)
    ut1s = np.arange(int(start_time.timestamp()), int(end_time.timestamp()), 1, dtype=np.int64)
    mag = np.full((86400, 4), np.nan, dtype=np.float32)

    for hour in range(24):
        for minute in range(0, 60, 10):
            key = (year, month, day, hour, minute, station_id)
            fbytes = file_map.get(key)
            if fbytes is None:
                continue
            fbytes.seek(0)
            buf = load_w_block(fbytes)
            if buf.size == 0:
                continue
            spos = hour * 3600 + minute * 60
            epos = min(spos + buf.shape[0], 86400)
            mag[spos:epos, :3] = buf[:epos - spos, :]
            mag[spos:epos, 3] = np.sqrt(np.nansum(buf[:epos - spos, :]**2, axis=1))

    # spike cleaner (same as yours; adjust threshold if needed)
    mag[np.abs(mag) > 100000] = 99999.99
    return ut1s, mag


def resample_resolution(ut1s, mag, resolution):
    """
    Downsample to MIN (1-min) or keep SEC.
    """
    if resolution == "SEC":
        return ut1s, mag
    elif resolution == "MIN":
        # take every 60th sample (simple decimation)
        return ut1s[::60], mag[::60, :]
    else:
        raise ValueError("resolution must be 'SEC' or 'MIN'.")


def to_dataframe(ut1s, mag, station_id):
    timestamps = pd.to_datetime(ut1s, unit="s", utc=True)
    df = pd.DataFrame(mag, columns=["H (nT)", "D (nT)", "Z (nT)", "F (nT)"])
    df.insert(0, "Timestamp", timestamps)
    df["Station"] = station_id
    return df


# --------------------------------------------------------------------------------------
# ------------------------------ Upload Handling ---------------------------------------
# --------------------------------------------------------------------------------------

def ingest_uploaded_files(uploaded_files):
    """
    Accepts list of UploadedFile objects (Streamlit).
    Returns:
      file_map: {(year,month,day,hour,minute,station_id): BytesIO}
      dates_available: {(year,month,day): set(station_ids)}
    """
    file_map = {}
    dates_available = defaultdict(set)

    for uf in uploaded_files:
        parsed = parse_w_filename(uf.name)
        if parsed is None:
            continue
        year, mm, dd, hh, mi, station_id = parsed
        bio = io.BytesIO(uf.read())
        file_map[(year, mm, dd, hh, mi, station_id)] = bio
        dates_available[(year, mm, dd)].add(station_id)

    return file_map, dates_available


def ingest_zip(uploaded_zip):
    """
    Accept a single UploadedFile that is a ZIP. Extract in memory and return parsed file_map.
    """
    zbio = io.BytesIO(uploaded_zip.read())
    file_map = {}
    dates_available = defaultdict(set)

    with zipfile.ZipFile(zbio) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = os.path.basename(info.filename)
            parsed = parse_w_filename(base)
            if parsed is None:
                continue
            year, mm, dd, hh, mi, station_id = parsed
            with zf.open(info) as f:
                bio = io.BytesIO(f.read())
            file_map[(year, mm, dd, hh, mi, station_id)] = bio
            dates_available[(year, mm, dd)].add(station_id)
    return file_map, dates_available



# --------------------------------------------------------------------------------------
# ------------------------------ Plot Helpers ------------------------------------------
# --------------------------------------------------------------------------------------

def plot_components(df, title_prefix=""):
    """
    Plot H,D,Z,F vs time using Matplotlib with different colors.
    """
    time = df["Timestamp"]
    comps = ["H (nT)", "D (nT)", "Z (nT)", "F (nT)"]
    colors = ["blue", "red", "green", "black"]

    for i, c in enumerate(comps):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(time, df[c], color=colors[i], label=c)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("nT")
        ax.set_title(f"{title_prefix}{c}")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)



def plot_differences(df, title_prefix=""):
    """
    Plot 1st differences of each component with different colors.
    """
    comps = ["H (nT)", "D (nT)", "Z (nT)", "F (nT)"]
    colors = ["blue", "red", "green", "black"]
    ddf = df[comps].diff()
    time = df["Timestamp"]

    for i, c in enumerate(comps):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(time, ddf[c], color=colors[i], label=f"Δ{c.split()[0]}")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("ΔnT")
        ax.set_title(f"{title_prefix}Δ{c.split()[0]}")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)



# --------------------------------------------------------------------------------------
# ------------------------------ Streamlit UI ------------------------------------------
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="W-Format MAGDAS Viewer", layout="wide")
st.title("W-Format MAGDAS Data Viewer / Converter")

st.markdown("""
**Instructions:**
1. Upload either:
   - A **ZIP** of your station folder (easiest), **or**
   - **Multiple W-files** selected together.
2. Choose date, station, and resolution.
3. View plots, download CSV / NPY, or bulk-convert all.
""")

tab_zip, tab_multi = st.tabs(["Upload ZIP Folder", "Upload Multiple W Files"])

all_file_map = {}
all_dates_available = defaultdict(set)

with tab_zip:
    upzip = st.file_uploader("Upload a ZIP containing W-files (subfolders OK)", type=["zip"])
    if upzip is not None:
        f_map, d_avail = ingest_zip(upzip)
        all_file_map.update(f_map)
        for k, v in d_avail.items():
            all_dates_available[k].update(v)
        st.success(f"Loaded {len(f_map)} W-blocks from ZIP.")

with tab_multi:
    upfiles = st.file_uploader("Upload one or more W-files", accept_multiple_files=True)
    if upfiles:
        f_map, d_avail = ingest_uploaded_files(upfiles)
        all_file_map.update(f_map)
        for k, v in d_avail.items():
            all_dates_available[k].update(v)
        st.success(f"Loaded {len(f_map)} W-blocks from uploaded files.")

if not all_file_map:
    st.info("Upload files to begin.")
    st.stop()

# Build selection widgets from available dates/stations
dates_sorted = sorted(all_dates_available.keys())
date_labels = [f"{y}-{m:02d}-{d:02d}" for (y, m, d) in dates_sorted]
date_choice = st.selectbox("Select Date", options=date_labels)
sel_idx = date_labels.index(date_choice)
sel_year, sel_month, sel_day = dates_sorted[sel_idx]

stations_for_date = sorted(all_dates_available[(sel_year, sel_month, sel_day)])
station_choice = st.selectbox("Select Station", options=stations_for_date)

resolution_choice = st.radio("Resolution", options=["SEC", "MIN", "BOTH"], horizontal=True)


if st.button("Process Selected Date"):
    resolutions = ["SEC", "MIN"] if resolution_choice == "BOTH" else [resolution_choice]

    for res in resolutions:
        st.markdown(f"### 📉 Resolution: {res}")

        ut1s, mag = read_wdata_from_mapping(sel_year, sel_month, sel_day, station_choice, all_file_map)
        ut1s_r, mag_r = resample_resolution(ut1s, mag, res)
        df = to_dataframe(ut1s_r, mag_r, station_choice)

        # Preview table
        st.subheader(f"Preview Data ({res})")
        st.dataframe(df.head(20))

        # Download CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label=f"Download CSV ({res})",
            data=csv_buffer.getvalue(),
            file_name=f"{station_choice}_{res}_{sel_year}{sel_month:02d}{sel_day:02d}.csv",
            mime="text/csv"
        )



        # Plots
        st.subheader(f"Magnetic Components ({res})")
        plot_components(df, title_prefix=f"{station_choice} {sel_year}-{sel_month:02d}-{sel_day:02d} ")

        st.subheader(f"Component Differences ({res})")
        plot_differences(df, title_prefix=f"{station_choice} {sel_year}-{sel_month:02d}-{sel_day:02d} ")



# --------------------------------------------------------------------------------------
# ----------------------- Bulk Conversion (all uploaded dates) -------------------------
# --------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("Bulk Convert All Uploaded Data to CSV (per date+station+resolution)")

bulk_res_choice = st.radio("Bulk Resolution", options=["SEC", "MIN", "BOTH"], horizontal=True, key="bulkres")


if st.button("Run Bulk Conversion & Download ZIP"):
    bulk_res_list = ["SEC", "MIN"] if bulk_res_choice == "BOTH" else [bulk_res_choice]

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for (y, m, d), stations in all_dates_available.items():
            for stn in stations:
                ut1s, mag = read_wdata_from_mapping(y, m, d, stn, all_file_map)
                for res in bulk_res_list:
                    ut1s_r, mag_r = resample_resolution(ut1s, mag, res)
                    df = to_dataframe(ut1s_r, mag_r, stn)

                    # CSV
                    csv_str = df.to_csv(index=False)
                    csv_name = f"{stn}/{res}/{y}/{stn}_{res}_{y}{m:02d}{d:02d}.csv"
                    z.writestr(csv_name, csv_str)

    zbuf.seek(0)
    st.download_button(
        "Download All Converted Data (ZIP)",
        data=zbuf,
        file_name="converted_magdas_data.zip",
        mime="application/zip"
    )
