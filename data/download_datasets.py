#!/usr/bin/env python3
"""
Download public datasets used in the paper.

Usage:
    python data/download_datasets.py          # download all to data/
    python data/download_datasets.py --dest /path/to/dir

Some datasets require accepting terms on their hosting site. Where direct
download is not possible, this script prints instructions instead.
"""
from __future__ import annotations

import argparse
import io
import os
import urllib.request
import zipfile


def _download(url: str, dest: str, description: str) -> None:
    if os.path.exists(dest):
        print(f"  [skip] {description} already at {dest}")
        return
    print(f"  Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest} ({os.path.getsize(dest):,} bytes)")
    except Exception as e:
        print(f"  [FAIL] Could not download {description}: {e}")
        print(f"         URL: {url}")


def download_card(dest_dir: str) -> None:
    """Card (1995) schooling data. From Wooldridge textbook datasets."""
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Wage.csv"
    # The Card dataset is typically from the AER R package. A widely used CSV
    # mirror is on Vincent Arel-Bundock's Rdatasets collection.
    rdatasets_url = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Card.csv"
    _download(rdatasets_url, os.path.join(dest_dir, "data_card.csv"), "Card schooling (AER)")


def download_401k(dest_dir: str) -> None:
    """401(k) data from the AER R package (Rdatasets mirror)."""
    # This is often called the "k401k" dataset in Chernozhukov et al.
    # Available via Rdatasets.
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/PSID1976.csv"
    # The actual 401k dataset used in the IV literature is from Abadie (2003).
    # Try hdm R package mirror:
    print("  [NOTE] 401(k) data: The standard source is the hdm R package.")
    print("         Install R and run: data('pension', package='hdm')")
    print("         Then export to CSV. Alternatively, use the Wooldridge")
    print("         textbook version:")
    url_401k = "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/k401k.csv"
    _download(url_401k, os.path.join(dest_dir, "data_401k.csv"), "401(k) (wooldridge)")


def download_rhc(dest_dir: str) -> None:
    """Right Heart Catheterization data (Connors et al., 1996).

    The canonical source is the Vanderbilt Biostatistics datasets page.
    """
    url = "https://hbiostat.org/data/repo/rhc.csv"
    _download(url, os.path.join(dest_dir, "data_rhc.csv"), "RHC (Vanderbilt)")


def download_cattaneo(dest_dir: str) -> None:
    """Cattaneo (2010) smoking/birthweight data.

    Available from the causaldata R package or directly from Cattaneo's site.
    """
    url = "https://raw.githubusercontent.com/scunning1975/causal-inference-python/master/data/cattaneo2.csv"
    _download(url, os.path.join(dest_dir, "data_cattaneo.csv"), "Cattaneo birthweight")


def download_lalonde(dest_dir: str) -> None:
    """LaLonde NSW data. From the MatchIt R package via Rdatasets."""
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv"
    _download(url, os.path.join(dest_dir, "data_lalonde.csv"), "LaLonde NSW (MatchIt)")


def download_ihdp(dest_dir: str) -> None:
    """IHDP (Infant Health and Development Program) data.

    Commonly used benchmark from Hill (2011). Available from several
    GitHub repos that host causal inference benchmarks.
    """
    url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
    _download(url, os.path.join(dest_dir, "data_ihdp.csv"), "IHDP (npci)")


def download_acs_pums(dest_dir: str) -> None:
    """ACS PUMS 2022 (Census Bureau).

    The full microdata is very large. We provide instructions for
    downloading the 1-year person-level file.
    """
    dest = os.path.join(dest_dir, "acs_pums_2022.csv")
    if os.path.exists(dest):
        print(f"  [skip] ACS PUMS already at {dest}")
        return
    print("  [MANUAL] ACS PUMS 2022:")
    print("    1. Go to https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/")
    print("    2. Download csv_pus.zip for your state or csv_pall.zip for national")
    print("    3. Extract and preprocess to create acs_pums_2022.csv with columns:")
    print("       age_group, sex, race, edu, region")
    print("    Alternatively, use the Census API or folktables Python package:")
    print("       pip install folktables")
    print("       from folktables import ACSDataSource")


def download_cps_asec(dest_dir: str) -> None:
    """CPS ASEC 2022 (Census Bureau)."""
    dest = os.path.join(dest_dir, "cps_asec_2022.csv")
    if os.path.exists(dest):
        print(f"  [skip] CPS ASEC already at {dest}")
        return
    print("  [MANUAL] CPS ASEC 2022:")
    print("    1. Go to https://www.census.gov/data/datasets/2022/demo/cps/cps-asec-2022.html")
    print("    2. Download the CSV version of the person file")
    print("    3. Save as cps_asec_2022.csv in the data directory")
    print("    Alternatively, use IPUMS CPS: https://cps.ipums.org/cps/")


def download_brfss(dest_dir: str) -> None:
    """BRFSS 2022 (CDC). The raw SAS transport file is ~300MB."""
    dest = os.path.join(dest_dir, "brfss_2022_raw.csv")
    if os.path.exists(dest):
        print(f"  [skip] BRFSS already at {dest}")
        return
    print("  [MANUAL] BRFSS 2022:")
    print("    1. Go to https://www.cdc.gov/brfss/annual_data/annual_2022.html")
    print("    2. Download the SAS Transport Format data file (LLCP2022.XPT)")
    print("    3. Convert to CSV. In Python:")
    print("       import pandas as pd")
    print("       df = pd.read_sas('LLCP2022.XPT', format='xport')")
    print("       df.to_csv('brfss_2022_raw.csv', index=False)")


def download_ces(dest_dir: str) -> None:
    """CES 2022 (Conference Board).

    The Consumer Expenditure Survey microdata is not freely available
    from the Conference Board. The BLS Consumer Expenditure Survey is
    a possible alternative.
    """
    dest = os.path.join(dest_dir, "ces_2022.csv")
    if os.path.exists(dest):
        print(f"  [skip] CES already at {dest}")
        return
    print("  [MANUAL] CES 2022:")
    print("    The Consumer Expenditure Survey data can be obtained from:")
    print("    https://www.bls.gov/cex/pumd_data.htm")
    print("    Download the Interview Survey public-use microdata and extract")
    print("    demographic variables: age_group, sex, race, edu, region")


def download_gss(dest_dir: str) -> None:
    """GSS Cumulative data (NORC/GSS)."""
    dest = os.path.join(dest_dir, "gss_cumulative.csv")
    if os.path.exists(dest):
        print(f"  [skip] GSS already at {dest}")
        return
    print("  [MANUAL] GSS Cumulative:")
    print("    1. Go to https://gss.norc.org/get-the-data/stata")
    print("    2. Download the cumulative cross-sectional data file")
    print("    3. Convert to CSV with columns: age, sex, race, educ, region, income")
    print("    Alternatively, use the GSS Data Explorer:")
    print("    https://gssdataexplorer.norc.org/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for paper experiments")
    parser.add_argument(
        "--dest",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__))),
        help="Destination directory for downloaded files",
    )
    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    print(f"Downloading datasets to {args.dest}\n")

    print("=== Directly downloadable datasets ===\n")

    print("[1/11] Card schooling data")
    download_card(args.dest)

    print("\n[2/11] 401(k) data")
    download_401k(args.dest)

    print("\n[3/11] RHC data")
    download_rhc(args.dest)

    print("\n[4/11] Cattaneo birthweight data")
    download_cattaneo(args.dest)

    print("\n[5/11] LaLonde NSW data")
    download_lalonde(args.dest)

    print("\n[6/11] IHDP data")
    download_ihdp(args.dest)

    print("\n=== Datasets requiring manual download ===\n")

    print("[7/11] ACS PUMS 2022")
    download_acs_pums(args.dest)

    print("\n[8/11] CPS ASEC 2022")
    download_cps_asec(args.dest)

    print("\n[9/11] BRFSS 2022")
    download_brfss(args.dest)

    print("\n[10/11] CES 2022")
    download_ces(args.dest)

    print("\n[11/11] GSS Cumulative")
    download_gss(args.dest)

    print("\n" + "=" * 60)
    print("Done. Some datasets require manual download (see [MANUAL] notes above).")
    print("After downloading, place the CSV files in:", args.dest)


if __name__ == "__main__":
    main()
