#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2026/3/22 18:11
# @Author  : david_van
# @Desc    : simple HDF5 reader for rqalpha bundle files

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


DEFAULT_H5_PATH = Path("D:/datas/bundle/funds.h5")


def _preview_dataset(dataset: h5py.Dataset, preview_rows: int, from_tail: bool) -> None:
    print(f"shape: {dataset.shape}")
    print(f"dtype: {dataset.dtype}")

    data = dataset[()]
    print(f"loaded python type: {type(data)}")

    preview_label = "last" if from_tail else "first"

    def _preview_slice(values):
        if from_tail:
            return values[-preview_rows:]
        return values[:preview_rows]

    if getattr(data, "dtype", None) is not None and data.dtype.names:
        print(f"fields: {data.dtype.names}")
        print(f"{preview_label} {preview_rows} rows:")
        print(_preview_slice(data))
        return

    if hasattr(data, "__getitem__") and not isinstance(data, (bytes, str)):
        print(f"{preview_label} {preview_rows} items:")
        print(_preview_slice(data))
        return

    print("value:")
    print(data)


def read_h5(
    file_path: Path,
    key_name: str | None = None,
    preview_rows: int = 5,
    from_tail: bool = False,
) -> None:
    if not file_path.exists():
        print(f"h5 file not exist: {file_path}")
        return

    with h5py.File(str(file_path), "r") as h5_file:
        keys = list(h5_file.keys())
        print(f"file: {file_path}")
        print(f"top-level key count: {len(keys)}")
        print(f"sample keys: {keys[:10]}")

        if not key_name:
            print("no key specified, only listed top-level keys")
            return

        if key_name not in h5_file:
            print(f"key not found: {key_name}")
            return

        obj = h5_file[key_name]
        print(f"object name: {obj.name}")
        print(f"object type: {type(obj)}")

        if isinstance(obj, h5py.Dataset):
            _preview_dataset(obj, preview_rows, from_tail)
            return

        if isinstance(obj, h5py.Group):
            child_keys = list(obj.keys())
            print(f"group keys: {child_keys}")
            return

        print("unsupported HDF5 object type")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a key from an HDF5 bundle file")
    parser.add_argument(
        "--file",
        default=str(DEFAULT_H5_PATH),
        help="Path to the HDF5 file, defaults to funds.h5",
    )
    parser.add_argument(
        "--key",
        help="Top-level HDF5 key to inspect, for example 510050.XSHG",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="How many rows/items to preview for datasets",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Preview the last N rows/items instead of the first N",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    args.key = '515880.XSHG'
    args.tail = True
    args.preview_rows = 100
    read_h5(Path(args.file), args.key, max(1, args.preview_rows), args.tail)
