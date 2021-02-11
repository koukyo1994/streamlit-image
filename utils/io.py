import pandas as pd
import streamlit as st

from pathlib import Path
from PIL import Image


@st.cache
def read_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


@st.cache
def read_image(path: Path) -> Image:
    return Image.open(path)


def write_image_info_to_sidebar(path: Path, image: Image):
    st.sidebar.subheader(f"Image file: {path.name}")
    st.sidebar.markdown("#### Basic info")
    st.sidebar.text(f"shape: {image.size}")
    width, height = image.size
    if width == 2144 and height == 1424:
        camera = "C1"
    elif width == 4288 and height == 2848:
        camera = "C2"
    elif width == 2048 and height == 1536:
        camera = "C3"
    else:
        camera = "Unknown"
    st.sidebar.text(f"camera: {camera}")


def check_folder(folder: str):
    path = Path(folder)
    if not path.exists():
        st.warning("specified folder does not exist")
        return
    else:
        pngs = list(path.glob("*.png"))
        jpgs = list(path.glob("*.jpg"))
        jpegs = list(path.glob("*.jpeg"))
        subdirs = [
            subpaths for subpaths in path.glob("*") if subpaths.is_dir()
        ]

        if len(pngs) > 0:
            st.success(f"Found {len(pngs)} png files")
            return path
        if len(jpgs) > 0:
            st.success(f"Found {len(jpgs)} jpg files")
            return path
        if len(jpegs) > 0:
            st.success(f"Found {len(jpegs)} jpeg files")
            return path
        if len(subdirs) == 0:
            st.warning("No image files found under the directory you specified")
            return
        else:
            subdir_names = sorted([subdir.name for subdir in subdirs])
            subfolder = st.selectbox(f"Pick one folder below {str(folder)}",
                                     options=subdir_names,
                                     key=f"{str(folder)}")
            new_folder = path / subfolder
            return check_folder(str(new_folder))
