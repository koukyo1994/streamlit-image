import streamlit as st

import utils


if __name__ == "__main__":
    st.title("Image Checking Tool")

    base_folder = st.text_input("Specify directory which contains image file.")
    uploaded = st.file_uploader("Choose a csv file:", type="csv")
    if uploaded is not None:
        dataframe = utils.read_csv(uploaded)
        st.write(dataframe)

    path = utils.check_folder(base_folder)
    if path is not None:
        image_files = sorted([
            f.name for f in (list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg")))
        ], key=lambda x: int(x.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")))
        image_filename = st.selectbox("Choose Image File", options=image_files)
        image_id = image_filename.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
        if uploaded is not None:
            annotation = dataframe.query(f"ID == {int(image_id)}").reset_index(drop=True)
            st.text("Found Corresponding Label")
            st.dataframe(annotation)

        image_path = path / image_filename
        image = utils.read_image(image_path)
        caption = f"Filename: {image_id}"
        if uploaded is not None:
            columns = annotation.columns[1:]
            for c in columns:
                if annotation.loc[0, c] == 1:
                    caption += f", {c}"
        st.image(image, caption=caption, use_column_width=True)
        utils.write_image_info_to_sidebar(image_path, image)

        options = st.sidebar.selectbox("Image option",
                                       options=["normal", "augmentations"])
