       
from hashlib import sha256
from io import BytesIO
from math import log10
import csv
import io

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st

Image.MAX_IMAGE_PIXELS = 36_000_000

st.set_page_config(
    page_title="Image Forensics Analyzer",
    page_icon="🔎",
    layout="wide"
)


# ----------------------------------------
# IMAGE PROCESSING FUNCTIONS
# ----------------------------------------

def image_bytes(image, format="PNG", **options):
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer, format=format, **options
    )
    return buffer.getvalue()


def load_image(data):
    with Image.open(BytesIO(data)) as source:
        if source.format not in ("PNG", "JPEG", "WEBP"):
            raise ValueError(
                "Please upload PNG, JPG, or WebP images."
            )

        source.load()

        return ImageOps.exif_transpose(
            source
        ).convert("RGB")


# ----------------------------------------
# DIFFERENCE HASH
# ----------------------------------------

def dhash(image):
    """
    Calculate a 64-bit difference hash.
    Similar images tend to have similar hashes.
    """

    pixels = np.asarray(
        image.convert("L").resize((9, 8)),
        dtype=np.int16
    )

    bits = (
        pixels[:, :-1] > pixels[:, 1:]
    ).flatten()

    value = 0

    for bit in bits:
        value = (value << 1) | int(bit)

    return value


# ----------------------------------------
# COLOR HISTOGRAM
# ----------------------------------------

def histogram(image):
    """
    Extract normalized RGB color features.
    """

    pixels = np.asarray(
        image.resize((128, 128)),
        dtype=np.uint8
    )

    chunks = [
        np.histogram(
            pixels[:, :, i],
            bins=16,
            range=(0, 256)
        )[0]
        for i in range(3)
    ]

    vector = np.concatenate(
        chunks
    ).astype(float)

    norm = np.linalg.norm(vector)

    return vector / norm if norm else vector


# ----------------------------------------
# CREATE IMAGE RECORD
# ----------------------------------------

def record(name, data):
    image = load_image(data)

    return {
        "name": name,
        "image": image,
        "sha256": sha256(data).hexdigest(),
        "dhash": dhash(image),
        "hist": histogram(image)
    }


# ----------------------------------------
# IMAGE SIMILARITY
# ----------------------------------------

def compare(query, candidate):

    distance = (
        query["dhash"] ^ candidate["dhash"]
    ).bit_count()

    hash_similarity = 1 - distance / 64

    color_similarity = float(
        np.clip(
            np.dot(
                query["hist"],
                candidate["hist"]
            ),
            0,
            1
        )
    )

    exact = (
        query["sha256"] == candidate["sha256"]
    )

    # Ranking heuristic, not a probability.
    score = (
        1.0 if exact
        else 0.75 * hash_similarity
        + 0.25 * color_similarity
    )

    return {
        "Image": candidate["name"],
        "Similarity score": round(score, 4),
        "dHash distance": distance,
        "Color similarity": round(
            color_similarity, 4
        ),
        "Exact file copy": exact
    }


# ----------------------------------------
# PIXEL ERROR AND PSNR
# ----------------------------------------

def mse_psnr(image_a, image_b):

    a = np.asarray(
        image_a.convert("RGB"),
        dtype=np.float32
    )

    b = np.asarray(
        image_b.convert("RGB"),
        dtype=np.float32
    )

    if a.shape != b.shape:
        raise ValueError(
            "Images must have equal dimensions."
        )

    mse = float(
        np.mean((a - b) ** 2)
    )

    psnr = (
        float("inf")
        if mse == 0
        else 10 * log10(255 ** 2 / mse)
    )

    return mse, psnr


# ----------------------------------------
# FORENSIC IMAGE TRANSFORMATIONS
# ----------------------------------------

def transformations(image, sigma):

    rng = np.random.default_rng(42)

    array = np.asarray(
        image,
        dtype=np.float32
    )

    noisy = np.clip(
        array + rng.normal(
            0, sigma, array.shape
        ),
        0,
        255
    ).astype("uint8")

    half = image.resize((
        max(1, image.width // 2),
        max(1, image.height // 2)
    ))

    resized = half.resize(image.size)

    return [
        (
            "JPEG quality 90",
            image_bytes(
                image, "JPEG", quality=90
            )
        ),
        (
            "JPEG quality 35",
            image_bytes(
                image, "JPEG", quality=35
            )
        ),
        (
            f"Gaussian noise sigma={sigma}",
            image_bytes(
                Image.fromarray(noisy)
            )
        ),
        (
            "Downsize then resize",
            image_bytes(resized)
        )
    ]


# ----------------------------------------
# GENERATE SAMPLE IMAGES
# ----------------------------------------

def demo_gallery():

    a = Image.new(
        "RGB", (256, 192), "#f5d8d2"
    )

    painter = ImageDraw.Draw(a)

    painter.ellipse(
        (85, 35, 175, 125),
        fill="#d83d38"
    )

    painter.rectangle(
        (55, 145, 205, 165),
        fill="#a64440"
    )

    b = Image.new(
        "RGB", (256, 192), "#d9e9f7"
    )

    painter = ImageDraw.Draw(b)

    for x in range(30, 230, 42):
        painter.rectangle(
            (x, 28, x + 20, 162),
            fill="#3269b4"
        )

    c = Image.new(
        "RGB", (256, 192), "#d9eed6"
    )

    painter = ImageDraw.Draw(c)

    painter.polygon(
        [
            (128, 25),
            (220, 170),
            (38, 170)
        ],
        fill="#3b9c63"
    )

    return [
        ("red_circle.png", image_bytes(a)),
        ("red_circle_copy.png", image_bytes(a)),
        (
            "red_circle_compressed.jpg",
            image_bytes(
                a, "JPEG", quality=35
            )
        ),
        ("blue_bars.png", image_bytes(b)),
        (
            "green_triangle.png",
            image_bytes(c)
        )
    ]


# ----------------------------------------
# EXPORT CSV RESULTS
# ----------------------------------------

def csv_data(rows):

    if not rows:
        return ""

    target = io.StringIO()

    writer = csv.DictWriter(
        target,
        fieldnames=list(rows[0])
    )

    writer.writeheader()
    writer.writerows(rows)

    return target.getvalue()


# ----------------------------------------
# OPTIONAL CLIP AI MODEL
# ----------------------------------------

@st.cache_resource
def load_clip():

    import torch

    from transformers import (
        CLIPModel,
        AutoProcessor
    )

    model_id = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(
        model_id
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_id
    )

    return model, processor, torch


def normalized_features(features):

    if hasattr(features, "pooler_output"):
        features = features.pooler_output

    vector = (
        features.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    return vector / (
        np.linalg.norm(vector) + 1e-12
    )


# ----------------------------------------
# APPLICATION INTERFACE
# ----------------------------------------

st.title(
    "🔎 Image Similarity & Digital Forensics Analyzer"
)

st.write(
    "Compare images, test exact duplicates, "
    "and study how compression and noise "
    "affect similarity."
)

st.caption(
    "Educational research prototype: similarity "
    "scores cannot prove that a photo is "
    "authentic or manipulated."
)


# ----------------------------------------
# LOAD GALLERY
# ----------------------------------------

mode = st.sidebar.radio(
    "Image source",
    [
        "Demo images",
        "Upload my own images"
    ]
)

if mode == "Demo images":

    raw_gallery = demo_gallery()

else:

    files = st.sidebar.file_uploader(
        "Upload 2-20 PNG/JPG/WebP images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    raw_gallery = [
        (file.name, file.getvalue())
        for file in files[:20]
    ] if files else []


items = []

for name, raw in raw_gallery:

    try:
        items.append(record(name, raw))

    except Exception as error:
        st.warning(
            f"Cannot use {name}: {error}"
        )


if not items:

    st.info(
        "Upload images on the left, "
        "or switch to Demo images."
    )

    st.stop()


# ----------------------------------------
# APPLICATION TABS
# ----------------------------------------

tab_search, tab_test, tab_ai = st.tabs([
    "🔍 Search similar images",
    "🧪 Noise/compression tests",
    "🤖 AI text search"
])


# ----------------------------------------
# TAB 1: IMAGE SIMILARITY SEARCH
# ----------------------------------------

with tab_search:

    st.subheader(
        "Image similarity search"
    )

    selected_name = st.selectbox(
        "Choose an image",
        [item["name"] for item in items]
    )

    query_file = st.file_uploader(
        "Or upload a different query image",
        type=["png", "jpg", "jpeg", "webp"],
        key="query"
    )

    query = next(
        item for item in items
        if item["name"] == selected_name
    )

    if query_file is not None:

        try:
            query = record(
                query_file.name,
                query_file.getvalue()
            )

        except Exception as error:

            st.error(
                f"Could not read query: {error}"
            )

            st.stop()

    st.image(
        query["image"],
        caption="Query: " + query["name"],
        width=260
    )

    ranked = sorted(
        (
            (compare(query, item), item)
            for item in items
        ),
        key=lambda entry:
            entry[0]["Similarity score"],
        reverse=True
    )

    count = st.slider(
        "Number of results",
        1,
        len(ranked),
        min(len(ranked), 5)
    )

    for row, item in ranked[:count]:

        left, right = st.columns([1, 3])

        with left:
            st.image(
                item["image"],
                width=190
            )

        with right:

            st.write(
                f"**{row['Image']}**"
            )

            st.write(
                f"Similarity: "
                f"{row['Similarity score']:.1%} "
                f"| dHash distance: "
                f"{row['dHash distance']}"
            )

            st.write(
                "Exact byte-for-byte copy: "
                f"**{row['Exact file copy']}**"
            )

    results = [
        row for row, _ in ranked
    ]

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download search results (CSV)",
        csv_data(results),
        "similarity_results.csv",
        "text/csv"
    )


# ----------------------------------------
# TAB 2: IMAGE FORENSICS EXPERIMENTS
# ----------------------------------------

with tab_test:

    st.subheader(
        "Compare JPEG compression, "
        "Gaussian noise, and resizing"
    )

    chosen = st.selectbox(
        "Choose reference image",
        [item["name"] for item in items],
        key="experiment"
    )

    original = next(
        item for item in items
        if item["name"] == chosen
    )

    sigma = st.slider(
        "Gaussian noise strength",
        1,
        60,
        16
    )

    st.image(
        original["image"],
        caption="Original",
        width=260
    )

    table = []

    for name, raw in transformations(
        original["image"],
        sigma
    ):

        test_image = record(name, raw)

        similarity = compare(
            original,
            test_image
        )

        mse, psnr = mse_psnr(
            original["image"],
            test_image["image"]
        )

        table.append({
            "Transformation": name,
            "Similarity score":
                similarity["Similarity score"],
            "dHash distance":
                similarity["dHash distance"],
            "MSE": round(mse, 2),
            "PSNR (dB)": round(psnr, 2)
        })

        st.image(
            test_image["image"],
            caption=(
                f"{name} | similarity "
                f"{similarity['Similarity score']:.1%}"
            ),
            width=260
        )

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download experiment results (CSV)",
        csv_data(table),
        "forensics_experiments.csv",
        "text/csv"
    )

    st.caption(
        "MSE and PSNR compare aligned pixels. "
        "These measurements do not establish "
        "forensic authenticity."
    )


# ----------------------------------------
# TAB 3: AI IMAGE SEARCH WITH CLIP
# ----------------------------------------

with tab_ai:

    st.subheader(
        "Search images with a text description"
    )

    st.write(
        "Try: **a red circle** "
        "or **blue vertical bars**"
    )

    st.info(
        "Optional AI packages must be installed "
        "before using CLIP search."
    )

    sentence = st.text_input(
        "Describe the image"
    )

    if st.button(
        "Find images using CLIP"
    ):

        if not sentence.strip():

            st.warning(
                "Please enter a description."
            )

        else:

            try:

                model, processor, torch = load_clip()

                with torch.inference_mode():

                    text_inputs = processor(
                        text=[sentence],
                        return_tensors="pt",
                        padding=True
                    )

                    text_features = normalized_features(
                        model.get_text_features(
                            **text_inputs
                        )
                    )

                    matches = []

                    for item in items:

                        image_inputs = processor(
                            images=item["image"],
                            return_tensors="pt"
                        )

                        image_features = normalized_features(
                            model.get_image_features(
                                **image_inputs
                            )
                        )

                        score = float(
                            np.dot(
                                text_features,
                                image_features
                            )
                        )

                        matches.append(
                            (score, item)
                        )

                matches.sort(
                    key=lambda match: match[0],
                    reverse=True
                )

                for score, item in matches[:5]:

                    st.image(
                        item["image"],
                        width=220,
                        caption=(
                            f"{item['name']} "
                            f"| CLIP similarity: "
                            f"{score:.3f}"
                        )
                    )

            except (ImportError, OSError) as error:

                st.error(
                    "Install the optional AI "
                    "packages first."
                )

                st.caption(str(error))

            except Exception as error:

                st.error(
                    f"CLIP could not run: {error}"
                )
