import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import streamlit as st
from rembg import new_session, remove
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(
    page_title="Keychain Laser Processor — Layer System",
    page_icon="🔑",
    layout="wide",
)


# =========================================================
# SESSION STATE INIT
# =========================================================

def init_session_state():
    """Init semua session state untuk layer locking."""
    defaults = {
        # Layer locks
        "lock_original": False,
        "lock_tone": False,
        "lock_dither": False,
        "lock_line": False,
        "lock_keychain": False,
        # Spot picker
        "picked_color": None,
        "picked_position": None,
        "picker_active": False,

        # Locked data
        "locked_original": None,
        "locked_tone": None,
        "locked_tone_skin_mask": None,
        "locked_tone_manual_mask": None,
        "locked_tone_line_source": None,
        "locked_tone_face_line_source": None,
        "locked_dither": None,
        "locked_dither_edges": None,
        "locked_line": None,
        "locked_keychain": None,

        # Canvas
        "canvas_version": 0,

        # Image hash untuk detect perubahan
        "last_image_hash": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =========================================================
# FUNGSI ASAS
# =========================================================

def mm_to_px(mm, dpi):
    return int(round((mm / 25.4) * dpi))


def resize_image_to_target(image, target_width_px, target_height_px):
    return image.resize(
        (target_width_px, target_height_px),
        Image.Resampling.LANCZOS,
    )


def encode_png(image, dpi=254):
    buffer = BytesIO()
    image.save(buffer, format="PNG", dpi=(dpi, dpi))
    return buffer.getvalue()


@st.cache_resource
def get_rembg_session():
    return new_session("u2net")


@st.cache_data(show_spinner=False)
def remove_background(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    cutout = remove(image, session=get_rembg_session()).convert("RGBA")
    white_background = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
    return Image.alpha_composite(white_background, cutout).convert("RGB")

def rgb_to_hex(rgb):
    """Convert RGB tuple ke hex string."""
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0]), int(rgb[1]), int(rgb[2])
    )

# =========================================================
# COLOR REPLACEMENT
# =========================================================

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def replace_color_range(image_rgb, target_color, replacement_color, tolerance=15):
    result = image_rgb.copy()
    target = np.array(target_color, dtype=np.int16)
    replacement = np.array(replacement_color, dtype=np.uint8)
    diff = np.abs(result.astype(np.int16) - target)
    mask = np.all(diff <= tolerance, axis=2)
    result[mask] = replacement
    return result


# =========================================================
# SKIN MASK
# =========================================================

def make_skin_mask(rgb_image, sensitivity, smoothing):
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    spread = int(sensitivity)

    ycrcb_mask = cv2.inRange(
        ycrcb_image,
        np.array([0, max(0, 133 - spread), max(0, 77 - spread)], dtype=np.uint8),
        np.array([255, min(255, 180 + spread), min(255, 135 + spread)], dtype=np.uint8),
    )

    hsv_mask = cv2.inRange(
        hsv_image,
        np.array([0, max(5, 20 - spread), max(20, 35 - spread)], dtype=np.uint8),
        np.array([min(179, 25 + spread), 235, 255], dtype=np.uint8),
    )

    skin_mask = cv2.bitwise_and(ycrcb_mask, hsv_mask)

    kernel_size = max(3, int(smoothing))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

    return skin_mask


# =========================================================
# TONE FUNCTIONS
# =========================================================

def apply_tone_curve(gray_image, gamma, contrast, brightness):
    gamma = max(0.05, float(gamma))
    lookup_table = np.array(
        [((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    corrected = cv2.LUT(gray_image, lookup_table).astype(np.float32)
    corrected = (corrected - 128.0) * contrast + 128.0 + brightness
    return np.clip(corrected, 0, 255).astype(np.uint8)


def apply_sharpening(gray_image, amount, radius, detail_threshold):
    if amount <= 0:
        return gray_image.copy()

    blurred = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=radius, sigmaY=radius)
    original_float = gray_image.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    detail = original_float - blurred_float
    sharpened = original_float + (detail * amount)
    sharpened = np.clip(sharpened, 0, 255)
    significant_detail = np.abs(detail) >= detail_threshold
    result = original_float.copy()
    result[significant_detail] = sharpened[significant_detail]
    return np.clip(result, 0, 255).astype(np.uint8)


def lift_shadows(gray_image, lift_amount=30, threshold=80):
    result = gray_image.astype(np.float32)
    dark_mask = gray_image < threshold
    result[dark_mask] = np.minimum(result[dark_mask] + lift_amount, threshold)
    return np.clip(result, 0, 255).astype(np.uint8)


def compress_shadows(gray_image, shadow_point=100, strength=0.6):
    """Compress shadow — tarik nilai gelap ke atas, gradient kekal."""
    result = gray_image.astype(np.float32)
    dark_mask = result < shadow_point
    result[dark_mask] = (
        result[dark_mask]
        + (shadow_point - result[dark_mask]) * strength
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def clip_whites(gray_image, white_threshold=245):
    result = gray_image.copy()
    result[result >= white_threshold] = 255
    return result


def apply_local_highlight_pop(gray_image, clip_limit=3.0, tile_size=8):
    """
    CLAHE (adaptive local-contrast) — bukan global gamma/contrast.

    Masalah asal: gamma/contrast dalam apply_tone_curve() bandingkan setiap
    pixel dengan nilai TETAP 128 (global midpoint). Kalau lipatan baju gelap
    (contohnya fold highlight cuma ~90-110), dia still di bawah 128, jadi
    "contrast" tarik dia LEBIH gelap, bukan lebih putih — sebab tu line
    lipat yang sepatutnya putih jadi terus hitam bila di-dither.

    CLAHE selesaikan ni sebab dia tengok setiap "tile" (kotak kecil) secara
    berasingan dan stretch histogram tile tu sendiri ke full range 0-255.
    Jadi dalam kotak kecil kat lengan baju yang gelap, mana-mana bahagian
    yang RELATIF lagi cerah (fold highlight) automatik ditarik dekat 255,
    manakala bahagian yang relatif gelap ditarik dekat 0 — walaupun secara
    global semua tu 'gelap'. Ni yang buat garisan lipat nampak putih bersih
    lepas dither, bukan jadi blok hitam + noise dots.
    """
    clip_limit = max(0.1, float(clip_limit))
    tile_size = max(2, int(tile_size))
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(gray_image)


# =========================================================
# MANUAL BRUSH MASK
# =========================================================

def canvas_to_mask(canvas_image_data, image_width, image_height):
    if canvas_image_data is None:
        return np.zeros((image_height, image_width), dtype=np.uint8)

    canvas_rgba = np.asarray(canvas_image_data, dtype=np.uint8)

    if canvas_rgba.ndim != 3 or canvas_rgba.shape[2] < 4:
        return np.zeros((image_height, image_width), dtype=np.uint8)

    red = canvas_rgba[:, :, 0].astype(np.int16)
    green = canvas_rgba[:, :, 1].astype(np.int16)
    blue = canvas_rgba[:, :, 2].astype(np.int16)
    alpha = canvas_rgba[:, :, 3]

    paint_pixels = (alpha > 5) & (red > green + 40) & (red > blue + 40)
    erase_pixels = (alpha > 5) & (blue > red + 40) & (blue > green + 20)

    paint_mask = np.where(paint_pixels, 255, 0).astype(np.uint8)
    erase_mask = np.where(erase_pixels, 255, 0).astype(np.uint8)

    erase_mask = cv2.dilate(erase_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    final_mask = cv2.bitwise_and(paint_mask, cv2.bitwise_not(erase_mask))
    final_mask = cv2.resize(
        final_mask,
        (image_width, image_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return final_mask


def apply_local_adjustment(
    gray_image, manual_mask,
    local_brightness, local_contrast,
    shadow_recovery, feather_radius,
):
    if manual_mask is None or not np.any(manual_mask):
        return gray_image.copy()

    mask_float = manual_mask.astype(np.float32) / 255.0

    if feather_radius > 0:
        mask_float = cv2.GaussianBlur(
            mask_float, (0, 0),
            sigmaX=feather_radius,
            sigmaY=feather_radius,
        )

    mask_float = np.clip(mask_float, 0.0, 1.0)
    original = gray_image.astype(np.float32)
    adjusted = (original - 128.0) * local_contrast + 128.0 + local_brightness
    darkness = 1.0 - (original / 255.0)
    adjusted += shadow_recovery * darkness
    adjusted = np.clip(adjusted, 0, 255)
    result = original * (1.0 - mask_float) + adjusted * mask_float

    return np.clip(result, 0, 255).astype(np.uint8)


# =========================================================
# SKIN PROTECTION
# =========================================================

def protect_skin(
    gray_image, skin_mask,
    feature_threshold, shadow_threshold,
    shadow_brightness, shadow_density,
    use_soft_skin=False,
):
    result = gray_image.copy()
    skin = skin_mask > 0

    features = skin & (gray_image <= feature_threshold)
    shadows = skin & (gray_image > feature_threshold) & (gray_image <= shadow_threshold)
    highlights = skin & (gray_image > shadow_threshold)

    if use_soft_skin:
        original_skin = gray_image[skin].astype(np.float32)
        density = shadow_density / 100.0
        adjusted_skin = 255.0 - ((255.0 - original_skin) * density)
        result[skin] = np.clip(adjusted_skin, 0, 255).astype(np.uint8)
        result[features] = gray_image[features]
    else:
        result[highlights] = 255
        result[features] = gray_image[features]

    result[highlights] = 255
    result[features] = gray_image[features]

    original_shadows = np.maximum(gray_image[shadows], shadow_brightness).astype(np.float32)
    density = shadow_density / 100.0
    adjusted_shadows = 255.0 - ((255.0 - original_shadows) * density)
    result[shadows] = np.clip(adjusted_shadows, 0, 255).astype(np.uint8)

    return result


# =========================================================
# DITHERING
# =========================================================

def apply_floyd_steinberg(gray_image, density=1.0):
    density = np.clip(float(density), 0.0, 1.0)
    working = gray_image.astype(np.float32)
    height, width = working.shape

    for y in range(height):
        for x in range(width):
            old_pixel = working[y, x]
            new_pixel = 255.0 if old_pixel >= 127.5 else 0.0
            working[y, x] = new_pixel
            error = (old_pixel - new_pixel) * density

            if x + 1 < width:
                working[y, x + 1] += error * 7 / 16
            if y + 1 < height:
                if x > 0:
                    working[y + 1, x - 1] += error * 3 / 16
                working[y + 1, x] += error * 5 / 16
                if x + 1 < width:
                    working[y + 1, x + 1] += error * 1 / 16

    return np.clip(working, 0, 255).astype(np.uint8)


def apply_atkinson(gray_image, density=1.0):
    density = np.clip(float(density), 0.0, 1.0)
    working = gray_image.astype(np.float32)
    height, width = working.shape

    for y in range(height):
        if y % 2 == 0:
            x_positions = range(width)
            neighbours = [(1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)]
        else:
            x_positions = range(width - 1, -1, -1)
            neighbours = [(-1, 0), (-2, 0), (1, 1), (0, 1), (-1, 1), (0, 2)]

        for x in x_positions:
            old_pixel = working[y, x]
            new_pixel = 255.0 if old_pixel >= 127.5 else 0.0
            working[y, x] = new_pixel
            distributed_error = ((old_pixel - new_pixel) / 8.0) * density

            for offset_x, offset_y in neighbours:
                neighbour_x = x + offset_x
                neighbour_y = y + offset_y

                if 0 <= neighbour_x < width and 0 <= neighbour_y < height:
                    working[neighbour_y, neighbour_x] += distributed_error

    return np.clip(working, 0, 255).astype(np.uint8)


# =========================================================
# KEYCHAIN STYLE
# =========================================================

def make_subject_mask(gray_image, background_threshold=240):
    subject_mask = (gray_image < background_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    subject_mask = cv2.morphologyEx(subject_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    subject_mask = cv2.morphologyEx(subject_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return subject_mask


def make_black_outline(subject_mask, outline_thickness=4):
    kernel_size = outline_thickness * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inner = cv2.erode(subject_mask, kernel, iterations=1)
    return cv2.subtract(subject_mask, inner)


def make_white_interior(subject_mask, outline_thickness=4, white_border=2):
    total_erosion = outline_thickness + white_border
    kernel_size = total_erosion * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(subject_mask, kernel, iterations=1)


def make_keychain_style(
    gray_image, dither_density=0.7,
    outline_thickness=4, white_border=2,
    background_threshold=240, dither_method="Atkinson",
):
    subject_mask = make_subject_mask(gray_image, background_threshold)
    outline = make_black_outline(subject_mask, outline_thickness)
    white_interior = make_white_interior(subject_mask, outline_thickness, white_border)

    if dither_method == "Atkinson":
        dithered = apply_atkinson(gray_image, dither_density)
    else:
        dithered = apply_floyd_steinberg(gray_image, dither_density)

    result = np.full_like(gray_image, 255)
    fill_region = white_interior > 0
    result[fill_region] = dithered[fill_region]
    result[outline > 0] = 0

    return result, subject_mask, outline, white_interior


# =========================================================
# LINE ART
# =========================================================

def make_line_art(gray_image, low_threshold, high_threshold, thickness):
    low = min(int(low_threshold), int(high_threshold) - 1)
    high = max(int(high_threshold), low + 1)
    edges = cv2.Canny(gray_image, low, high)

    if thickness > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(thickness), int(thickness))
        )
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


# =========================================================
# STAGE 1: PROCESS TONE
# =========================================================

def process_tone(image, manual_mask, settings):
    """Process sampai tone sahaja — TANPA dithering."""
    rgb_image = np.array(image.convert("RGB"))
    original_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    working_gray = original_gray.copy()

    if settings["denoise_enabled"]:
        working_gray = cv2.bilateralFilter(
            working_gray, settings["denoise_size"], 35, 35
        )

    # Local highlight pop (CLAHE) — kena jalan SEBELUM gamma/contrast global,
    # supaya fold highlight yang gelap secara global tapi cerah secara
    # tempatan sempat "diselamatkan" dulu sebelum contrast global tarik dia
    # ke hitam. Kalau letak lepas tone_enabled, dah terlambat — detail dah
    # hilang dalam blok hitam yang rata.
    if settings.get("clahe_enabled", False):
        working_gray = apply_local_highlight_pop(
            working_gray,
            settings["clahe_clip_limit"],
            settings["clahe_tile_size"],
        )

    if settings["tone_enabled"]:
        working_gray = apply_tone_curve(
            working_gray,
            settings["gamma"],
            settings["contrast"],
            settings["brightness"],
        )

    if settings["clip_whites_enabled"]:
        working_gray = clip_whites(working_gray, settings["white_clip_threshold"])

    if settings["color_replace_enabled"]:
        temp_rgb = cv2.cvtColor(working_gray, cv2.COLOR_GRAY2RGB)
        target_rgb = hex_to_rgb(settings["color_target_hex"])
        replace_rgb = hex_to_rgb(settings["color_replace_hex"])
        temp_rgb = replace_color_range(
            temp_rgb, target_rgb, replace_rgb, settings["color_tolerance"]
        )
        working_gray = cv2.cvtColor(temp_rgb, cv2.COLOR_RGB2GRAY)

    if settings["local_adjustment_enabled"]:
        working_gray = apply_local_adjustment(
            working_gray, manual_mask,
            settings["local_brightness"],
            settings["local_contrast"],
            settings["shadow_recovery"],
            settings["mask_feather"],
        )

    face_line_source = working_gray.copy()

    if settings["sharpen_enabled"]:
        working_gray = apply_sharpening(
            working_gray,
            settings["sharpen_amount"],
            settings["sharpen_radius"],
            settings["sharpen_detail_threshold"],
        )

    line_source = working_gray.copy()

    skin_mask = make_skin_mask(
        rgb_image,
        settings["skin_sensitivity"],
        settings["mask_smoothing"],
    )

    if settings["exclude_brush_from_skin"]:
        skin_mask[manual_mask > 0] = 0

    if settings["skin_enabled"]:
        working_gray = protect_skin(
            working_gray, skin_mask,
            settings["feature_threshold"],
            settings["shadow_threshold"],
            settings["shadow_brightness"],
            settings["shadow_density"],
            settings["use_soft_skin"],
        )

    if settings["lift_shadows_enabled"]:
        working_gray = lift_shadows(
            working_gray,
            settings["shadow_lift_amount"],
            settings["shadow_lift_threshold"],
        )

    if settings["shadow_compression_enabled"]:
        working_gray = compress_shadows(
            working_gray,
            settings["shadow_compression_point"],
            settings["shadow_compression_strength"],
        )

    return {
        "tone": working_gray,
        "skin_mask": skin_mask,
        "manual_mask": manual_mask,
        "line_source": line_source,
        "face_line_source": face_line_source,
        "original_gray": original_gray,
    }


# =========================================================
# STAGE 2: DITHERING
# =========================================================

def process_dithering(tone_data, settings):
    """Apply dithering pada tone."""
    working_gray = tone_data["tone"].copy()

    if settings["dither_enabled"]:
        if settings["dither_method"] == "Atkinson":
            result = apply_atkinson(working_gray, settings["dither_density"])
        else:
            result = apply_floyd_steinberg(working_gray, settings["dither_density"])
    else:
        _, result = cv2.threshold(
            working_gray, settings["binary_threshold"], 255, cv2.THRESH_BINARY
        )

    result = np.array(result, dtype=np.uint8, copy=True)
    return result


# =========================================================
# STAGE 3: LINE ART
# =========================================================

def process_line_art(dither_result, tone_data, settings):
    """Apply line art pada hasil dithering."""
    result = dither_result.copy()
    skin_mask = tone_data["skin_mask"]
    line_source = tone_data["line_source"]
    face_line_source = tone_data["face_line_source"]

    edges = np.zeros_like(result)

    if settings["line_enabled"]:
        clothing_edges = make_line_art(
            line_source,
            settings["edge_low"],
            settings["edge_high"],
            settings["edge_thickness"],
        )
        edges = clothing_edges.copy()

        if settings["protect_face_from_lines"]:
            face_edges = make_line_art(
                face_line_source,
                settings["face_edge_low"],
                settings["face_edge_high"],
                1,
            )

            local_darkness = cv2.erode(
                face_line_source,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )

            dark_feature_support = (
                local_darkness <= settings["face_feature_threshold"]
            ).astype(np.uint8) * 255

            dark_feature_support = cv2.dilate(
                dark_feature_support,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )

            face_edges[dark_feature_support == 0] = 0

            inner_skin_mask = skin_mask.copy()
            margin = settings["skin_inner_margin"]

            if margin > 0:
                kernel_size = margin * 2 + 1
                inner_skin_mask = cv2.erode(
                    inner_skin_mask,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                    ),
                    iterations=1,
                )

            inside_skin = inner_skin_mask > 0
            edges[inside_skin] = face_edges[inside_skin]

    if settings["line_halo_enabled"]:
        halo_edges = edges.copy()

        if settings["clothing_halo_only"]:
            halo_edges[skin_mask > 0] = 0

        halo_radius = settings["line_halo_size"]
        kernel_size = (halo_radius * 2) + 1
        halo_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        expanded_edges = cv2.dilate(halo_edges, halo_kernel, iterations=1)
        result[expanded_edges > 0] = 255

    if settings["line_enabled"]:
        result[edges > 0] = 0

    return result, edges


# =========================================================
# STAGE 4: KEYCHAIN STYLE
# =========================================================

def process_keychain(line_result, tone_data, settings):
    """Apply keychain style pada hasil line art."""
    if not settings["keychain_style_enabled"]:
        return line_result

    working_gray = tone_data["tone"]
    result, subject_mask, outline, white_interior = make_keychain_style(
        working_gray,
        dither_density=settings["dither_density"],
        outline_thickness=settings["outline_thickness"],
        white_border=settings["white_border"],
        background_threshold=settings["keychain_background_threshold"],
        dither_method=settings["dither_method"],
    )

    return result


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("🔑 Keychain Laser Processor — Layer System")

st.write(
    "Multi-layer processing — lock setiap stage macam Photoshop."
)


# =========================================================
# SIDEBAR: SETUP KEYCHAIN
# =========================================================

st.sidebar.header("📐 Setup Keychain")

keychain_shape = st.sidebar.selectbox(
    "Bentuk keychain",
    ["Segi Empat", "Bulat", "Oval"],
)

if keychain_shape == "Bulat":
    keychain_diameter = st.sidebar.slider("Diameter (mm)", 20, 80, 40, 1)
    keychain_width_mm = keychain_diameter
    keychain_height_mm = keychain_diameter
elif keychain_shape == "Oval":
    keychain_width_mm = st.sidebar.slider("Lebar (mm)", 20, 80, 50, 1)
    keychain_height_mm = st.sidebar.slider("Tinggi (mm)", 20, 80, 35, 1)
else:
    keychain_width_mm = st.sidebar.slider("Lebar (mm)", 20, 100, 40, 1)
    keychain_height_mm = st.sidebar.slider("Tinggi (mm)", 20, 100, 40, 1)

st.sidebar.header("⚙️ DPI Mesin Laser")

laser_dpi = st.sidebar.selectbox(
    "DPI mesin laser",
    [254, 318, 400, 500],
    index=0,
)

target_width_px = mm_to_px(keychain_width_mm, laser_dpi)
target_height_px = mm_to_px(keychain_height_mm, laser_dpi)

st.sidebar.info(
    f"📏 Saiz: **{target_width_px} × {target_height_px} px**\n\n"
    f"({keychain_width_mm}mm × {keychain_height_mm}mm @ {laser_dpi} DPI)"
)


# =========================================================
# SIDEBAR: LAYER CONTROLS
# =========================================================

st.sidebar.markdown("---")
st.sidebar.header("🗂️ Layer Controls")

# Reset button
if st.sidebar.button("🔄 Reset Semua Layer", use_container_width=True):
    for key in ["lock_original", "lock_tone", "lock_dither", "lock_line", "lock_keychain"]:
        st.session_state[key] = False
    for key in [
        "locked_original", "locked_tone", "locked_tone_skin_mask",
        "locked_tone_manual_mask", "locked_tone_line_source",
        "locked_tone_face_line_source", "locked_dither",
        "locked_dither_edges", "locked_line", "locked_keychain",
    ]:
        st.session_state[key] = None
    st.success("✅ Semua layer reset!")
    st.rerun()

# Status layer
st.sidebar.markdown("**Status Layer:**")

layer_status = {
    "1. Original": st.session_state.lock_original,
    "2. Tone": st.session_state.lock_tone,
    "3. Dithering": st.session_state.lock_dither,
    "4. Line Art": st.session_state.lock_line,
    "5. Keychain": st.session_state.lock_keychain,
}

for layer_name, is_locked in layer_status.items():
    icon = "🔒" if is_locked else "🔓"
    st.sidebar.write(f"{icon} {layer_name}")


# =========================================================
# SIDEBAR: LAYER 1 — ORIGINAL
# =========================================================

with st.sidebar.expander("📷 Layer 1: Original", expanded=False):
    remove_bg_enabled = st.checkbox("AI background removal", value=True)

    if st.button("🔒 Lock Original", use_container_width=True):
        st.session_state.lock_original = True
        st.success("Original locked!")

    if st.button("🔓 Unlock Original", use_container_width=True):
        st.session_state.lock_original = False
        st.session_state.locked_original = None


# =========================================================
# SIDEBAR: LAYER 2 — TONE (DROPDOWN)
# =========================================================

with st.sidebar.expander("🎨 Layer 2: Tone Preparation", expanded=True):

    # ---- Denoise ----
    with st.expander("🔇 Denoise", expanded=False):
        denoise_enabled = st.checkbox("Enable denoise", value=True)
        denoise_size = st.slider("Smooth strength", 3, 15, 5, 2)

    # ---- Local Highlight Pop (CLAHE) ----
    with st.expander("✨ Local Highlight Pop (CLAHE)", expanded=True):
        st.caption(
            "Ni fix untuk lipatan/garisan cerah pada kain gelap yang "
            "'hilang' jadi hitam bila di-dither. Adaptive — bukan global."
        )
        clahe_enabled = st.checkbox("Enable local highlight pop", value=True)
        clahe_clip_limit = st.slider(
            "Clip limit (kekuatan pop)", 1.0, 8.0, 3.0, 0.5,
            help="Tinggi = lebih agresif tarik fold highlight ke putih.",
        )
        clahe_tile_size = st.slider(
            "Tile size (kehalusan tempatan)", 2, 24, 8, 1,
            help="Kecil = lebih 'local' (detect fold kecil). "
                 "Besar = lebih 'global' (macam contrast biasa).",
        )

    # ---- Global Tone ----
    with st.expander("🎨 Global Tone", expanded=False):
        tone_enabled = st.checkbox("Enable tone correction", value=True)
        gamma = st.slider("Gamma", 0.5, 2.5, 1.20, 0.05)
        contrast = st.slider("Contrast", 0.5, 2.5, 1.10, 0.05)
        brightness = st.slider("Brightness", -80, 80, 5, 5)

    # ---- White Clip ----
    with st.expander("⚪ White Clip", expanded=False):
        clip_whites_enabled = st.checkbox("Enable white clip", value=True)
        white_clip_threshold = st.slider("White clip threshold", 200, 255, 245, 1)

    # ---- Color Replacement ----
    with st.expander("🎨 Color Replacement", expanded=False):
        color_replace_enabled = st.checkbox(
            "Enable color replacement", value=False
        )
    # Mode: manual hex atau spot picker
    color_mode = st.radio(
        "Color selection mode",
        ["Manual (hex)", "Spot picker (klik gambar)"],
    )
    
    if color_mode == "Manual (hex)":
        color_target_hex = st.color_picker("Target color", value="#2b2b2b")
        color_replace_hex = st.color_picker("Replacement color", value="#FFFFFF")
        
    else:  # Spot picker
        st.info("Klik pada gambar untuk pilih warna target")
        
        # Butang aktifkan picker
        if st.button("🎯 Aktifkan Spot Picker"):
            st.session_state.picker_active = True
        
        if st.button("❌ Reset Picker"):
            st.session_state.picker_active = False
            st.session_state.picked_color = None
            st.session_state.picked_position = None
        
        # Tunjuk warna yang dipilih
        if st.session_state.picked_color is not None:
            picked_hex = rgb_to_hex(st.session_state.picked_color)
            st.success(f"Warna dipilih: {picked_hex}")
            
            # Preview warna
            st.markdown(
                f'<div style="background-color:{picked_hex}; '
                f'width:100%; height:50px; border-radius:5px;"></div>',
                unsafe_allow_html=True,
            )
            
            # Set sebagai target
            color_target_hex = picked_hex
        else:
            color_target_hex = "#2b2b2b"
        
        color_replace_hex = st.color_picker("Replacement color", value="#FFFFFF")
    
    color_tolerance = st.slider("Tolerance", 0, 50, 15, 1)

    # ---- Sharpness ----
    with st.expander("🔪 Sharpness", expanded=False):
        sharpen_enabled = st.checkbox("Enable sharpening", value=True)
        sharpen_amount = st.slider("Sharp strength", 0.0, 3.0, 0.80, 0.10)
        sharpen_radius = st.slider("Sharp radius", 0.5, 5.0, 1.20, 0.10)
        sharpen_detail_threshold = st.slider("Minimum detail", 0, 50, 8, 1)

    # ---- Shadow Lifting ----
    with st.expander("🌑 Shadow Lifting", expanded=False):
        lift_shadows_enabled = st.checkbox("Enable shadow lifting", value=True)
        shadow_lift_amount = st.slider("Lift amount", 0, 150, 60, 5)
        shadow_lift_threshold = st.slider("Lift threshold", 30, 200, 100, 5)

    # ---- Shadow Compression ----
    with st.expander("🌑 Shadow Compression", expanded=False):
        shadow_compression_enabled = st.checkbox(
            "Enable shadow compression", value=True
        )
        shadow_compression_point = st.slider("Shadow point", 50, 180, 100, 5)
        shadow_compression_strength = st.slider(
            "Compression strength", 0.0, 1.0, 0.6, 0.05
        )

    # ---- Skin Protection ----
    with st.expander("🧴 Skin Protection", expanded=False):
        skin_enabled = st.checkbox("Enable skin protection", value=True)
        skin_sensitivity = st.slider("Skin sensitivity", 0, 35, 10, 1)
        mask_smoothing = st.slider("Mask smoothing", 3, 21, 7, 2)
        feature_threshold = st.slider("Feature gelap", 20, 160, 80, 5)
        shadow_threshold = st.slider(
            "Had shadow kulit",
            feature_threshold + 1, 230,
            max(150, feature_threshold + 1), 5,
        )
        shadow_brightness = st.slider("Kecerahan shadow kulit", 150, 250, 220, 5)
        shadow_density = st.slider("Ketumpatan shadow kulit", 0, 100, 60, 5)
        use_soft_skin = st.checkbox("Soft skin filter", value=True)
        exclude_brush_from_skin = st.checkbox(
            "Anggap brush sebagai bukan kulit", value=True
        )

    # ---- Manual Brush ----
    with st.expander("🖌️ Manual Brush", expanded=False):
        drawing_mode_label = st.selectbox(
            "Drawing tool",
            ["Brush", "Eraser", "Polygon / Lasso", "Rectangle", "Circle"],
        )
        brush_size = st.slider("Brush size", 2, 100, 30, 2)
        eraser_size = st.slider("Eraser size", 2, 150, 40, 2)
        local_brightness = st.slider("Local brightness", -100, 150, 35, 5)
        local_contrast = st.slider("Local contrast", 0.5, 3.0, 1.40, 0.05)
        shadow_recovery = st.slider("Shadow recovery", 0, 150, 60, 5)
        mask_feather = st.slider("Mask feather", 0, 30, 8, 1)

    # ---- Lock Tone Button ----
    st.markdown("---")
    col_lock1, col_lock2 = st.columns(2)

    with col_lock1:
        if st.button("🔒 Lock Tone", use_container_width=True, key="lock_tone_btn"):
            st.session_state.lock_tone = True
            st.success("Tone locked!")

    with col_lock2:
        if st.button("🔓 Unlock", use_container_width=True, key="unlock_tone_btn"):
            st.session_state.lock_tone = False
            st.session_state.locked_tone = None
            st.rerun()


# =========================================================
# SIDEBAR: LAYER 3 — DITHERING (DROPDOWN)
# =========================================================

with st.sidebar.expander("🎚️ Layer 3: Dithering", expanded=False):

    dither_enabled = st.checkbox("Enable dithering", value=True)
    dither_method = st.selectbox("Algorithm", ["Floyd-Steinberg", "Atkinson"])
    dither_density = st.slider(
        "Density", 0.10, 1.00, 0.70, 0.05,
        help="0.10 = cerah, 1.00 = padat",
    )
    binary_threshold = st.slider("Threshold (tanpa dither)", 0, 255, 128, 5)

    st.markdown("---")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if st.button("🔒 Lock Dither", use_container_width=True, key="lock_dither_btn"):
            st.session_state.lock_dither = True
            st.success("Dither locked!")

    with col_d2:
        if st.button("🔓 Unlock", use_container_width=True, key="unlock_dither_btn"):
            st.session_state.lock_dither = False
            st.session_state.locked_dither = None
            st.rerun()


# =========================================================
# SIDEBAR: LAYER 4 — LINE ART (DROPDOWN)
# =========================================================

with st.sidebar.expander("✏️ Layer 4: Line Art", expanded=False):

    line_enabled = st.checkbox("Enable line art", value=True)

    with st.expander("Edge Detection", expanded=True):
        edge_low = st.slider("Edge threshold bawah", 0, 200, 60, 5)
        edge_high = st.slider("Edge threshold atas", 10, 255, 150, 5)
        edge_thickness = st.slider("Ketebalan line", 1, 5, 1, 1)

    with st.expander("Face Protection", expanded=False):
        protect_face_from_lines = st.checkbox("Kurangkan line pada muka", value=True)
        face_edge_low = st.slider("Face edge low", 0, 220, 70, 5)
        face_edge_high = st.slider("Face edge high", 10, 255, 170, 5)
        face_feature_threshold = st.slider("Feature gelap muka", 20, 200, 125, 5)
        skin_inner_margin = st.slider("Jarak perlindungan kulit", 0, 5, 1, 1)

    with st.expander("White Halo", expanded=False):
        line_halo_enabled = st.checkbox("Lindungi line dengan ruang putih", value=True)
        line_halo_size = st.slider("Ketebalan ruang putih", 1, 10, 2, 1)
        clothing_halo_only = st.checkbox("White halo pada pakaian sahaja", value=True)

    st.markdown("---")
    col_l1, col_l2 = st.columns(2)

    with col_l1:
        if st.button("🔒 Lock Line", use_container_width=True, key="lock_line_btn"):
            st.session_state.lock_line = True
            st.success("Line locked!")

    with col_l2:
        if st.button("🔓 Unlock", use_container_width=True, key="unlock_line_btn"):
            st.session_state.lock_line = False
            st.session_state.locked_line = None
            st.rerun()


# =========================================================
# SIDEBAR: LAYER 5 — KEYCHAIN STYLE (DROPDOWN)
# =========================================================

with st.sidebar.expander("🔑 Layer 5: Keychain Style", expanded=False):

    keychain_style_enabled = st.checkbox(
        "Enable keychain style", value=False
    )

    outline_thickness = 4
    white_border = 2
    keychain_background_threshold = 240

    if keychain_style_enabled:
        outline_thickness = st.slider("Ketebalan outline", 1, 10, 4, 1)
        white_border = st.slider("Ketebalan white border", 1, 10, 2, 1)
        keychain_background_threshold = st.slider(
            "Background threshold", 200, 255, 240, 1
        )

    st.markdown("---")
    col_k1, col_k2 = st.columns(2)

    with col_k1:
        if st.button("🔒 Lock Keychain", use_container_width=True, key="lock_kc_btn"):
            st.session_state.lock_keychain = True
            st.success("Keychain locked!")

    with col_k2:
        if st.button("🔓 Unlock", use_container_width=True, key="unlock_kc_btn"):
            st.session_state.lock_keychain = False
            st.session_state.locked_keychain = None
            st.rerun()


# =========================================================
# SIDEBAR: CANVAS VIEW
# =========================================================

st.sidebar.markdown("---")
st.sidebar.header("👁️ Canvas View")

canvas_zoom = st.sidebar.slider(
    "Canvas zoom", 25, 200, 100, 5, format="%d%%"
)


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Pilih gambar portrait (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
)


# =========================================================
# BUILD SETTINGS DICTIONARY
# =========================================================

settings = {
    "denoise_enabled": denoise_enabled,
    "denoise_size": denoise_size,
    "clahe_enabled": clahe_enabled,
    "clahe_clip_limit": clahe_clip_limit,
    "clahe_tile_size": clahe_tile_size,
    "tone_enabled": tone_enabled,
    "gamma": gamma,
    "contrast": contrast,
    "brightness": brightness,
    "clip_whites_enabled": clip_whites_enabled,
    "white_clip_threshold": white_clip_threshold,
    "color_replace_enabled": color_replace_enabled,
    "color_target_hex": color_target_hex,
    "color_replace_hex": color_replace_hex,
    "color_tolerance": color_tolerance,
    "sharpen_enabled": sharpen_enabled,
    "sharpen_amount": sharpen_amount,
    "sharpen_radius": sharpen_radius,
    "sharpen_detail_threshold": sharpen_detail_threshold,
    "lift_shadows_enabled": lift_shadows_enabled,
    "shadow_lift_amount": shadow_lift_amount,
    "shadow_lift_threshold": shadow_lift_threshold,
    "shadow_compression_enabled": shadow_compression_enabled,
    "shadow_compression_point": shadow_compression_point,
    "shadow_compression_strength": shadow_compression_strength,
    "skin_enabled": skin_enabled,
    "skin_sensitivity": skin_sensitivity,
    "mask_smoothing": mask_smoothing,
    "feature_threshold": feature_threshold,
    "shadow_threshold": shadow_threshold,
    "shadow_brightness": shadow_brightness,
    "shadow_density": shadow_density,
    "use_soft_skin": use_soft_skin,
    "exclude_brush_from_skin": exclude_brush_from_skin,
    "local_adjustment_enabled": True,
    "local_brightness": local_brightness,
    "local_contrast": local_contrast,
    "shadow_recovery": shadow_recovery,
    "mask_feather": mask_feather,
    "dither_enabled": dither_enabled,
    "dither_method": dither_method,
    "dither_density": dither_density,
    "binary_threshold": binary_threshold,
    "line_enabled": line_enabled,
    "edge_low": edge_low,
    "edge_high": edge_high,
    "edge_thickness": edge_thickness,
    "protect_face_from_lines": protect_face_from_lines,
    "face_edge_low": face_edge_low,
    "face_edge_high": face_edge_high,
    "face_feature_threshold": face_feature_threshold,
    "skin_inner_margin": skin_inner_margin,
    "line_halo_enabled": line_halo_enabled,
    "line_halo_size": line_halo_size,
    "clothing_halo_only": clothing_halo_only,
    "keychain_style_enabled": keychain_style_enabled,
    "outline_thickness": outline_thickness,
    "white_border": white_border,
    "keychain_background_threshold": keychain_background_threshold,
}


# =========================================================
# MAIN PROCESSING
# =========================================================

if uploaded_file is not None:
    source_bytes = uploaded_file.getvalue()

    try:
        source_image = Image.open(BytesIO(source_bytes)).convert("RGB")
    except Exception as error:
        st.error(f"Gambar tidak dapat dibaca: {error}")
        st.stop()

    source_image = resize_image_to_target(
        source_image, target_width_px, target_height_px
    )

    if remove_bg_enabled:
        try:
            with st.spinner("AI sedang membuang background..."):
                source_image = remove_background(
                    encode_png(source_image, laser_dpi)
                )
        except Exception as error:
            st.warning(f"Background removal gagal: {error}")

        # Spot picker
    if (
        color_replace_enabled
        and color_mode == "Spot picker (klik gambar)"
        and st.session_state.picker_active
    ):
        st.markdown("### 🎯 Spot Picker — Klik pada gambar")

        picker_image = source_image.convert("RGB")

        value = streamlit_image_coordinates(
            picker_image,
            width=picker_image.width,
            key="spot_picker",
        )

        if value is not None:
            x = int(value["x"])
            y = int(value["y"])

            if 0 <= x < picker_image.width and 0 <= y < picker_image.height:
                picked_rgb = picker_image.getpixel((x, y))

                st.session_state.picked_color = picked_rgb
                st.session_state.picked_position = (x, y)

                # Tamatkan satu sesi pemilihan sebelum rerun.
                st.session_state.picker_active = False
                st.rerun()

    # Canvas
    if st.sidebar.button("Clear semua brush", use_container_width=True):
        st.session_state.canvas_version += 1
        st.rerun()

    st.subheader("🖌️ Brush adjustment")

    canvas_column, preview_column = st.columns([1, 1], gap="large")

    with canvas_column:
        base_canvas_width = min(800, source_image.width)
        canvas_width = max(150, round(base_canvas_width * canvas_zoom / 100.0))
        canvas_ratio = canvas_width / float(source_image.width)
        canvas_height = max(1, round(source_image.height * canvas_ratio))

        canvas_background = source_image.resize(
            (canvas_width, canvas_height), Image.Resampling.LANCZOS
        )

        drawing_mode_map = {
            "Brush": "freedraw",
            "Eraser": "freedraw",
            "Polygon / Lasso": "polygon",
            "Rectangle": "rect",
            "Circle": "circle",
        }

        drawing_mode = drawing_mode_map[drawing_mode_label]

        if drawing_mode_label == "Eraser":
            active_stroke_color = "rgba(0, 80, 255, 1.0)"
            active_fill_color = "rgba(0, 80, 255, 0.50)"
            active_stroke_width = eraser_size
        else:
            active_stroke_color = "rgba(255, 0, 0, 1.0)"
            active_fill_color = "rgba(255, 0, 0, 0.50)"
            active_stroke_width = brush_size

        canvas_result = st_canvas(
            fill_color=active_fill_color,
            stroke_width=active_stroke_width,
            stroke_color=active_stroke_color,
            background_image=canvas_background,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            return_image_data=True,
            key=f"canvas_{st.session_state.canvas_version}",
        )

    manual_mask = canvas_to_mask(
        canvas_result.image_data,
        source_image.width,
        source_image.height,
    )

    # =====================================================
    # PROCESS LAYER BY LAYER
    # =====================================================

    # LAYER 2: TONE
    if st.session_state.lock_tone and st.session_state.locked_tone is not None:
        tone_data = st.session_state.locked_tone
        st.info("🔒 Layer 2 (Tone) — LOCKED")
    else:
        with st.spinner("Processing tone..."):
            tone_data = process_tone(source_image, manual_mask, settings)

        if st.session_state.lock_tone:
            st.session_state.locked_tone = tone_data

    # LAYER 3: DITHERING
    if st.session_state.lock_dither and st.session_state.locked_dither is not None:
        dither_result = st.session_state.locked_dither
        st.info("🔒 Layer 3 (Dither) — LOCKED")
    else:
        with st.spinner("Processing dithering..."):
            dither_result = process_dithering(tone_data, settings)

        if st.session_state.lock_dither:
            st.session_state.locked_dither = dither_result

    # LAYER 4: LINE ART
    if st.session_state.lock_line and st.session_state.locked_line is not None:
        line_result = st.session_state.locked_line
        st.info("🔒 Layer 4 (Line) — LOCKED")
    else:
        with st.spinner("Processing line art..."):
            line_result, edges = process_line_art(
                dither_result, tone_data, settings
            )

        if st.session_state.lock_line:
            st.session_state.locked_line = line_result

    # LAYER 5: KEYCHAIN STYLE
    if st.session_state.lock_keychain and st.session_state.locked_keychain is not None:
        final_result = st.session_state.locked_keychain
        st.info("🔒 Layer 5 (Keychain) — LOCKED")
    else:
        with st.spinner("Processing keychain style..."):
            final_result = process_keychain(line_result, tone_data, settings)

        if st.session_state.lock_keychain:
            st.session_state.locked_keychain = final_result

    # =====================================================
    # DISPLAY ALL LAYERS
    # =====================================================

    with preview_column:
        st.markdown("### 🔑 Final Result")
        st.image(
            Image.fromarray(final_result),
            caption=f"Keychain {keychain_width_mm}mm × {keychain_height_mm}mm @ {laser_dpi} DPI",
            use_container_width=True,
        )

        st.download_button(
            label="⬇️ Download PNG",
            data=encode_png(Image.fromarray(final_result), laser_dpi),
            file_name=f"keychain_{keychain_width_mm}x{keychain_height_mm}mm_{laser_dpi}dpi.png",
            mime="image/png",
            use_container_width=True,
        )

    # =====================================================
    # LAYER PREVIEWS
    # =====================================================

    with st.expander("🔍 Preview Semua Layer", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Layer 2: Tone**")
            st.image(
                Image.fromarray(tone_data["tone"]),
                use_container_width=True,
            )

        with col2:
            st.markdown("**Layer 3: Dithering**")
            st.image(
                Image.fromarray(dither_result),
                use_container_width=True,
            )

        with col3:
            st.markdown("**Layer 4: Line Art**")
            st.image(
                Image.fromarray(line_result),
                use_container_width=True,
            )

    with st.expander("📋 Diagnosis Mask", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Skin Mask**")
            st.image(
                Image.fromarray(tone_data["skin_mask"]),
                use_container_width=True,
            )

        with col2:
            st.markdown("**Manual Brush Mask**")
            st.image(
                Image.fromarray(tone_data["manual_mask"]),
                use_container_width=True,
            )