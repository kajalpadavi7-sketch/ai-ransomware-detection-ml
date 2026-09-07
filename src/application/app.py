from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

from detector import detect_file


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# UPLOAD CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# FILE DETECTION
# ============================================================

@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if "file" not in request.files:

        return render_template(
            "index.html",
            error="Please select an EXE file."
        )

    file = request.files["file"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select a file."
        )

    # --------------------------------------------------------
    # Only EXE files
    # --------------------------------------------------------

    if not file.filename.lower().endswith(".exe"):

        return render_template(
            "index.html",
            error="Only Windows EXE files are supported."
        )

    # --------------------------------------------------------
    # Secure filename
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    # --------------------------------------------------------
    # Create file path
    # --------------------------------------------------------

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    file.save(
        file_path
    )

    # --------------------------------------------------------
    # Detect ransomware
    # --------------------------------------------------------

    try:

        result = detect_file(
            file_path
        )

    except Exception as e:

        # Delete file if detection fails
        try:
            os.remove(file_path)
        except:
            pass

        return render_template(
            "index.html",
            error=f"Detection failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Delete uploaded file after analysis
    # --------------------------------------------------------

    try:

        os.remove(
            file_path
        )

    except:

        pass

    # --------------------------------------------------------
    # Show result
    # --------------------------------------------------------

    return render_template(
        "result.html",
        filename=filename,
        result=result
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )