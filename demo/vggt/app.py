import json
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file

app = Flask(__name__, template_folder="templates")

APP_DIR = Path(__file__).parent


@app.route("/")
def index():
    summary_path = APP_DIR / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    cameras_path = APP_DIR / "cameras.json"
    cameras_data = []
    if cameras_path.exists():
        with open(cameras_path) as f:
            cameras_data = json.load(f)

    frames = sorted((APP_DIR / "input_frames").glob("*.png"))
    frame_count = len(frames)
    frame_names = [f.name for f in frames]

    return render_template("index.html",
                           summary=summary,
                           cameras_data=cameras_data,
                           frame_count=frame_count,
                           frame_names=frame_names)


@app.route("/frames/<path:filename>")
def serve_frame(filename):
    return send_file(APP_DIR / "input_frames" / filename)


@app.route("/depth/<path:filename>")
def serve_depth(filename):
    return send_file(APP_DIR / "depth_maps" / filename)


@app.route("/confidence/<path:filename>")
def serve_confidence(filename):
    return send_file(APP_DIR / "confidence_maps" / filename)


@app.route("/pointcloud/<path:filename>")
def serve_pointcloud(filename):
    return send_file(APP_DIR / "pointcloud_views" / filename)


@app.route("/camera/<path:filename>")
def serve_camera(filename):
    return send_file(APP_DIR / "camera_views" / filename)


@app.route("/unproject/<path:filename>")
def serve_unproject(filename):
    return send_file(APP_DIR / "unproject_views" / filename)


@app.route("/track/<path:filename>")
def serve_track(filename):
    return send_file(APP_DIR / "track_views" / filename)


@app.route("/video")
def serve_video():
    video_path = APP_DIR / "vggt-demo.mp4"
    return send_file(str(video_path), mimetype="video/mp4")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
