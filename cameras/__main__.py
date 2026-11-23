"""Command line tool.

    python -m cameras list
    python -m cameras url camera1
    python -m cameras view            (all cameras in a grid, press q to quit)
    python -m cameras view camera1
    python -m cameras snapshot camera1 out.jpg
"""

import argparse
import sys

from . import get_url, list_cameras, open_camera, read_frame


def _view(names):
    import cv2
    import numpy as np

    caps = {name: open_camera(name) for name in names}
    window = "cameras (press q to quit)"
    tile_w, tile_h = 640, 360
    cols = 1 if len(names) == 1 else 2
    try:
        while True:
            tiles = []
            for name, cap in caps.items():
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                    cv2.putText(frame, "no signal", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame = cv2.resize(frame, (tile_w, tile_h))
                cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                tiles.append(frame)
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
            cv2.imshow(window, np.vstack(rows))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m cameras", description="Access the IP cameras.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print all camera names")
    p_url = sub.add_parser("url", help="print the RTSP URL of a camera")
    p_url.add_argument("name")
    p_view = sub.add_parser("view", help="show live video (all cameras if no name given)")
    p_view.add_argument("names", nargs="*")
    p_snap = sub.add_parser("snapshot", help="save one frame to an image file")
    p_snap.add_argument("name")
    p_snap.add_argument("output", nargs="?", default="snapshot.jpg")
    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            print("\n".join(list_cameras()))
        elif args.command == "url":
            print(get_url(args.name))
        elif args.command == "view":
            _view(args.names or list_cameras())
        elif args.command == "snapshot":
            import cv2

            cv2.imwrite(args.output, read_frame(args.name))
            print(f"Saved {args.output}")
    except (KeyError, RuntimeError, ConnectionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
