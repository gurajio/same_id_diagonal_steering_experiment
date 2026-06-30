import sys
import time
from pathlib import Path
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

SUPPORTED_EXTENSIONS = {".png"}


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def wait_until_file_is_ready(path: Path, timeout: float = 10.0) -> bool:
    """
    ファイルコピー中に変換を始めないように，
    ファイルサイズが安定するまで待つ．
    """
    start = time.time()
    previous_size = -1

    while time.time() - start < timeout:
        if not path.exists():
            return False

        current_size = path.stat().st_size

        if current_size == previous_size and current_size > 0:
            return True

        previous_size = current_size
        time.sleep(0.5)

    return False


def convert_to_eps(image_path: Path, output_dir: Path) -> None:
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return

    if not image_path.exists() or not image_path.is_file():
        return

    if not wait_until_file_is_ready(image_path):
        print(f"ファイルの読み込み準備が完了しませんでした: {image_path.name}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_path.with_suffix(".eps").name

    try:
        with Image.open(image_path) as im:
            fig = im.convert("RGB")
            fig.save(output_path, format="EPS")

        print(f"変換完了: {image_path.name} -> {output_path.name}", flush=True)

    except Exception as e:
        print(f"変換失敗: {image_path.name}", flush=True)
        print(e, flush=True)


class PngHandler(FileSystemEventHandler):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.last_converted = {}

    def should_convert(self, path: Path) -> bool:
        """
        on_created と on_modified が連続して呼ばれることがあるため，
        短時間の重複変換を避ける．
        """
        now = time.time()
        previous = self.last_converted.get(path)

        if previous is not None and now - previous < 1.0:
            return False

        self.last_converted[path] = now
        return True

    def handle_path(self, path: Path):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        if self.should_convert(path):
            convert_to_eps(path, self.output_dir)

    def on_created(self, event):
        if not event.is_directory:
            self.handle_path(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self.handle_path(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            self.handle_path(Path(event.dest_path))


def main():
    base_dir = get_base_dir()
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"監視開始: {input_dir}", flush=True)
    print(f"出力先: {output_dir}", flush=True)

    # 起動時点ですでにinputにあるPNGも変換する
    for image_path in sorted(input_dir.glob("*.png")):
        convert_to_eps(image_path, output_dir)

    event_handler = PngHandler(output_dir)
    observer = Observer()
    observer.schedule(event_handler, str(input_dir), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("監視を終了します．", flush=True)
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()