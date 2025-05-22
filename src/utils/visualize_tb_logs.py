import csv
import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tensorboard.backend.event_processing import event_accumulator


def collect_scalar_events(tb_log_dir: Path):
    """
    TensorBoard event ファイルを再帰的に検索し、
    タグごとの [(step, value), ...] をまとめて返す。
    """
    tag_history = dict()  # tag -> list[(step, value)]
    for ev_file in tb_log_dir.rglob("events.*"):
        ea = event_accumulator.EventAccumulator(
            str(ev_file), size_guidance={"scalars": 0}
        )
        try:
            ea.Reload()
        except Exception as e:
            logging.warning(f"Skip broken file: {ev_file} ({e})")
            continue

        for tag in ea.Tags().get("scalars", []):
            for scalar in ea.Scalars(tag):
                tag_history.setdefault(tag, []).append((scalar.step, scalar.value))

    # step 昇順でソート
    for tag, hist in tag_history.items():
        tag_history[tag] = sorted(hist, key=lambda x: x[0])
    return tag_history


def save_plot(tag: str, history, out_dir: Path, figsize, dpi):
    """1 つのメトリクスを PNG として保存"""
    steps, values = zip(*history, strict=False)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(steps, values)
    plt.title(tag)
    plt.xlabel("step")
    plt.ylabel("value")
    fname = tag.replace("/", "_") + ".png"
    plt.tight_layout()
    plt.savefig(out_dir / fname)
    plt.close()


def append_test_log(tag: str, history, csv_path: Path):
    """test 系メトリクスを CSV に追記 (step, tag, value)"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["step", "tag", "value"])
        for step, val in history:
            writer.writerow([step, tag, val])


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    tb_dir = Path(cfg.tb_logs.paths.tb_log_dir)
    out_dir = Path(cfg.tb_logs.paths.output_img_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg.tb_logs.paths.test_log_path)

    tag_hist = collect_scalar_events(tb_dir)

    print("\n=== 取得メトリクス一覧 ===============================")
    for tag, history in sorted(tag_hist.items()):
        print(f" • {tag:<40} : {len(history)} points")
    print("====================================================\n")

    for tag, history in tag_hist.items():
        tag_lower = tag.lower()
        if "test" in tag_lower:
            append_test_log(tag, history, csv_path)
            logging.info(f"Logged test metric '{tag}' to {csv_path.name}")
        else:
            save_plot(
                tag,
                history,
                out_dir,
                tuple(cfg.tb_logs.plot.figsize),
                cfg.tb_logs.plot.dpi,
            )
            logging.info(f"Saved plot for '{tag}'")

    logging.info("完了しました")


if __name__ == "__main__":
    main()
