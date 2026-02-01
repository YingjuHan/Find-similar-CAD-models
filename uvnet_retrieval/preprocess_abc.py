import subprocess
from pathlib import Path

from uvnet_retrieval.data.abc_dataset import discover_step_dirs


def run_solid_to_graph(uvnet_root, input_dir, output_dir, args):
    cmd = [
        args.python,
        "-m",
        "process.solid_to_graph",
        str(input_dir),
        str(output_dir),
        "--curv_u_samples",
        str(args.curv_u_samples),
        "--surf_u_samples",
        str(args.surf_u_samples),
        "--surf_v_samples",
        str(args.surf_v_samples),
        "--num_processes",
        str(args.num_processes),
    ]
    subprocess.run(cmd, cwd=uvnet_root, check=True)


def mirror_output_dir(step_dir, root, out_root):
    rel = Path(step_dir).relative_to(root)
    return Path(out_root) / rel


def main():
    import argparse

    parser = argparse.ArgumentParser("Preprocess ABC STEP files into UV-Net DGL graphs")
    parser.add_argument("--root", required=True, help="ABC dataset root (chunks directory parent)")
    parser.add_argument("--out_root", required=True, help="Output root for .bin graphs")
    parser.add_argument("--layout", default="abc", choices=["abc", "flat"], help="Dataset layout")
    parser.add_argument("--uvnet_root", default="external/uvnet", help="Path to UV-Net repo")
    parser.add_argument("--python", default="python", help="Python executable to run UV-Net scripts")
    parser.add_argument("--curv_u_samples", type=int, default=10)
    parser.add_argument("--surf_u_samples", type=int, default=10)
    parser.add_argument("--surf_v_samples", type=int, default=10)
    parser.add_argument("--num_processes", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    uvnet_root = Path(args.uvnet_root)

    if not uvnet_root.exists():
        raise SystemExit(f"UV-Net repo not found at {uvnet_root}")

    step_dirs = discover_step_dirs(root, layout=args.layout)
    if not step_dirs:
        print("No STEP directories found.")
        return

    for step_dir in step_dirs:
        out_dir = mirror_output_dir(step_dir, root, out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing {step_dir} -> {out_dir}")
        run_solid_to_graph(uvnet_root, step_dir, out_dir, args)


if __name__ == "__main__":
    main()
