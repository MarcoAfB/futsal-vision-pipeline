import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from futsal.datasets.datasetFiles import readJson
from futsal.datasets.objectCurriculum import prepareObjectCurriculum
from futsal.training.modelWeights import modelSizes
from futsal.training.objectCampaign import trainCampaign, worker


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare or train the public-to-own 2 -> 3 -> 5 object curriculum.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("sourceDatasetDir", "mediaStoreDir", "planDir", "datasetDir"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=project / "configs/objectCurriculum.yaml")
    train = commands.add_parser("train")
    for name in ("workDir", "datasetDir", "mediaStoreDir"):
        train.add_argument(f"--{name}", type=Path, required=True)
    train.add_argument("--campaignDir", type=Path)
    train.add_argument("--weightsRoot", type=Path, help="Contains yolo26n/yolo26n.pt and equivalent size directories.")
    train.add_argument("--config", type=Path, default=project / "configs/training/yolo26ObjectCurriculum.yaml")
    train.add_argument("--evaluationConfig", type=Path, default=project / "configs/evaluation/researchValidation.yaml")
    train.add_argument("--modelSizes", nargs="+", choices=modelSizes, default=list(modelSizes))
    train.add_argument("--resume", action="store_true")
    internal = commands.add_parser("worker", help=argparse.SUPPRESS)
    internal.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            return worker(readJson(args.request))
        if args.command == "prepare":
            result = prepareObjectCurriculum(args.sourceDatasetDir, args.config, args.mediaStoreDir, args.planDir, args.datasetDir)
            print(f"Object curriculum view: {result['contractId']}")
            return 0
        if len(args.modelSizes) != len(set(args.modelSizes)):
            raise ValueError("Model sizes must be unique.")
        if args.resume and args.campaignDir is None:
            raise ValueError("--resume requires --campaignDir.")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        directory = args.campaignDir or args.workDir / "launch" / f"objectCurriculumAllSizesSeed42_{stamp}"
        return trainCampaign(project, Path(__file__).resolve(), datasetDir=args.datasetDir, mediaStoreDir=args.mediaStoreDir,
            workDir=args.workDir, campaignDir=directory, configPath=args.config, evaluationConfig=args.evaluationConfig,
            weightsRoot=args.weightsRoot or args.workDir / "data/modelWeights", modelSizes=args.modelSizes, resume=args.resume)
    except (OSError, ValueError, KeyError) as error:
        print(f"Object curriculum could not complete: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
