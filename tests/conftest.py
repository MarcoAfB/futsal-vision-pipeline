import copy
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
import pytest
import yaml


@pytest.fixture
def smallDataset(tmp_path, monkeypatch):
    projectRoot = Path(__file__).resolve().parents[1]
    catalog = yaml.safe_load((projectRoot / "configs/dataSources.yaml").read_text(encoding="utf-8"))
    dataRoot = tmp_path / "sources"
    configRoot = tmp_path / "configuration"
    dataRoot.mkdir()
    configRoot.mkdir()
    (configRoot / "labelMap.yaml").write_bytes((projectRoot / "configs/labelMap.yaml").read_bytes())
    oldVideo = copy.deepcopy(catalog["ownVideos"][0])
    oldVideo.update({"videoPath": "old.mp4", "annotationPath": "old.xml"})
    oldVideo["reviewedIntervals"][0]["endFrame"] = 1
    montage = copy.deepcopy(oldVideo)
    montage.update(
        {
            "videoId": "montage",
            "legacyMatchId": None,
            "videoPath": "montage.mp4",
            "annotationPath": "montage.xml",
        }
    )
    montage["reviewedIntervals"][0]["intervalId"] = "montageReviewed"
    montage["reviewedIntervals"][0]["provenance"] = {
        "matchIdentityBasis": "researcherConfirmedSameMatch",
        "reviewedRangeBasis": "researcherDeclaration",
        "sameMatchAsVideoId": "video01",
    }
    catalog.update(
        {
            "legacyReviewPath": "reviewed.yaml",
            "ownVideos": [oldVideo, montage],
            "publicSources": [],
            "syntheticSources": [],
            "expected": {
                "reviewedFrames": 4,
                "matchCount": 1,
                "videoCount": 2,
                "reviewedIntervalCount": 2,
            },
        }
    )
    configPath = configRoot / "dataSources.yaml"
    configPath.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
    registry = {"partidas": {"Jogo1": {"video": "old.mp4", "xml": "old.xml", "maximoFrames": 2}}}
    (dataRoot / "reviewed.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
    (dataRoot / "old.mp4").write_bytes(b"synthetic media placeholder")
    (dataRoot / "montage.mp4").write_bytes(b"another synthetic media placeholder")
    (dataRoot / "old.xml").write_text(
        """<annotations><version>1.1</version>
      <meta><task><size>10000</size><original_size><width>100</width><height>100</height>
      </original_size></task></meta>
      <track id="1" label="Bola"><ellipse frame="0" cx="10" cy="10" rx="2" ry="3"/></track>
      <track id="2" label="Bola"><box frame="2" xtl="1" ytl="1" xbr="2" ybr="2"/></track>
      <track id="3" label="Unknown outside review"><points frame="99" points="nan,0"/></track>
      <track id="4" label="Bola"><box frame="1" outside="1"/></track>
      </annotations>""",
        encoding="utf-8",
    )
    (dataRoot / "montage.xml").write_text(
        """<annotations><version>1.1</version>
      <meta><job><original_size><width>100</width><height>100</height></original_size></job></meta>
      <track id="10" label="Escanteio inferior time b"><points frame="0" points="2,3"/></track>
      <track id="11" label="Escanteio inferior time b"><points frame="0" points="4,5"/></track>
      <track id="12" label="Bola"><box frame="1" xtl="1" ytl="2" xbr="3" ybr="4"/></track>
      <track id="13" label="Bola"><box frame="1" xtl="5" ytl="6" xbr="7" ybr="8"/></track>
      <track id="14" label="Juiz"><box frame="0" xtl="1" ytl="2" xbr="3" ybr="4"/></track>
      <track id="15" label="Jogador"><box frame="1" xtl="5" ytl="6" xbr="7" ybr="8"/></track>
      </annotations>""",
        encoding="utf-8",
    )

    def videoProbe(path):
        return {
            "byteSize": path.stat().st_size,
            "width": 100,
            "height": 100,
            "frameCount": 10000,
            "fps": 30,
            "metadataProvider": "testFixture",
        }

    monkeypatch.setattr("futsal.ownInventory.readVideoMetadata", videoProbe)
    return dataRoot, configPath, tmp_path / "results"


@pytest.fixture
def preparedSources(tmp_path):
    projectRoot = Path(__file__).resolve().parents[1]
    configRoot, dataRoot = tmp_path / "config", tmp_path / "sources"
    configRoot.mkdir()
    dataRoot.mkdir()
    labels = yaml.safe_load((projectRoot / "configs/labelMap.yaml").read_text(encoding="utf-8"))
    catalog = yaml.safe_load((projectRoot / "configs/dataSources.yaml").read_text(encoding="utf-8"))
    recipe = yaml.safe_load((projectRoot / "configs/curriculum.yaml").read_text(encoding="utf-8"))
    recipe.pop("splitPath")
    recipe["tasks"][0]["trainingSources"] = ["futsal4400"]
    recipe["tasks"][1]["trainingSources"] = ["futsalMot"]
    videos = []
    for number, matchId in enumerate(("matchA", "matchB", "matchC")):
        video = copy.deepcopy(catalog["ownVideos"][0])
        video.update(videoId=f"video{number}", legacyMatchId=None,
                     videoPath=f"video{number}.avi", annotationPath=f"video{number}.xml")
        video["reviewedIntervals"] = [{"intervalId": f"interval{number}", "matchId": matchId,
            "startFrame": 0, "endFrame": 1, "internalCutsStatus": "pending",
            "provenance": {"matchIdentityBasis": "researcherDeclaration"}}]
        writer = cv2.VideoWriter(str(dataRoot / video["videoPath"]),
                                 cv2.VideoWriter_fourcc(*"MJPG"), 5, (96, 64))
        assert writer.isOpened()
        for frameIndex in range(2):
            writer.write(np.full((64, 96, 3), 30 + number * 40 + frameIndex * 10, dtype=np.uint8))
        writer.release()
        root = ElementTree.Element("annotations")
        meta = ElementTree.SubElement(ElementTree.SubElement(root, "meta"), "task")
        dimensions = ElementTree.SubElement(meta, "original_size")
        ElementTree.SubElement(dimensions, "width").text = "96"
        ElementTree.SubElement(dimensions, "height").text = "64"
        for item in labels["canonicalClasses"]:
            if number == 1 and item["name"] == "lowerCornerSideB":
                continue
            source = next(entry["sourceLabel"] for entry in labels["labelMaps"]["ownCvat"]["entries"]
                          if entry.get("targetClass") == item["name"])
            track = ElementTree.SubElement(root, "track", id=str(item["classId"]), label=source)
            for frameIndex in range(2):
                if item["geometry"] == "point":
                    ElementTree.SubElement(track, "points", frame=str(frameIndex), points="40,30")
                elif item["name"] == "ball":
                    ElementTree.SubElement(track, "ellipse", frame=str(frameIndex),
                                           cx="10", cy="10", rx="2", ry="2")
                else:
                    ElementTree.SubElement(track, "box", frame=str(frameIndex),
                                           xtl="20", ytl="10", xbr="30", ybr="40")
        ElementTree.ElementTree(root).write(dataRoot / video["annotationPath"], encoding="utf-8")
        videos.append(video)
    sources = [copy.deepcopy(source) for source in catalog["publicSources"]
               if source["sourceId"] in {"futsal4400", "futsalMot"}]
    for source in sources:
        source.update(imageRoot=source["sourceId"], labelRoot=source["sourceId"],
                      metadataFiles=[], labelMapEvidence="publicEvidence.yaml")
        directory = dataRoot / source["sourceId"]
        directory.mkdir()
        assert cv2.imwrite(str(directory / "image.jpg"), np.full((64, 96, 3), 120, dtype=np.uint8))
        content = "0 0.5 0.5 0.1 0.1\n1 0.3 0.5 0.1 0.4\n"
        if source["sourceId"] == "futsalMot":
            content += "2 0.7 0.5 0.1 0.4\n"
        (directory / "image.txt").write_text(content, encoding="utf-8")
    catalog.update(ownVideos=videos, publicSources=sources, syntheticSources=[],
                   legacyReviewPath="reviewed.yaml",
                   expected={"reviewedFrames": 6, "matchCount": 3, "videoCount": 3,
                             "reviewedIntervalCount": 3})
    (dataRoot / "reviewed.yaml").write_text("partidas: {}\n", encoding="utf-8")
    (dataRoot / "publicEvidence.yaml").write_text("fixture: true\n", encoding="utf-8")
    for name, value in (("labelMap.yaml", labels), ("dataSources.yaml", catalog),
                        ("curriculum.yaml", recipe)):
        (configRoot / name).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return dataRoot, configRoot / "curriculum.yaml", tmp_path / "plan"
