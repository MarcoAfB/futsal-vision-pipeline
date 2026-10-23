import copy
from pathlib import Path

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
