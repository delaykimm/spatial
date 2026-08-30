# reproduce_datasets

`spatial/` 파이프라인이 쓰는 이미지 데이터셋 중 **코드로 재현 가능한 것만** 따로 정리한 폴더.
raw 이미지 파일은 여기 없음 — 전부 실행 시 생성되는 코드만 있음.

## 범위

`spatial/data/`의 이미지셋 8개 중 4개는 코드로, 1개(clevr)는 "공식 데이터셋 + 필터링
스크립트"로 만들 수 있음. 나머지 3개(`whatsup_a`, `whatsup_b`, `spatialtunnel_images`)는
이 repo에 재현 코드가 아예 없어서 **직접 다운로드해야 함**:
- `whatsup_a`, `whatsup_b`: What'sUp 벤치마크 공식 배포처에서 받아와야 함
- `spatialtunnel_images`: `contrastive-probing`의 tsv에서 한 번 추출된 것 — 이 repo엔
  추출 스크립트가 없어서 원본을 직접 구해와야 함

| 스크립트 | 만드는 것 | 의존성 |
|---|---|---|
| `generate_aug1.py` | `aug1_images/` (360장) | matplotlib뿐 |
| `generate_triplet3ax.py` | `triplet3ax_cross_axis_images/` (360장) + manifest json | matplotlib뿐 |
| `generate_sameaxis.py` | `sameaxis_images/` (180장) + manifest json | matplotlib뿐, `generate_triplet3ax.py`의 상수 재사용 |
| `generate_aug2.py` | `aug2_images/` (807장) + manifest json | **외부**: Blender 4.3.2 + FORG3D 툴킷 + GPU. **권장: 그냥 `aug2_images/`(99MB) 파일 자체를 전달받을 것** — 아래 참고 |
| `fetch_clevr_images.py` | `clevr_cross_axis_images/` (496장) | **외부**: 공식 CLEVR v1.0 데이터셋 (real photo라 코드로 생성 불가, 다운로드 후 496장만 필터링) |

`render3d.py`: 위 4개가 공유하는 matplotlib mplot3d 렌더러(도형 그리기, 카메라, 배경 그리드,
저장 시 크롭/리사이즈). 이 폴더 밖 어떤 것도 import하지 않음 — 완전히 독립적으로 읽고 실행
가능.

## 검증

세 개(aug1/triplet3ax/sameaxis)는 각각 몇 개 씬을 실제로 렌더링해서 기존 `spatial/data/`의
원본 이미지와 픽셀 단위로 비교 — **완전히 동일**(mean abs diff = 0.0)함을 확인함. 즉 seed와
렌더 로직이 원본 생성 당시와 정확히 일치.

aug2는 물체쌍 배정 로직(`disjoint_pairs_for_axes`)만 검증 — 계산된 pair 집합이 기존
manifest의 pair를 전부 포함함(기존 데이터는 overlap-skip으로 일부만 렌더된 부분집합).
실제 Blender 렌더링은 GPU를 오래 점유하는 작업이라 실행하지 않음.

## 사용법

```
cd reproduce_datasets
python generate_aug1.py             # 기본 출력: ../data/aug1_images/
python generate_triplet3ax.py       # 기본 출력: ../data/triplet3ax_cross_axis_images/ + ../data/*.json
python generate_sameaxis.py         # 기본 출력: ../data/sameaxis_images/ + ../data/*.json
```

각 스크립트는 `--out`(+ triplet3ax/sameaxis는 `--out-json`)으로 출력 경로 변경 가능.

(`generate_aug2.py`, `fetch_clevr_images.py`는 여기 안 넣었음 — 아래 각자 섹션 참고. aug2는
애초에 실행하지 말고 파일로 받는 걸 권장.)

### aug2: 스크립트로 재현하지 말고 그냥 파일을 받을 것

`aug2_images/`는 99MB밖에 안 되는데, 이걸 스크립트로 재현하려면 Blender 4.3.2(149MB) +
FORG3D 툴킷(3D 오브젝트 라이브러리, 그보다 훨씬 큼) 설치 + GPU + 807장 렌더링(장당 Blender
subprocess 1회 + GPU Cycles 렌더라 총 수십 분~시간 단위)이 필요함 — 99MB 얻자고 GB 단위
툴킷을 새로 설치하고 GPU를 몇 시간 점유하는 건 명백히 손해. **`aug2_images/` 폴더 자체를
파일로 전달받는 게 맞음.**

`generate_aug2.py`는 그래서 "실행해서 재현하는 용도"가 아니라 **어떻게 만들어졌는지 기록하는
용도** + 나중에 물체쌍을 더 추가하고 싶을 때 참고할 코드로 남겨둔 것. 코드 자체가 Blender
렌더러를 재구현한 게 아니라 이 서버(`/node_data/urp26su_jiyun/`)에 이미 설치된 외부 툴킷을
호출하고 결과를 복사해오는 wrapper라, 정말 다른 환경에서 실행하려면 Blender/FORG3D를
새로 설치하고 스크립트 상단의 `BLENDER`/`RENDER_SCRIPT`/`SCRIPTS_DIR`/`FORG3D_PROPERTIES_PATH`
경로를 그 환경에 맞게 고쳐야 함.

### clevr 실행 조건

CLEVR은 실제 렌더링된 사진(Johnson et al., CLEVR v1.0)이라 이 repo가 만든 게 아님 — 코드로
재현할 수 없음. 하지만 어떤 496장이 필요한지는 이미 `data/clevr_cross_axis_triplets.json`에
정해져 있으므로(seed 42로 고정, `triplet_pipeline.clevr_sample_triplets()`가 15,000장 중
이 496장을 이미 골라놓음), 공식 데이터셋에서 그 496장만 뽑아오면 됨:
1. https://cs.stanford.edu/people/jcjohns/clevr/ 에서 CLEVR v1.0 val 이미지 다운로드
2. `python fetch_clevr_images.py --clevr-val-dir /path/to/CLEVR_v1.0/images/val`
   (15,000장 전체가 아니라 실제 쓰이는 496장만 복사됨 — 실제 원본 폴더로 스모크 테스트해서
   496/496 정상 동작 확인함)

## 왜 `triplet_pipeline.py`랑 따로 있나

`spatial/triplet_pipeline.py`도 triplet3ax/sameaxis를 만들 수 있지만, 그건 "이미지 생성 →
바로 hidden-state 추출"까지 한 번에 처리하는 855줄짜리 실행 파이프라인의 일부임(모델 로딩,
axis_pipeline.py 등 여러 모듈에 의존). 여기 있는 버전은 **이미지 생성 로직만** 떼어내서
다른 어떤 실험 코드에도 의존하지 않게 만든 것 — "이 데이터셋이 어떻게 만들어지는가"만 보고
싶을 때 훨씬 짧고 읽기 쉬움. 실제 실험 실행은 여전히 `triplet_pipeline.py`를 쓰면 됨(있으면
재사용, 없으면 이 폴더와 동일한 로직으로 새로 생성).
