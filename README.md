# spatial

VLM(Qwen3-VL 등)의 공간 방향(좌우/상하/원근) 표상을 추출·분석하는 자기완결형 파이프라인.

## 준비

```
pip install -r requirements.txt
```

`data/`에 필요한 이미지/manifest 대부분이 들어있거나 코드로 재현 가능하지만, **`whatsup_a/`, `whatsup_b/`, `spatialtunnel_images/`, CLEVR 데이터셋(→ `clevr_cross_axis_images/`)은 이 repo에 없고 재현 코드도 없어서 별도로 직접 다운로드해야 함** (자세한 내용은 아래 표 참고). `results/`는 GPU로 재생성되는 출력물(비어있어도 됨).

`data/` 하위 이미지셋의 재현 가능 여부 (재현 코드는 전부 `reproduce_datasets/`에 정리돼있음 — 자세한 내용은 그 폴더의 README 참고):

| 데이터셋 | 재현 코드 | 비고 |
|---|---|---|
| `triplet3ax_cross_axis_images` | `reproduce_datasets/generate_triplet3ax.py` | matplotlib만 필요, 외부 의존성 없음 |
| `sameaxis_images` | `reproduce_datasets/generate_sameaxis.py` | 위와 동일 |
| `aug1_images` | `reproduce_datasets/generate_aug1.py` | 위와 동일. 원본과 픽셀 단위로 동일하게 재현됨 (검증됨) |
| `aug2_images` | `reproduce_datasets/generate_aug2.py` | **전달한 데이터셋 직접 사용** (99MB인데 Blender+FORG3D+GPU 새로 설치가 더 큰 비용) — 스크립트는 provenance 기록용 |
| `clevr_cross_axis_images` | `reproduce_datasets/fetch_clevr_images.py` | **직접 다운로드 필요**: 실제 CLEVR 사진이라 코드 생성 불가 — [공식 CLEVR v1.0](https://cs.stanford.edu/people/jcjohns/clevr/) 다운로드 후 필요한 496장만 필터링 |
| `whatsup_a`, `whatsup_b` | 없음 | **직접 다운로드 필요**: What'sUp 벤치마크 실촬영 사진, 이 repo엔 재현 코드가 없어서 공식 배포처에서 직접 받아와야 함 |
| `spatialtunnel_images` | 없음 | **직접 다운로드 필요**: `contrastive-probing`의 tsv에서 한 번 추출된 것, 이 repo엔 추출 스크립트가 없어서 원본을 직접 구해와야 함 |

(`triplet_pipeline.py`도 triplet3ax/sameaxis를 만들 수 있지만 그건 hidden-state 추출까지 한 번에 하는 실행 파이프라인 안에서임 — `reproduce_datasets/`는 그 생성 로직만 떼어내 다른 아무 의존성 없이 이미지만 재현하도록 정리한 버전.)

## 각 파일 설명

| 파일 | 하는 일 |
|---|---|
| `config.yaml` | 모든 실험 변수(seed, layer, threshold 등)를 모아둔 설정 파일 |
| `axis_pipeline.py` | What'sUp/SpatialTunnel/Aug1/Aug2 4개 데이터셋에서 방향 벡터(축) 추출 |
| `axis_steering.py` | 그 축 벡터를 모델 hidden state에 주입해서 실제로 답이 바뀌는지 테스트 |
| `triplet_pipeline.py` | clevr/triplet3ax(cross-axis)/sameaxis 3종 triplet 데이터셋 생성 + hidden state 추출 (다른 대부분의 스크립트가 이걸 먼저 필요로 함) |
| `cross_dataset_analysis.py` | 4개 외부 데이터셋끼리 축 벡터/steering 결과 비교 |
| `cross_axis_analysis.py` | compositionality: r_AB + r_BC ≈ r_AC 인지 (자기 표상 내부 일관성) |
| `cross_axis_alignment.py` | alignment: 모델 벡터가 실제 물리적 방향과 일치하는지 |
| `cross_axis_readout_vqa.py` | sign-match readout(기하학적) vs VQA(직접 질문) 정확도 비교 |
| `diffpair_steering.py` | 같은 이미지, 다른 pair 벡터를 주입해도 causal하게 작동하는지 |
| `diffimage_steering.py` | 다른 이미지, 같은 관계(pair) 벡터를 주입해도 causal하게 작동하는지 |
| `sameaxis_4way_readout_vqa.py` | sameaxis에서 A/B/C/D 4개 중 극값(제일 왼쪽 등) 찾기 readout vs VQA |
| `chain_hop_pipeline.py` | N-hop 랜덤워크 체인(A→B→C→...→Z, 한 축 위, 각 hop 부호 랜덤) **렌더링** + hidden state 추출 |
| `clevr_chain_pipeline.py` | 위와 같은 체인을 **실제 CLEVR 사진에서** 장면당 임의 N개 오브젝트/랜덤 순서로 구성 (렌더링 없음, horizontal/closefar 2축만) + hidden state 추출 |
| `chain_hop_readout_vqa.py` | hop 수(2~6)별로 latent computation(Σc_i,i+1 축 투영 sign) vs MLLM generation(VQA) 정확도 비교, hop-accuracy 플롯 생성. `--source {synthetic,clevr}`로 위 두 백엔드 중 선택 |

## 데이터셋 (`--dataset`)

- **clevr**: 실제 CLEVR 사진에서 A→B는 축1(가로), B→C는 축2(원근)로 나뉜 L자형 triplet
- **triplet3ax**: clevr과 같은 L자형 구조(A→B/B→C가 서로 다른 축)를 합성 렌더링으로 만든 것
- **sameaxis**: A,B,C가 전부 하나의 같은 축 위에 정렬된 triplet (+ 관련 없는 물체 D 하나 추가)

## 실행 순서

**1) 외부 데이터셋 축 추출** (whatsup/spatialtunnel/aug1/aug2 — `cross_axis_analysis.py`의 view B, `cross_axis_alignment.py`의 datasetaxis 비교, `cross_dataset_analysis.py`가 이 결과를 필요로 함)
```
python axis_pipeline.py --dataset whatsup --model qwen3vl      # spatialtunnel/aug1/aug2도 동일하게 반복
python axis_steering.py --dataset whatsup                      # 동일하게 4개 반복
```

**2) triplet 데이터셋 추출** (clevr/triplet3ax/sameaxis — 이 아래 대부분의 스크립트가 필요로 함)
```
python triplet_pipeline.py --dataset clevr --model qwen3vl
python triplet_pipeline.py --dataset triplet3ax --model qwen3vl
python triplet_pipeline.py --dataset sameaxis --model qwen3vl
```

**3) triplet 기반 핵심 분석/steering** (2번만 있으면 됨, 순서 무관)
```
python cross_axis_alignment.py --dataset clevr --model qwen3vl      # readout_vqa가 이 결과를 씀 → 먼저 실행
python cross_axis_readout_vqa.py --dataset clevr --model qwen3vl
python cross_axis_analysis.py --dataset clevr --model qwen3vl
python diffpair_steering.py --dataset sameaxis --model qwen3vl
python diffimage_steering.py --dataset clevr --model qwen3vl
python sameaxis_4way_readout_vqa.py --model qwen3vl
```
(`--dataset`는 `clevr`/`triplet3ax`/`sameaxis` 중 스크립트가 지원하는 것으로 바꿔가며 반복)

**4) 마무리 요약 플롯**

- **외부 데이터셋 4개끼리 서로 얼마나 비슷한 축을 갖는지 비교**: 1번의 `axis_pipeline.py`/`axis_steering.py` 결과가 있어야 함
  ```
  python cross_dataset_analysis.py
  ```
- **compositionality + alignment + magnitude fidelity를 한 그래프로**: 3번의 `cross_axis_alignment.py` 결과가 있어야 함
  ```python
  import cross_axis_alignment as cag
  cag.draw_composition_grounding_joint_plot("clevr")   # 그림 1장 생성
  ```

**5) hop 수에 따른 latent computation vs MLLM generation** (위 1~4번과 독립, 두 백엔드 중 택1)

- **synthetic** (`chain_hop_pipeline.py`): 렌더링으로 새로 생성, 한 축 위 랜덤워크 체인(hop=2~6), 3축(horizontal/vertical/closefar) 전부 지원
  ```
  python chain_hop_pipeline.py --model qwen3vl
  python chain_hop_readout_vqa.py --model qwen3vl
  ```
- **clevr** (`clevr_chain_pipeline.py`): 렌더링 없이 실제 CLEVR val 15,000장 메타데이터(`data/clevr_val_scenes.json`, 이미 포함됨)에서 장면당 임의 N개 오브젝트를 랜덤 순서로 뽑아 체인 구성. horizontal/closefar 2축만 지원(CLEVR엔 vertical 위치 변화가 없음). 최초 1회만 로컬 CLEVR val 이미지 폴더 경로 필요
  ```
  python clevr_chain_pipeline.py --clevr-val-dir /path/to/CLEVR_v1.0/images/val --model qwen3vl
  python chain_hop_readout_vqa.py --model qwen3vl --source clevr
  ```

`results/plot/readout_vqa/[clevr_]chain_hop_accuracy_by_hops_{model}.png`가 x축=hop 수, y축=정확도인 메인 플롯 (축별 breakdown은 `..._by_axis_{model}.png`). hop/seed/step 크기 등은 `config.yaml`의 `chain_hop`/`clevr_chain_hop` 섹션에서 조정.

## 참고

- `--model`은 `qwen3vl`(기본)/`qwen2`/`llava` 중 선택. 모델별로 결과 파일명이 자동으로 구분됨.
- GPU 여러 개로 나눠 돌리고 싶으면 `diffpair_steering.py`/`diffimage_steering.py`는 `--axis`(축 하나만) + `--merge`(합치기) 옵션 지원.
- 결과는 `results/`(npz/json), 그림은 `results/png/`에 저장됨.
