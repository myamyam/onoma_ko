# 한국어 의성어 전사

오디오를 **HTSAT → projection + 위치 정보 → KoBART decoder**로 전달해 한국어 의성어를 생성합니다. 기존 BART 텍스트 encoder는 사용하지 않습니다. HTSAT는 로컬 `HTSAT.ckpt`, decoder는 KoBART 사전학습 가중치로 초기화합니다.

## 설치와 데이터 준비

Python 3.9–3.12 환경에서 실행합니다. CUDA 학습 환경에서는 해당 환경에 맞는 PyTorch를 먼저 설치하세요.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python load_model.py --then-data  # 기존 모델/데이터가 없다면 실행
python download_kobart.py
python prepare_dataset.py
python audit_dataset.py --hash-audio
```

설정은 `configs/korean.yaml`에 있습니다. 설정의 상대 경로는 프로젝트 루트를 기준으로 해석합니다. 명령 예시는 프로젝트 루트에서 실행합니다.

원본 CSV의 `B1_text`–`B5_text`를 사용합니다. 영어/로마자 레이블과 설명문은 입력하지 않습니다. 같은 오디오의 다섯 정답을 각각 학습하며, 평가에서는 오디오당 한 번 생성해 다섯 정답과 비교합니다.

현재 로컬 데이터 검사에서 원본 7,962개 오디오/39,810개 한국어 정답의 토큰 복원을 확인했습니다. split 사이 동일 파일 68그룹을 발견했고, 중복 행 94개를 제외한 준비 데이터는 train 6,278개, valid 793개, test 797개입니다. 자세한 제거 내역은 `outputs/korean_data/preparation.json`에 있습니다.

`prepare_dataset.py`는 파일 내용의 SHA-256으로 split 사이 동일 파일을 찾아 test > valid > train 순서로 보존하고, 중복되는 다른 split의 행을 제외합니다. 원본 CSV와 오디오는 수정하지 않으며 `outputs/korean_data/`에 준비된 CSV와 `preparation.json`을 기록합니다. 바이트가 다른 재인코딩/부분 녹음은 이 검사로 찾을 수 없어 원본 녹음 메타데이터에 대한 추가 검토가 필요합니다.

원본 KoBART tokenizer는 일부 희귀 의성어를 `[UNK]`로 바꿉니다. 기존 BPE 어휘/merge를 유지하면서 고정된 Unicode 한글 음절·자모 범위를 추가하고 NFKC를 NFC로 바꿨습니다. 어휘는 30,000개에서 36,635개로 확장되며, 새 임베딩은 초기화하고 기존 사전학습 임베딩은 보존합니다. 데이터에서 새 어휘를 학습하지 않습니다. 모든 정답은 토큰화 후 원문 복원, 미등록 토큰, 길이 초과 검사를 통과해야 합니다. 반복, 내부 띄어쓰기와 낱자를 보존하며 마침표를 강제로 붙이지 않습니다.

오디오는 soundfile로 읽고 32kHz mono로 리샘플링한 뒤 10초까지 zero padding합니다. 10초를 넘는 파일은 자르지 않고 오류를 내므로 별도의 정답 정렬된 구간화가 필요합니다. 학습/추론은 같은 log-mel frontend를 사용합니다. HTSAT가 32번 반복한 동일 특징을 압축해 32개의 시간 토큰으로 decoder에 전달하고, 패딩 구간의 cross-attention을 마스킹합니다. HTSAT 내부 attention은 고정 길이 입력을 사용하므로 그 내부까지 패딩을 완전히 격리하지는 않습니다.

## 검증과 학습

```bash
python -m unittest discover -s tests -v
python train.py --smoke-test --device cpu
python train.py --overfit 32 --epochs 100 --device cuda
python train.py --device cuda
```

`--smoke-test`는 실제 가중치로 optimizer 1스텝, 생성, 저장과 오프라인 복원을 확인합니다. 성능 평가는 아닙니다. `--overfit 32`는 처음 32개 학습 오디오를 학습/평가에 같이 사용해 loss 감소와 출력 학습을 확인합니다. 두 모드의 결과는 각각 `smoke/`, `overfit_32/`에 분리됩니다. 완료된 폴더는 덮어쓰지 않으며 재실행 시 `--output-dir outputs/new_run`을 사용하세요.

기본 전체 학습은 20 epoch, 초기 3 epoch HTSAT 고정, 이후 마지막 HTSAT stage만 해제합니다. 동결된 stage와 batch-normalization 통계는 계속 고정합니다. HTSAT/decoder/projection의 학습률은 각각 3e-6/3e-5/1e-4입니다. batch size 4, gradient accumulation 4, CUDA에서 FP16 AMP를 사용합니다. 실제 GPU 메모리에 맞춰 설정을 조정하세요. GPU가 없는 `auto` 환경에서는 CPU를 사용합니다.

```bash
python train.py --resume outputs/korean_htsat_kobart/last.pt --device cuda
python evaluate.py --checkpoint outputs/korean_htsat_kobart/best.pt --split test
python infer.py --checkpoint outputs/korean_htsat_kobart/best.pt path/to/audio.wav
```

재개는 epoch 경계에서 `last.pt`의 optimizer/scheduler/scaler와 RNG 상태를 복원합니다. `best.pt`는 검증 CER이 가장 낮은 추론용 가중치입니다. 체크포인트와 같은 폴더의 `tokenizer/`를 함께 보관하세요. 추론은 원본 KoBART/HTSAT 다운로드 경로 없이 체크포인트와 tokenizer만으로 복원합니다. 다른 기계에서 평가할 때 `evaluate.py --config configs/korean.yaml`로 데이터 위치를 지정할 수 있습니다.

출력에는 `config.json`, `history.jsonl`, epoch별 예측 JSONL, `best.pt`, `last.pt`, `tokenizer/`가 포함됩니다. 현재 실행 환경에서 전체 CUDA 학습 성능은 아직 검증하지 않았습니다.

## 평가 정의

- `min_cer`: 공백을 제외한 한글 음절 문자열 기준으로 정답별 CER을 계산하고, 오디오별 최솟값을 구한 뒤 오디오에 대해 macro 평균합니다. 모델 선택 지표입니다.
- `min_cer_with_spaces`: 공백을 포함해 같은 방식으로 계산합니다.
- `min_jamo_cer`: 공백 제거 후 Unicode NFD 분해를 적용합니다. 호환 자모는 그대로 유지합니다.
- `exact_match` / `exact_match_no_spaces`: 다섯 정답 중 하나와 정확히 같은 출력의 비율입니다.

비율은 0–1 스케일이지만 CER은 삽입 오류가 많으면 1을 넘을 수 있습니다. 반복 의성어를 보존하기 위해 repetition penalty는 1.0, n-gram 반복 금지는 0입니다. 기존 평가와 비교할 때는 중복 제거된 split과 이 지표 정의를 동일하게 사용해야 합니다.
