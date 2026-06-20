# XAI506 Final Project — 오프사이드 의심 장면 분석

월드컵 테마 기말 프로젝트. Hugging Face **SAM2**로 선수·공을 세그멘테이션하고, 잔디 줄무늬 기준선 보정으로 **오프사이드 포지션**을 근사 판별하는 `demo.py`입니다.

---

## 프로젝트 구조

```
XAI506_Final-Project/
├── demo.py              # 오프사이드 데모 (메인)
├── sam2_helpers.py      # SAM2 추론 + PyQt5 클릭 픽커
├── requirements.txt
├── imgs/
│   └── offside.png      # 기본 예제 이미지
├── outputs/             # 실행 결과 저장 (자동 생성)
└── scripts/             # 중간고사 예제 노트북 (참고용)
```

---

## 1. 저장소 복제

```bash
git clone https://github.com/KimDoYoung1997/XAI506_Final-Project.git
cd XAI506_Final-Project
```

---

## 2. Conda 설치

아직 Conda가 없다면 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 또는 [Anaconda](https://www.anaconda.com/download)를 설치합니다.

```bash
conda --version
```

---

## 3. 가상환경 만들기·활성화

```bash
conda create -n DL-final-term python=3.11 -y
conda activate DL-final-term
```

프롬프트 앞에 `(DL-final-term)`이 보이면 활성화된 상태입니다.

---

## 4. 패키지 설치

프로젝트 루트에서:

```bash
pip install -r requirements.txt
```

| 패키지 | 용도 |
|--------|------|
| `torch`, `torchvision` | SAM2 추론 (CUDA / MPS / CPU) |
| `transformers`, `accelerate` | `Sam2Model` / `Sam2Processor` |
| `numpy`, `pillow`, `matplotlib` | 이미지 입출력·결과 시각화 |
| `scipy` | 마스크 후처리 |
| `PyQt5` | 마우스 클릭 픽커 (데스크톱 GUI) |

**NVIDIA GPU(CUDA)** 환경이면 [PyTorch 시작하기](https://pytorch.org/get-started/locally/)에서 OS·CUDA에 맞는 `torch`·`torchvision`을 먼저 설치한 뒤 `pip install -r requirements.txt`를 실행하는 편이 안전합니다.

모델 가중치(`facebook/sam2.1-hiera-base-plus`)는 첫 실행 시 Hugging Face Hub에서 `~/.cache/huggingface/hub/`로 자동 다운로드됩니다.

---

## 5. `demo.py` 실행

```bash
conda activate DL-final-term
python demo.py
```

다른 이미지를 쓰려면:

```bash
python demo.py --image imgs/offside.png
```

### 입력 순서 (4단계)

창이 뜨면 아래 순서로 **좌클릭** 후 **Enter**로 확정합니다. **공은 사용하지 않습니다** (SAM2 2회만 실행).

| 단계 | 내용 |
|------|------|
| 1 | 잔디 줄무늬 경계 **2점** — 골라인과 평행한 방향 보정 |
| 2 | 골대/골 쪽 **1점** — 공격 방향 지정 |
| 3 | 패스 받을 **공격수** 클릭 → SAM2 마스크 (빨강) |
| 4 | **최후방 수비수** 클릭 → SAM2 마스크 (파랑) |

### 출력

- 터미널: 오프사이드 포지션 여부, 공격수가 수비 라인보다 골에 얼마나 가까운지
- `outputs/<이미지명>_offside.png`: 마스크 오버레이 + 오프사이드 라인 시각화
- matplotlib 창: 결과 미리보기

### 디바이스 강제 (선택)

```bash
SAM2_DEVICE=mps python demo.py    # Apple Silicon
SAM2_DEVICE=cuda python demo.py   # NVIDIA GPU
SAM2_DEVICE=cpu python demo.py    # CPU
```

---

## 6. 오프사이드 판별 원리 (요약)

두 선수 마스크만으로는 판별할 수 없습니다. 카메라가 비스듬하기 때문에 **골라인과 평행한 기준선**이 필요합니다.

1. 잔디 줄무늬(골라인과 평행) 2점 → `pitch_dir`
2. 골 방향 1점 → `goal_dir`
3. 최후방 수비수 마스크에서 **골 방향으로 가장 앞선 픽셀**을 지나는 `pitch_dir` 평행선 = 오프사이드 라인
4. 공격수 마스크에서도 **골 방향으로 가장 앞선 픽셀**이 그 라인보다 앞이면 **오프사이드 포지션**

```
오프사이드 포지션 ⟺ 공격수 몸(마스크) 중 골에 가장 가까운 픽셀이 수비수의 그 픽셀보다 앞
```

※ 발만 보지 않습니다. SAM2로 마스킹한 **선수 전체 픽셀**을 `goal_dir`에 투영해 최댓값을 씁니다.

※ 본 데모는 **수비 라인 기준**만 검사합니다. 실제 규칙(Law 11)은 패스 순간 **공** 위치도 함께 봅니다. 단안 2D 근사이며, VAR은 다중 카메라·메트릭 좌표를 사용합니다.

---

## 7. 자주 겪는 이슈

- **Qt 창이 안 뜸**: SSH·원격 서버 등 헤드리스 환경에서는 PyQt5 GUI가 동작하지 않습니다. 로컬 데스크톱에서 실행하세요.
- **`transformers` 버전 오류**: `transformers>=4.50.0` 이상인지 확인하세요. SAM2 API는 비교적 최근 버전이 필요합니다.
- **모델 다운로드 실패**: 네트워크·HF Hub 접근을 확인하거나 `huggingface-cli login` 후 재시도하세요.

---

## 8. 참고: `scripts/` 노트북 (중간고사 예제)

HF 파운데이션 모델 실험용 노트북입니다. Jupyter로 실행하려면 선택 패키지를 추가로 설치하세요.

```bash
pip install notebook ipykernel num2words
jupyter notebook scripts/
```

- `01_hf_sam2.ipynb` — SAM2 포인트 세그멘테이션
- `02_hf_depth_anything.ipynb` — Depth Anything V2
- `03_hf_smolvlm.ipynb` — SmolVLM2
- `04_hf_qwen3_vl.ipynb` — Qwen3-VL
