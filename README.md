# XAI506 Final Project — 오프사이드 의심 장면 분석

월드컵 테마 기말 프로젝트. Hugging Face **SAM2** + **SmolVLM2** + **Qwen3-TTS**로 오프사이드 의심 장면을 분석하고, VAR 해설을 **음성**으로 들려줍니다.

---

## 프로젝트 구조

```
XAI506_Final-Project/
├── demo.py              # CLI 진입점
├── offside_core.py      # 피치 보정·기하 판정·시각화
├── sam2_helpers.py      # SAM2 추론 + PyQt5 클릭 픽커
├── smolvlm_helpers.py   # SmolVLM2 VAR 텍스트 (--explain)
├── tts_helpers.py       # Qwen3-TTS 해설 음성 (기본 포함)
├── utils.py             # 공통 유틸 (메모리 해제)
├── requirements.txt
├── imgs/                # 예제 이미지
└── outputs/             # 실행 결과 (자동 생성, git 제외)
```

---

## 설치

```bash
git clone https://github.com/KimDoYoung1997/XAI506_Final-Project.git
cd XAI506_Final-Project

conda create -n DL-final-term python=3.11 -y
conda activate DL-final-term
pip install -r requirements.txt
```

| 패키지 | 용도 |
|--------|------|
| `torch`, `torchvision` | SAM2 / SmolVLM2 추론 |
| `transformers`, `accelerate` | SAM2, SmolVLM2 |
| `numpy`, `pillow`, `matplotlib` | 이미지·시각화 |
| `scipy` | SAM2 마스크 후처리 |
| `PyQt5` | 마우스 클릭 픽커 (로컬 GUI 필요) |
| `num2words` | SmolVLM2 프로세서 의존성 |
| `qwen-tts`, `soundfile`, `torchaudio` | Qwen3-TTS |

**NVIDIA GPU(CUDA)** 환경이면 [PyTorch 시작하기](https://pytorch.org/get-started/locally/)에서 OS·CUDA에 맞는 `torch`·`torchvision`을 먼저 설치한 뒤 `pip install -r requirements.txt`를 실행하세요.

모델 가중치는 첫 실행 시 Hugging Face Hub에서 `~/.cache/huggingface/hub/`로 자동 다운로드됩니다.

### 설치 확인 (선택)

검증용 임시 환경 예시 (`xai506-verify`). **코드·requirements가 바뀐 뒤에는 반드시 `pip install -r requirements.txt`를 다시 실행**하세요.

```bash
conda create -n xai506-verify python=3.11 -y
conda activate xai506-verify
pip install -r requirements.txt
python demo.py --help
python -c "import qwen_tts; from transformers import Sam2Model; print('install ok')"
```

`install ok`가 출력되면 SAM2·Qwen3-TTS 의존성이 맞게 설치된 것입니다.

검증이 끝나면 임시 환경을 지워도 됩니다:

```bash
conda deactivate
conda env remove -n xai506-verify -y
```

본인 개발 환경(`DL-final-term` 등)에서도 같은 오류가 나면:

```bash
conda activate DL-final-term   # 본인 환경 이름
pip install -r requirements.txt
```

---

## 실행

기본값: `imgs/offside.png` + VAR 해설(SmolVLM2 + Qwen3-TTS, **한국어**, speaker `Sohee`).

```bash
python demo.py
python demo.py --no-explain          # SAM2 판정만
python demo.py --image imgs/offside.png
```

### 파이프라인 (3 FM)

| 단계 | 모델 | 역할 |
|------|------|------|
| 1–4 | **SAM2** + 클릭 | 선수 마스크 + 기하 오프사이드 판정 |
| 5 (기본) | **SmolVLM2** | 오버레이 이미지 → VAR 텍스트 (`Analysis` + `Broadcast`) |
| 6 (기본) | **Qwen3-TTS** | `Broadcast` 문장 → 해설 음성 WAV |

### 입력 순서 (4단계)

창이 뜨면 **좌클릭** 후 **Enter**로 확정합니다. 공은 사용하지 않습니다.

| 단계 | 내용 |
|------|------|
| 1 | 잔디 줄무늬 경계 **2점** — 골라인과 평행한 방향 보정 |
| 2 | 골대/골 쪽 **1점** — 공격 방향 지정 |
| 3 | 패스 받을 **공격수** 클릭 → SAM2 마스크 (빨강) |
| 4 | **최후방 수비수** 클릭 → SAM2 마스크 (파랑) |

### 출력 (`outputs/`)

| 파일 | 설명 |
|------|------|
| `<이미지>_offside.png` | 마스크 오버레이 + 오프사이드 라인 |
| `<이미지>_player.png` | 공격수·수비수 ROI crop |
| `<이미지>_report.txt` | SmolVLM2 VAR 텍스트 (기본 포함) |
| `<이미지>_commentary.wav` | Qwen3-TTS 해설 (Mac: `afplay` 자동 재생) |

### 디바이스·TTS 옵션 (선택)

```bash
SAM2_DEVICE=mps python demo.py
SMOLVLM_DEVICE=mps python demo.py
QWEN_TTS_DEVICE=cpu python demo.py          # Mac 권장
QWEN_TTS_SPEAKER=Ryan python demo.py        # 영어 speaker (언어는 기본 Korean)
QWEN_TTS_LANGUAGE=English python demo.py    # TTS 언어 강제 (기본: Korean)
QWEN_TTS_INSTRUCT="차분하게 VAR 심판처럼 말하세요." python demo.py
```

---

## 오프사이드 판별 원리 (요약)

카메라가 비스듬하기 때문에 **골라인과 평행한 기준선**이 필요합니다.

1. 잔디 줄무늬 2점 → `pitch_dir`
2. 골 방향 1점 → `goal_dir`
3. 최후방 수비수 마스크에서 골 방향 **최전방 픽셀**을 지나는 평행선 = 오프사이드 라인
4. 공격수 마스크의 최전방 픽셀이 그 라인보다 앞이면 **오프사이드 포지션**

```
오프사이드 포지션 ⟺ 공격수 몸(마스크) 중 골에 가장 가까운 픽셀이 수비수의 그 픽셀보다 앞
```

※ 발만이 아니라 SAM2 **전체 마스크** 픽셀을 사용합니다.  
※ 본 데모는 수비 라인 기준만 검사합니다 (실제 Law 11은 패스 순간 공 위치도 포함).

---

## 자주 겪는 이슈

- **Qt 창이 안 뜸**: SSH·원격 서버 등 헤드리스 환경에서는 PyQt5 GUI가 동작하지 않습니다.
- **`transformers` 버전**: `transformers>=4.50.0` 이상 필요 (SAM2 API).
- **Mac TTS**: Qwen3-TTS는 **CPU** 기본 권장 (`QWEN_TTS_DEVICE=cpu`). 기본 **한국어** + speaker `Sohee`.
- **모델 다운로드 실패**: 네트워크·HF Hub 접근 확인 또는 `huggingface-cli login`.
