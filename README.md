# BlazePose

## 설치 및 실행 방법

### 간편 실행 (Windows)
1. `run_blazepose.bat` 파일을 더블클릭하여 실행합니다.
2. 자동으로 가상환경이 활성화되고 필요한 패키지가 설치됩니다.
3. 실행할 애플리케이션을 선택하여 사용합니다.

### 수동 설정 (모든 OS)
1. 가상환경 생성:
   ```
   python -m venv blazepose_env
   ```

2. 가상환경 활성화:
   - Windows:
     ```
     blazepose_env\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source blazepose_env/bin/activate
     ```

3. 필요한 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

4. 애플리케이션 실행:
   ```
   python run.py
   ```

5. 사용이 끝나면 가상환경 비활성화:
   ```
   deactivate
   ```

## 시스템 요구사항
- Python 3.7 이상
- 웹캠
- 충분한 그래픽 성능 (실시간 3D 시각화에 필요)

## 주요 기능
- 웹캠을 통한 실시간 포즈 추적
- 2D 및 3D 시각화
- 관절 각도 및 자세 측정
- JSON 형식으로 데이터 저장
