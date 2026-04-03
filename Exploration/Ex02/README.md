# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 송세미
- 리뷰어 : 김도현

# PRT(Peer Review Template)
- [ ] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부

ResNet 모델 구현
<img width="426" height="179" alt="Image" src="https://github.com/user-attachments/assets/1a8e6c0e-b78d-414a-b872-ac6e26a547aa" />
ResNet-34와 PlainNet-34 accuracy 비교
<img width="426" height="250" alt="Image" src="https://github.com/user-attachments/assets/cb9a5f55-5c89-4ad2-bfbe-bf14fafdb4a8" />
<img width="426" height="250" alt="Image" src="https://github.com/user-attachments/assets/07f2553f-61d8-40f2-ae17-ecd8530a3d12" />

- [ ] **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

<img width="573" height="163" alt="Image" src="https://github.com/user-attachments/assets/79072bcc-2556-4997-b95a-91363ad776a4" />

<img width="685" height="458" alt="Image" src="https://github.com/user-attachments/assets/a1e7b17e-fbaf-41b2-bf4b-56d3242a39fa" />
        
        
- [ ] **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나 새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

CUDA Driver/Library Link Error 해결 시도     
<img width="724" height="266" alt="Image" src="https://github.com/user-attachments/assets/54ff024f-651d-43a0-a815-d830d4cfafb5" />
        
- [ ] **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해 배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

<img width="763" height="332" alt="Image" src="https://github.com/user-attachments/assets/2b6cabae-1a5f-4377-ac57-8cbb468dbea9" />

- [ ] **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

# 회고(참고 링크 및 코드 개선)
\`\`\`
- Tensorflow를 활용해 ResNet의 skip connection과 블록 구조를 체계적으로 잘 구현
- 코드의 중복을 최소화했고, 전체적인 흐름이 간결했음
- 학습 결과에 대한 분석도 테이블로 정리하여 수치 비교하기가 편했고 모델의 핵심 내용(skip connection)과 함께 결과 분석을 잘 정리했음
\`\`\`
