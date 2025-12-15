# Random Forest
## A. 프로젝트 파일 구조
•	1_preprocessing.py: 데이터 로딩, 전처리, 증강\
•	feature_engineering.py: 17차원 특징 추출\
•	3_model_training.py: Random Forest 학습 및 저장\
•	4_main.py: 전체 파이프라인 통합 실행\
•	5_classifier_test.py: 테스트 데이터 평가 (CSV 입력)\
•	run_all_RF.py: 배치 추론 (TXT 입력)\
•	generate_testdata_ver2.py: 합성 테스트 데이터 생성\
---
## B. 실행 순서
1.	학습: python 4_main.py
2.	테스트 데이터 생성: python generate_testdata_ver2.py
3.	평가: python 5_classifier_test.py
4.	배치 추론: python run_all_RF.py
