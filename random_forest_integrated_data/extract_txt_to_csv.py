import csv
import os
import glob
from typing import Optional, Tuple, List

# ----------------------------------------------------------------------
# 1. 데이터 추출 함수 (이전과 동일하게 유지)
# ----------------------------------------------------------------------
def extract_xyz(line: str) -> Optional[Tuple[int, int, int]]:
    """
    라인에서 센서 데이터(x, y, z)를 추출합니다.
    예시: r,39534,...,392/-440/-84,...,#
    """
    line = line.strip()
    
    if not line or not line.startswith("r,"):
        return None
    
    fields = line.split(",")
    
    # 7번째 필드(인덱스 6)에 xyz 데이터가 있다고 가정
    if len(fields) < 7:
        return None
    
    xyz_field = fields[6]
    parts = xyz_field.split("/")
    
    if len(parts) != 3:
        return None
    
    try:
        x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
        return x, y, z
    except ValueError:
        return None

# ----------------------------------------------------------------------
# 2. 파일 변환 메인 함수
# ----------------------------------------------------------------------
def convert_txt_to_individual_csv(log_directory: str, output_directory: str):
    """
    지정된 폴더의 모든 TXT 파일을 개별 CSV 파일로 변환하여 
    지정된 출력 폴더에 저장합니다.
    """
    
    # 1. 입력 폴더에서 모든 TXT 파일 목록을 가져옵니다.
    input_file_list = glob.glob(os.path.join(log_directory, "*.txt"))
    
    print(f"검색 폴더: {log_directory}")
    print(f"검색된 TXT 파일 수: {len(input_file_list)}개")
    
    if not input_file_list:
        print("\n" + "="*50)
        print("경고: 지정된 폴더에서 .txt 파일을 찾을 수 없습니다.")
        print("="*50)
        return

    # 2. 출력 폴더가 없으면 생성합니다.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"출력 폴더 생성: '{output_directory}'")
    
    total_data_extracted = 0
    total_files_processed = 0

    print("\n" + "="*50)
    print(f"CSV 변환 시작 (출력 경로: {os.path.abspath(output_directory)})")
    print("="*50)

    # 3. 각 TXT 파일별로 순회하며 CSV를 생성합니다.
    for input_file_path in input_file_list:
        # 파일 이름(예: 1.txt)을 가져와 확장자를 .csv로 변경합니다. (예: 1.csv)
        base_name = os.path.basename(input_file_path) # '1.txt'
        file_name_without_ext = os.path.splitext(base_name)[0] # '1'
        output_csv_filename = file_name_without_ext + ".csv" # '1.csv'
        
        # CSV가 저장될 최종 경로를 만듭니다.
        output_csv_path = os.path.join(output_directory, output_csv_filename)
        
        data_in_file = 0

        try:
            # TXT 파일 열기
            with open(input_file_path, 'r', encoding='utf-8') as txtfile:
                # CSV 파일 열기 (개별 파일마다 새로 생성)
                with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # CSV 헤더 작성
                    writer.writerow(['x', 'y', 'z'])
                    
                    for line in txtfile:
                        data = extract_xyz(line)
                        
                        if data is not None:
                            writer.writerow(data)
                            data_in_file += 1
            
            print(f"✅ 성공: '{base_name}' -> '{output_csv_filename}' (총 {data_in_file}개 레코드)")
            total_data_extracted += data_in_file
            total_files_processed += 1

        except Exception as e:
            print(f"❌ 오류: '{base_name}' 파일 처리 중 예외 발생: {e}")
            
    print("\n" + "="*50)
    print(f"총 {total_files_processed}개 파일 처리 완료.")
    print(f"전체 추출된 데이터 레코드 수: {total_data_extracted}개")
    print(f"결과는 '{output_directory}' 폴더에 저장되었습니다.")
    print("="*50)


if __name__ == "__main__":
    # --- 설정: 이 부분만 수정하시면 됩니다. ---
    
    # 1. TXT 파일들이 있는 경로
    # 사용자 지정 경로: C:\ML_gipsy\ML25_Gipsy\random_forest_integrated_data\Raw Data\demo
    INPUT_LOG_DIRECTORY = r"C:\ML_gipsy\ML25_Gipsy\random_forest_integrated_data\Raw Data\demo"
    
    # 2. 변환된 CSV 파일들을 저장할 출력 폴더 이름
    # 스크립트 파일이 있는 위치에 'csv_output'이라는 폴더가 생성됩니다.
    OUTPUT_CSV_DIRECTORY = "csv_output"
    
    # 3. 함수 실행
    convert_txt_to_individual_csv(INPUT_LOG_DIRECTORY, OUTPUT_CSV_DIRECTORY)